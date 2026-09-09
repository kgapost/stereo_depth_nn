"""YoloStereoNet - YOLO26 encoder/decoder around a stereo cost volume.

Motivation. On this project's rig (fx*b ~ 23 px*m) disparity is 2.3 px at 10 m
and 0.77 px at 30 m, so beyond roughly 10 m there is almost no triangulation
signal left and the estimate has to lean on a learned prior instead. That
argues for a much stronger feature extractor than the three-stage conv stack in
common.py, and YOLO26's is both stronger per FLOP and already a known quantity
for Jetson export.

What is borrowed from ultralytics' monocular depth model (docs.ultralytics.com/
tasks/depth) and what is not. YOLO26-depth takes one image and regresses depth
directly; we take two and must match between them, so its `Depth` head is
replaced. Borrowed:

  * the backbone + PAN neck, parsed from yolo26_stereo.yaml with ultralytics'
    own parse_model rather than re-implemented (see that file's header);
  * the head's decoder *shape* - project each pyramid level to a common width,
    fuse coarse-to-fine with bilinear upsample + add + two convs, then a
    transposed conv to 1/4 resolution;
  * the unbounded log output, adapted: YOLO26-depth emits exp(logit) as an
    absolute depth, we emit exp(residual) as a *multiplicative correction* to
    the cost volume's disparity.

Kept from this repo's own models: the group-wise correlation volume, the 3D
aggregation hourglass, soft-argmin and the full-resolution refinement head, all
imported unchanged from common.py.

Why a multiplicative correction. Every other model here refines additively
(F.relu(disp + correction)), which spreads its capacity uniformly in absolute
disparity and so is biased toward the near field - the same bias the disparity
smooth-L1 loss has (see losses.py). disp * exp(r) is uniform in *relative*
depth instead, which is what abs-rel measures and what the log losses optimise.
It also softens the max_disp ceiling: the cost volume still bounds the search,
but the output is no longer clipped to it.

Disparities are in FULL-RESOLUTION pixel units everywhere (see common.py).
Same interface as FastStereoNet: forward(left, right) -> {"disp", "aux"}.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import parse_model

from .common import (CostAggregation3D, DisparityRefinement, gwc_volume,
                     soft_argmin, upsample_disp)

_YAML = os.path.join(os.path.dirname(__file__), "yolo26_stereo.yaml")

# Layer indices in yolo26_stereo.yaml (documented in its header).
_P3_BACKBONE = 4            # 1/8 features, where the cost volume is built
_NECK_OUT = (16, 19, 22)    # P3 / P4 / P5 neck outputs feeding the decoder
_RIGHT_LAYERS = _P3_BACKBONE + 1  # the right view is only ever used for matching


class YoloStereoNet(nn.Module):
    STRIDE = 8  # coarse disparity scale

    def __init__(self, max_disp=128, scale="n", groups=8, c_mid=64,
                 full_res_refine=True):
        super().__init__()
        assert max_disp % self.STRIDE == 0
        self.max_disp = max_disp
        self.num_disp8 = max_disp // self.STRIDE
        self.groups = groups

        with open(_YAML) as f:
            cfg = yaml.safe_load(f)
        cfg["scale"] = scale
        layers, save = parse_model(cfg, ch=3, verbose=False)
        self.layers = layers
        # parse_model's save list only covers the skip connections the neck
        # itself consumes; the outputs *we* read are not among them, and
        # anything not saved is dropped during the forward pass.
        self.save = set(save) | {_P3_BACKBONE, *_NECK_OUT}

        p3_ch, neck_ch = self._probe_channels()
        assert p3_ch % groups == 0, (
            f"P3 has {p3_ch} channels, not divisible by groups={groups}")

        # Decoder: ultralytics' Depth head shape, coarsest level first.
        self.proj = nn.ModuleList(Conv(c, c_mid, k=1) for c in neck_ch)
        self.fuse = nn.ModuleList(
            nn.Sequential(Conv(c_mid, c_mid, k=3), Conv(c_mid, c_mid, k=3))
            for _ in neck_ch[:-1])
        # The decoder is monocular up to this point - it sees only the left
        # image. This is where the stereo estimate enters it.
        self.inject = Conv(c_mid + 1, c_mid, k=3)
        self.head = nn.Sequential(
            Conv(c_mid, c_mid // 2, k=3),
            nn.ConvTranspose2d(c_mid // 2, c_mid // 2, kernel_size=2, stride=2),
            Conv(c_mid // 2, c_mid // 4, k=3),
            nn.Conv2d(c_mid // 4, 1, kernel_size=1),
        )
        # exp(0) = 1, so the decoder starts as the identity and the cost
        # volume's disparity is what the loss sees on step one.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.aggregation = CostAggregation3D(groups)
        self.refine = DisparityRefinement(max_disp) if full_res_refine else None

    @torch.no_grad()
    def _probe_channels(self):
        """Run a dummy tensor through to learn the P3 and neck widths.

        They depend on the compound scale, and reproducing parse_model's
        width/max_channels rounding here would just be a second implementation
        of it waiting to disagree.
        """
        was_training = self.training
        self.eval()
        y = self._run(torch.zeros(1, 3, 64, 64), len(self.layers))
        self.train(was_training)
        return y[_P3_BACKBONE].shape[1], tuple(y[i].shape[1] for i in _NECK_OUT)

    def _run(self, x, n_layers):
        """Forward the first `n_layers` parsed layers, honouring their `from`
        indices. Mirrors ultralytics' BaseModel._predict_once."""
        y = []
        for m in self.layers[:n_layers]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else \
                    [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return y

    def forward(self, left, right):
        h, w = left.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(
                f"YoloStereoNet needs image sizes that are multiples of 32 "
                f"(the backbone reaches 1/32), got {h}x{w}. The other models "
                f"only need multiples of 16, so a --crop that works for them "
                f"may not work here; 480x640 and 256x448 both do.")

        # The right view is only ever used to build the cost volume, so it
        # stops at P3 instead of running the whole backbone and neck.
        yl = self._run(left, len(self.layers))
        yr = self._run(right, _RIGHT_LAYERS)
        f8l, f8r = yl[_P3_BACKBONE], yr[_P3_BACKBONE]

        # Stereo: the geometry stays exactly as in the other models.
        vol = gwc_volume(f8l, f8r, self.num_disp8, self.groups)
        disp8 = soft_argmin(self.aggregation(vol)) * self.STRIDE  # full-res units

        # Decoder over the left image's pyramid, coarse to fine.
        feats = [self.proj[i](yl[j]) for i, j in enumerate(_NECK_OUT)]
        out = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="bilinear",
                                align_corners=True)
            out = self.fuse[i](out + feats[i])
        out = self.inject(torch.cat([out, disp8 / self.max_disp], 1))

        residual = self.head(out)                                  # (B,1,H/4,W/4)
        disp4 = upsample_disp(disp8, 2) * torch.exp(residual.clamp(-1.0, 1.0))
        disp = upsample_disp(disp4, 4)
        if self.refine is not None:
            disp = self.refine(disp, left)

        return {"disp": disp, "aux": [(disp8, 8), (disp4, 4)]}
