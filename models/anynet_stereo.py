"""AnyStereoNet - AnyNet-style coarse-to-fine stereo, no 3D convolutions.

Follows "Anytime Stereo Image Depth Estimation on Mobile Devices" (Wang et al.,
ICRA 2019). The expensive part of FastStereoNet is CostAggregation3D, a 3D-conv
hourglass over a (B,G,D,H,W) volume; AnyNet's idea is that you don't need it.
Instead:

  stage 1 (1/16) - full disparity sweep where the disparity axis is smallest
                   (max_disp/16 levels), scored by *2D* convs over the volume
                   flattened to (B, G*D, H, W), then soft-argmin.
  stage 2 (1/8)  - don't re-search the full range: warp the right features by
                   the upsampled stage-1 disparity and search a narrow band of
                   residual offsets around it. Cheap, because the band is a
                   handful of hypotheses instead of the whole range.
  stage 3 (1/4)  - same again, narrower still.
  refinement     - the shared edge-aware full-resolution residual head.

Each stage emits a disparity, so the network is "anytime": stopping after
stage 1 or 2 gives a usable (coarser) result at a fraction of the cost. All
three are returned in "aux", so whichever --loss is selected supervises every
stage (losses.py), exactly as it does for the other models.

Disparities are in FULL-RESOLUTION pixel units everywhere (see common.py).
Same interface as FastStereoNet: forward(left, right) -> {"disp", "aux"}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (conv_bn_relu, ResBlock, gwc_volume, soft_argmin,
                     DisparityRefinement, upsample_disp)


class PyramidFeatures(nn.Module):
    """Siamese encoder returning features at 1/4, 1/8 and 1/16.

    Deliberately thinner than common.FeatureExtractor: AnyNet spends its
    budget on the coarse-to-fine disparity pyramid rather than on features.
    """

    def __init__(self, ch=24):
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu(3, 16, stride=2), ResBlock(16))    # 1/2
        self.down4 = nn.Sequential(conv_bn_relu(16, ch, stride=2), ResBlock(ch))  # 1/4
        self.down8 = nn.Sequential(conv_bn_relu(ch, ch, stride=2), ResBlock(ch))  # 1/8
        self.down16 = nn.Sequential(conv_bn_relu(ch, ch, stride=2), ResBlock(ch))  # 1/16

    def forward(self, x):
        x4 = self.down4(self.stem(x))
        x8 = self.down8(x4)
        return x4, x8, self.down16(x8)


class Cost2D(nn.Module):
    """Score a (B,G,D,H,W) correlation volume with 2D convs instead of 3D.

    Flattening the group and disparity axes into channels lets plain 2D convs
    do the aggregation - this is what removes AnyNet's need for 3D convs.
    """

    def __init__(self, groups, num_disp, hidden=32):
        super().__init__()
        self.num_disp = num_disp
        self.net = nn.Sequential(
            conv_bn_relu(groups * num_disp, hidden),
            conv_bn_relu(hidden, hidden),
            nn.Conv2d(hidden, num_disp, 3, 1, 1))

    def forward(self, vol):
        B, G, D, H, W = vol.shape
        return self.net(vol.view(B, G * D, H, W))


class ResidualStage(nn.Module):
    """Refine a disparity estimate by searching a narrow band of residual
    offsets around it: warp right features by the current disparity plus each
    candidate offset, correlate, and soft-select. Offsets are full-res pixels."""

    def __init__(self, offsets, stride, groups=8, hidden=32):
        super().__init__()
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32))
        self.stride = stride
        self.groups = groups
        k = len(offsets)
        self.net = nn.Sequential(
            conv_bn_relu(k * groups, hidden),
            nn.Conv2d(hidden, k, 3, 1, 1))

    def forward(self, fl, fr, disp):
        B, C, h, w = fl.shape
        cpg = C // self.groups
        K = self.offsets.numel()

        # All K hypotheses in one grid_sample rather than K sequential ones:
        # these models are small enough that per-kernel launch overhead, not
        # FLOPs, sets the latency on a desktop GPU.
        ys, xs = torch.meshgrid(
            torch.arange(h, device=fl.device, dtype=torch.float32),
            torch.arange(w, device=fl.device, dtype=torch.float32), indexing="ij")
        cand = disp + self.offsets.view(1, -1, 1, 1)                  # (B,K,h,w)
        gx = 2.0 * (xs - cand / self.stride) / (w - 1) - 1.0
        gy = (2.0 * ys / (h - 1) - 1.0).expand(B, K, h, w)
        grid = torch.stack([gx, gy], -1).view(B * K, h, w, 2).to(fl.dtype)

        fr_rep = fr.unsqueeze(1).expand(B, K, C, h, w).reshape(B * K, C, h, w)
        fr_w = F.grid_sample(fr_rep, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)
        fl_rep = fl.unsqueeze(1).expand(B, K, C, h, w)
        corr = (fl_rep * fr_w.view(B, K, C, h, w)) \
            .view(B, K, self.groups, cpg, h, w).mean(3)               # (B,K,G,h,w)

        prob = F.softmax(self.net(corr.reshape(B, K * self.groups, h, w)), 1)
        return (prob * cand).sum(1, keepdim=True).clamp(min=0.0)


class AnyStereoNet(nn.Module):
    STRIDE = 16  # coarse disparity scale (stage 1)

    def __init__(self, max_disp=128, feat_ch=24, groups=8):
        super().__init__()
        assert max_disp % self.STRIDE == 0
        self.max_disp = max_disp
        self.num_disp16 = max_disp // self.STRIDE
        self.groups = groups
        self.features = PyramidFeatures(feat_ch)
        self.cost16 = Cost2D(groups, self.num_disp16)
        self.stage8 = ResidualStage([-4, -2, -1, 0, 1, 2, 4], stride=8, groups=groups)
        self.stage4 = ResidualStage([-2, -1, 0, 1, 2], stride=4, groups=groups)
        self.refine = DisparityRefinement(max_disp)

    def forward(self, left, right):
        f4l, f8l, f16l = self.features(left)
        f4r, f8r, f16r = self.features(right)

        vol = gwc_volume(f16l, f16r, self.num_disp16, self.groups)
        disp16 = soft_argmin(self.cost16(vol)) * self.STRIDE      # full-res units
        disp8 = self.stage8(f8l, f8r, upsample_disp(disp16, 2))
        disp4 = self.stage4(f4l, f4r, upsample_disp(disp8, 2))
        disp = self.refine(upsample_disp(disp4, 4), left)

        return {"disp": disp, "aux": [(disp16, 16), (disp8, 8), (disp4, 4)]}
