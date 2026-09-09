"""StereoConv3DNet - the brute-force spatiotemporal baseline (PAPER.md
Section 8 / Section 9). C3D/I3D-style: a siamese 3D-conv encoder over a
stacked frame window (no pose input, no recurrence), temporally collapsed
into 2D features, then the same group-wise correlation + 3D-conv hourglass +
soft-argmin + edge-aware refinement this repo's `FastStereoNet` uses.

The network itself is agnostic to the window length T - the temporal encoder
collapses whatever T it is given via a mean over the downsampled time axis
before matching happens. `StereoConv3DNet` (T=10, Section 8) and its "Fast"
sibling (T=3, Section 9) are therefore the same class, registered under two
`--model` names that only differ in the window length the dataset hands
them - not two different architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (gwc_volume, soft_argmin, CostAggregation3D,
                     DisparityRefinement, upsample_disp)


def conv3d_bn_relu(in_ch, out_ch, spatial_stride, time_stride):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, (time_stride, spatial_stride, spatial_stride),
                 1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class Siamese3DEncoder(nn.Module):
    """Shared 3D-conv stem down to 1/8 spatial resolution: spatial stride-2 in
    every block, temporal stride-2 in the last two (C3D/I3D-style)."""

    def __init__(self, ch=32):
        super().__init__()
        self.stem = conv3d_bn_relu(3, ch // 2, spatial_stride=2, time_stride=1)    # 1/2, T
        self.down4 = conv3d_bn_relu(ch // 2, ch, spatial_stride=2, time_stride=2)   # 1/4, T/2
        self.down8 = conv3d_bn_relu(ch, ch, spatial_stride=2, time_stride=2)        # 1/8, T/4

    def forward(self, x):
        return self.down8(self.down4(self.stem(x)))   # (B, ch, T', H/8, W/8)


class FxbConditioning(nn.Module):
    """Concatenates `fxb` as a constant feature map and projects back to `ch`
    channels - the same "condition on the calibration scalar" idea as
    `StereoConvNet` (Section 6), applied here at the collapsed-temporal
    bottleneck instead of a 2D U-Net bottleneck."""

    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(ch + 1, ch, 1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

    def forward(self, f, fxb):
        fxb_map = (fxb / 100.0).view(-1, 1, 1, 1).expand(-1, 1, *f.shape[2:])
        return self.proj(torch.cat([f, fxb_map], 1))


class StereoConv3DNet(nn.Module):
    """C3D/I3D-style spatiotemporal baseline: stack the last T frames, encode
    with 3D convolutions, collapse to 2D before matching, then reuse
    `FastStereoNet`'s group-wise correlation + 3D hourglass + refinement.
    No ego-motion input, no recurrent state - the brute-force comparison
    point for `TempoBandNet` (Section 10)."""

    STRIDE = 8

    def __init__(self, max_disp=128, feat_ch=32, groups=8):
        super().__init__()
        assert max_disp % self.STRIDE == 0
        self.max_disp = max_disp
        self.num_disp8 = max_disp // self.STRIDE
        self.groups = groups
        self.encoder = Siamese3DEncoder(feat_ch)
        self.condition = FxbConditioning(feat_ch)
        self.aggregation = CostAggregation3D(groups)
        self.refine = DisparityRefinement(max_disp)

    def forward(self, left, right, fxb):
        # left, right: (B, T, 3, H, W) -> (B, 3, T, H, W) for Conv3d
        l = left.permute(0, 2, 1, 3, 4)
        r = right.permute(0, 2, 1, 3, 4)
        fl = self.encoder(l).mean(2)   # collapse T' -> (B, ch, H/8, W/8)
        fr = self.encoder(r).mean(2)
        fl = self.condition(fl, fxb)
        fr = self.condition(fr, fxb)
        vol = gwc_volume(fl, fr, self.num_disp8, self.groups)
        cost = self.aggregation(vol)
        disp8 = soft_argmin(cost) * self.STRIDE       # full-res units, 1/8 grid
        disp_up = upsample_disp(disp8, self.STRIDE)
        disp = self.refine(disp_up, left[:, -1])       # condition on current frame only
        return {"disp": disp, "aux": [(disp8, self.STRIDE)]}
