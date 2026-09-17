"""MobileStereoNet - MobileNetV2-flavored lightweight single-pair baseline.

Same siamese / group-wise-correlation / soft-argmin / refine pipeline as
FastStereoNet (see baseline_net.py), but the two stages that run densely over
every pixel of the image - the feature extractor and the full-resolution
refinement head - are built from MobileNetV2 inverted-residual (expand ->
depthwise -> project) blocks instead of full 3x3 convolutions. The 1/8-scale
cost aggregation is already small (16 disparity levels x 8 groups at 1/8
resolution) so it's reused unchanged from common.py. No temporal information,
same interface as FastStereoNet: forward(left, right).
"""

import torch.nn as nn

from .common import (MobileFeatureExtractor, MobileRefinement, gwc_volume,
                     soft_argmin, CostAggregation3D, FxbConditioning,
                     upsample_disp)


class MobileStereoNet(nn.Module):
    STRIDE = 8  # coarse disparity scale

    def __init__(self, max_disp=128, feat_ch=32, groups=8):
        super().__init__()
        assert max_disp % self.STRIDE == 0
        self.max_disp = max_disp
        self.num_disp8 = max_disp // self.STRIDE
        self.groups = groups
        self.features = MobileFeatureExtractor(feat_ch)
        self.aggregation = CostAggregation3D(groups)
        self.refine = MobileRefinement(max_disp)

    def forward(self, left, right):
        _, fl8 = self.features(left)
        _, fr8 = self.features(right)
        vol = gwc_volume(fl8, fr8, self.num_disp8, self.groups)
        cost = self.aggregation(vol)
        disp8 = soft_argmin(cost) * self.STRIDE          # full-res units, 1/8 grid
        disp_up = upsample_disp(disp8, self.STRIDE)
        disp = self.refine(disp_up, left)
        return {"disp": disp, "aux": [(disp8, self.STRIDE)]}


class MobileStereoNetFxb(MobileStereoNet):
    """MobileStereoNet conditioned on fxb (focal x baseline) - same idea as
    FastStereoNetFxb (baseline_net.py), applied to the MobileNetV2 siamese
    features before the cost volume is built."""

    def __init__(self, max_disp=128, feat_ch=32, groups=8):
        super().__init__(max_disp=max_disp, feat_ch=feat_ch, groups=groups)
        self.condition = FxbConditioning(feat_ch)

    def forward(self, left, right, fxb):
        _, fl8 = self.features(left)
        _, fr8 = self.features(right)
        fl8 = self.condition(fl8, fxb)
        fr8 = self.condition(fr8, fxb)
        vol = gwc_volume(fl8, fr8, self.num_disp8, self.groups)
        cost = self.aggregation(vol)
        disp8 = soft_argmin(cost) * self.STRIDE
        disp_up = upsample_disp(disp8, self.STRIDE)
        disp = self.refine(disp_up, left)
        return {"disp": disp, "aux": [(disp8, self.STRIDE)]}
