"""FastStereoNet - the lightweight single-pair baseline (~0.3M params).

StereoNet/GwcNet-flavored: siamese features at 1/8, group-wise correlation
cost volume, a small 3D-conv hourglass, soft-argmin, then one edge-aware
refinement at full resolution. No temporal information.
"""

import torch.nn as nn

from .common import (FeatureExtractor, gwc_volume, soft_argmin,
                     CostAggregation3D, DisparityRefinement, upsample_disp)


class FastStereoNet(nn.Module):
    STRIDE = 8  # coarse disparity scale

    def __init__(self, max_disp=128, feat_ch=32, groups=8):
        super().__init__()
        assert max_disp % self.STRIDE == 0
        self.max_disp = max_disp
        self.num_disp8 = max_disp // self.STRIDE
        self.groups = groups
        self.features = FeatureExtractor(feat_ch)
        self.aggregation = CostAggregation3D(groups)
        self.refine = DisparityRefinement(max_disp)

    def forward(self, left, right):
        _, fl8 = self.features(left)
        _, fr8 = self.features(right)
        vol = gwc_volume(fl8, fr8, self.num_disp8, self.groups)
        cost = self.aggregation(vol)
        disp8 = soft_argmin(cost) * self.STRIDE          # full-res units, 1/8 grid
        disp_up = upsample_disp(disp8, self.STRIDE)
        disp = self.refine(disp_up, left)
        return {"disp": disp, "aux": [(disp8, self.STRIDE)]}
