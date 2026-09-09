"""Shared building blocks for the stereo networks.

Convention used everywhere: disparities are expressed in FULL-RESOLUTION pixel
units regardless of the feature-map scale; sampling code divides by the stride.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, k=3, stride=1, dilation=1):
    pad = dilation * (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride, pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv1 = conv_bn_relu(ch, ch, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + x, inplace=True)


class FeatureExtractor(nn.Module):
    """Siamese encoder shared by left/right. Returns 1/4 and 1/8 features."""

    def __init__(self, ch=32):
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu(3, 16, stride=2), ResBlock(16))       # 1/2
        self.down4 = nn.Sequential(conv_bn_relu(16, ch, stride=2),
                                   ResBlock(ch), ResBlock(ch))                        # 1/4
        self.down8 = nn.Sequential(conv_bn_relu(ch, ch, stride=2),
                                   ResBlock(ch), ResBlock(ch, dilation=2))            # 1/8

    def forward(self, x):
        x4 = self.down4(self.stem(x))
        return x4, self.down8(x4)


def conv_bn_relu6(in_ch, out_ch, k=3, stride=1):
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride, pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    """MobileNetV2 block: 1x1 expand -> 3x3 depthwise -> 1x1 project (linear).

    Residual add only when shape is preserved (stride 1, in_ch == out_ch).
    """

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4, dilation=1):
        super().__init__()
        hid = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hid, 1, bias=False),
                      nn.BatchNorm2d(hid), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hid, hid, 3, stride, dilation, dilation=dilation,
                      groups=hid, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class MobileFeatureExtractor(nn.Module):
    """Siamese encoder, MobileNetV2-flavored. Returns 1/4 and 1/8 features.

    Drop-in replacement for FeatureExtractor: same stem/down4/down8 shape and
    the same (x4, x8) return, but every stage after the first is an inverted-
    residual (expand -> depthwise -> project) block instead of a full 3x3
    ResBlock, for a fraction of the parameters and FLOPs.
    """

    def __init__(self, ch=32):
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu6(3, 16, stride=2),
                                  InvertedResidual(16, 16, expand_ratio=1))     # 1/2
        self.down4 = nn.Sequential(InvertedResidual(16, ch, stride=2),
                                   InvertedResidual(ch, ch))                     # 1/4
        self.down8 = nn.Sequential(InvertedResidual(ch, ch, stride=2),
                                   InvertedResidual(ch, ch),
                                   InvertedResidual(ch, ch, dilation=2))         # 1/8

    def forward(self, x):
        x4 = self.down4(self.stem(x))
        return x4, self.down8(x4)


class MobileRefinement(nn.Module):
    """Edge-aware residual refinement at image resolution, MobileNetV2-flavored.

    Drop-in replacement for DisparityRefinement. This stage runs over every
    pixel at full resolution, so unlike MobileFeatureExtractor's blocks (which
    run at 1/4-1/8 resolution, where a 4x channel expansion is cheap) these
    blocks use expand_ratio=1: expanding channels at full 480x640 res was
    blowing past 24GB at bs=16 (one intermediate tensor alone was ~1.2GB per
    block). expand_ratio=1 skips the expand conv, leaving a plain depthwise-
    separable block at a constant `ch` channels - a smaller correction
    function, but this stage only applies a residual correction on top of an
    already-computed coarse disparity, not primary disparity estimation.
    """

    def __init__(self, max_disp, ch=16):
        super().__init__()
        self.max_disp = max_disp
        self.head = conv_bn_relu6(4, ch)
        self.blocks = nn.Sequential(
            *[InvertedResidual(ch, ch, dilation=d, expand_ratio=1) for d in (1, 2, 4, 8, 1)])
        self.out = nn.Conv2d(ch, 1, 3, 1, 1)

    def forward(self, disp, img):
        x = torch.cat([disp / self.max_disp, img], 1)
        return F.relu(disp + self.out(self.blocks(self.head(x))))


def gwc_volume(fl, fr, num_disp, groups):
    """Group-wise correlation volume. fl/fr (B,C,H,W) -> (B,G,D,H,W).

    Disparity axis is in feature-pixel steps (multiply by stride for full-res).
    """
    B, C, H, W = fl.shape
    cpg = C // groups
    vol = fl.new_zeros(B, groups, num_disp, H, W)
    for d in range(num_disp):
        if d == 0:
            vol[:, :, d] = (fl * fr).view(B, groups, cpg, H, W).mean(2)
        else:
            vol[:, :, d, :, d:] = (fl[..., d:] * fr[..., :-d]) \
                .view(B, groups, cpg, H, W - d).mean(2)
    return vol


def soft_argmin(cost):
    """cost (B,D,H,W) scores -> expected disparity index (B,1,H,W)."""
    prob = F.softmax(cost, dim=1)
    disps = torch.arange(cost.shape[1], device=cost.device, dtype=cost.dtype)
    return (prob * disps.view(1, -1, 1, 1)).sum(1, keepdim=True)


class CostAggregation3D(nn.Module):
    """Small 3D-conv hourglass over (B,G,D,H,W) -> (B,D,H,W) scores.

    D, H, W must be even (crop sizes multiple of 16 at 1/8 scale).
    """

    def __init__(self, in_ch, ch=16):
        super().__init__()

        def c3d(i, o, s=1):
            return nn.Sequential(nn.Conv3d(i, o, 3, s, 1, bias=False),
                                 nn.BatchNorm3d(o), nn.ReLU(inplace=True))

        self.enc1 = c3d(in_ch, ch)
        self.enc2 = nn.Sequential(c3d(ch, ch * 2, s=2), c3d(ch * 2, ch * 2))
        self.up = nn.ConvTranspose3d(ch * 2, ch, 4, 2, 1, bias=False)
        self.up_bn = nn.Sequential(nn.BatchNorm3d(ch), nn.ReLU(inplace=True))
        self.out = nn.Conv3d(ch, 1, 3, 1, 1)

    def forward(self, vol):
        e1 = self.enc1(vol)
        e2 = self.enc2(e1)
        d = self.up_bn(self.up(e2))
        return self.out(d + e1).squeeze(1)


class DisparityRefinement(nn.Module):
    """Edge-aware residual refinement at image resolution (StereoNet-style)."""

    def __init__(self, max_disp, ch=16):
        super().__init__()
        self.max_disp = max_disp
        self.head = conv_bn_relu(4, ch)
        self.blocks = nn.Sequential(*[ResBlock(ch, dilation=d) for d in (1, 2, 4, 8, 1)])
        self.out = nn.Conv2d(ch, 1, 3, 1, 1)

    def forward(self, disp, img):
        x = torch.cat([disp / self.max_disp, img], 1)
        return F.relu(disp + self.out(self.blocks(self.head(x))))


def upsample_disp(disp, scale):
    """Bilinear spatial upsample; values stay in full-res units (no rescale)."""
    return F.interpolate(disp, scale_factor=scale, mode="bilinear", align_corners=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
