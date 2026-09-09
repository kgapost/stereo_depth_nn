"""StereoConvNet - the "obvious first thing you'd try" 2D baseline (PAPER.md
Section 6). DispNetC-style: plain 1D dot-product correlation (no learned 3D
convolution), a 2D-conv U-Net over the cost volume treating disparity as
channels, and a learned (transposed-conv) upsample back to full resolution -
mostly convolution layers, no 3D conv, no recurrence, no attention.

Unlike `FastStereoNet` (Section 7), this model takes the calibration scalar
as an explicit network input - not just in the final `depth = fxb / disp`
conversion - so one trained model can be conditioned on whatever baseline is
given at inference time (Section 1.5). This codebase already threads `fxb`
(focal length x baseline) through every dataset/batch as the single
calibration-derived scalar disparity is computed from, so that is what gets
broadcast into the network here rather than a separately-plumbed raw
baseline distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import conv_bn_relu, ResBlock, soft_argmin, upsample_disp


class SiameseEncoder2D(nn.Module):
    """Shared 2D-conv stem down to 1/4 res, keeping 1/2-res features for the
    upsample stage's skip connection."""

    def __init__(self, ch=64):
        super().__init__()
        self.down2 = nn.Sequential(conv_bn_relu(3, ch // 2, stride=2),
                                   ResBlock(ch // 2))                  # 1/2
        self.down4 = nn.Sequential(conv_bn_relu(ch // 2, ch, stride=2),
                                   ResBlock(ch), ResBlock(ch))          # 1/4

    def forward(self, x):
        f2 = self.down2(x)
        f4 = self.down4(f2)
        return f2, f4


def correlation_1d(fl, fr, num_disp):
    """Plain dot-product correlation (DispNetC-style). fl/fr (B,C,H,W) ->
    (B,D,H,W). Disparity axis is in feature-pixel steps."""
    B, C, H, W = fl.shape
    vol = fl.new_zeros(B, num_disp, H, W)
    for d in range(num_disp):
        if d == 0:
            vol[:, d] = (fl * fr).sum(1)
        else:
            vol[:, d, :, d:] = (fl[..., d:] * fr[..., :-d]).sum(1)
    return vol


class CostAggregationUNet(nn.Module):
    """2D-conv U-Net over the cost volume (B,D,H/4,W/4) -> (B,D,H/4,W/4)
    scores, treating disparity as channels - no 3D convolution anywhere.

    `fxb` (focal x baseline) is concatenated as a constant feature map at the
    bottleneck, so the same trained weights generalise across the multiple
    simulated baselines this project records (Section 1.5 / Section 2.2).
    """

    def __init__(self, num_disp, ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(conv_bn_relu(num_disp, ch), ResBlock(ch))           # 1/4
        self.enc2 = nn.Sequential(conv_bn_relu(ch, ch * 2, stride=2),
                                  ResBlock(ch * 2))                                     # 1/8
        self.enc3 = nn.Sequential(conv_bn_relu(ch * 2, ch * 4, stride=2),
                                  ResBlock(ch * 4))                                     # 1/16
        self.bottleneck = conv_bn_relu(ch * 4 + 1, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False)
        self.up2_bn = nn.Sequential(nn.BatchNorm2d(ch * 2), nn.ReLU(inplace=True))
        self.dec2 = conv_bn_relu(ch * 2 + ch * 2, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False)
        self.up1_bn = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.dec1 = conv_bn_relu(ch + ch, ch)
        self.out = nn.Conv2d(ch, num_disp, 3, 1, 1)

    def forward(self, vol, fxb):
        e1 = self.enc1(vol)                     # 1/4,  ch
        e2 = self.enc2(e1)                       # 1/8,  2ch
        e3 = self.enc3(e2)                       # 1/16, 4ch
        fxb_map = (fxb / 100.0).view(-1, 1, 1, 1).expand(-1, 1, *e3.shape[2:])
        e3 = self.bottleneck(torch.cat([e3, fxb_map], 1))
        d2 = self.up2_bn(self.up2(e3))
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1_bn(self.up1(d2))
        d1 = self.dec1(torch.cat([d1, e1], 1))
        return self.out(d1)


class LearnedUpsample(nn.Module):
    """1/4-res disparity -> full-res: transposed-conv upsample with encoder
    skip connections (DispNetC-style), refining a bilinear base."""

    def __init__(self, ch=64):
        super().__init__()
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1, ch // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch // 2), nn.ReLU(inplace=True))                # 1/4 -> 1/2
        self.dec2 = conv_bn_relu(ch // 2 + ch // 2, ch // 2)                # + skip f2
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch // 4), nn.ReLU(inplace=True))                # 1/2 -> full
        self.dec1 = conv_bn_relu(ch // 4 + 3, ch // 4)                      # + left image
        self.out = nn.Conv2d(ch // 4, 1, 3, 1, 1)

    def forward(self, disp4, f2, left):
        base = upsample_disp(disp4, 4)           # bilinear base, full-res px units
        x = self.dec2(torch.cat([self.up2(disp4), f2], 1))
        x = self.dec1(torch.cat([self.up1(x), left], 1))
        return F.relu(base + self.out(x))


class StereoConvNet(nn.Module):
    """DispNetC-style 2D baseline (Section 6): 1D correlation + 2D-conv U-Net
    cost aggregation, conditioned on the calibration scalar, + learned
    upsample. Distinct from (and heavier than) this repo's `FastStereoNet`
    (Section 7)."""

    STRIDE = 4  # coarse disparity scale

    def __init__(self, max_disp=128, feat_ch=64):
        super().__init__()
        assert max_disp % self.STRIDE == 0
        self.max_disp = max_disp
        self.num_disp4 = max_disp // self.STRIDE
        self.encoder = SiameseEncoder2D(feat_ch)
        self.aggregation = CostAggregationUNet(self.num_disp4, feat_ch)
        self.upsample = LearnedUpsample(feat_ch)

    def forward(self, left, right, fxb):
        f2l, f4l = self.encoder(left)
        _, f4r = self.encoder(right)
        vol = correlation_1d(f4l, f4r, self.num_disp4)
        cost = self.aggregation(vol, fxb)
        disp4 = soft_argmin(cost) * self.STRIDE   # full-res units, 1/4 grid
        disp = self.upsample(disp4, f2l, left)
        return {"disp": disp, "aux": [(disp4, self.STRIDE)]}
