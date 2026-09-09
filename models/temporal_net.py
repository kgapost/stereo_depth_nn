"""TempoBandNet - temporal stereo with ego-motion-guided narrow-band matching.

Idea (the "elaborate" model): a drone always knows its ego-motion (here: sim
ground truth; on hardware: VIO/IMU). The previous frame's disparity is
forward-splatted into the current view with the relative camera pose, giving a
per-pixel disparity *prior*. Matching then only samples a narrow, log-spaced
band of disparity hypotheses around the prior instead of sweeping the full
range, and a confidence-gated ConvGRU fuses temporal context (the gate learns
to distrust the prior at disocclusions / dynamic objects / bad pose).

Pipeline per frame t:
  1. features at 1/4 and 1/8 (shared siamese encoder)
  2. full-sweep coarse volume at 1/8 (cold-start / fallback; skippable at
     inference when the prior covers the image)
  3. splat disp_{t-1} -> prior + validity mask; backward-warp GRU hidden state
  4. gated fusion: init = gate * prior + (1 - gate) * coarse
  5. narrow-band matching around init at 1/8 (11 log-spaced offsets), then
     around the upsampled result at 1/4 (5 offsets)
  6. full-resolution edge-aware refinement

All disparities are in full-resolution pixel units (see models/common.py).
State passed between frames: {"hidden": (B,C,H/8,W/8), "disp8": (B,1,H/8,W/8)}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (FeatureExtractor, gwc_volume, soft_argmin, conv_bn_relu,
                     CostAggregation3D, DisparityRefinement, upsample_disp)


# --------------------------------------------------------------------------
# Geometry (all at the 1/8 feature grid; K8 = intrinsics scaled to that grid)
# --------------------------------------------------------------------------

def scale_intrinsics(K, stride):
    """K (B,4)=[fx,fy,cx,cy] full-res -> intrinsics of the stride-s grid."""
    f = K[:, :2] / stride
    c = (K[:, 2:] + 0.5) / stride - 0.5
    return torch.cat([f, c], 1)


def _unproject(depth, K8):
    B, _, h, w = depth.shape
    fx, fy = K8[:, 0].view(-1, 1, 1), K8[:, 1].view(-1, 1, 1)
    cx, cy = K8[:, 2].view(-1, 1, 1), K8[:, 3].view(-1, 1, 1)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype), indexing="ij")
    z = depth[:, 0]
    x = (xs.unsqueeze(0) - cx) / fx * z
    y = (ys.unsqueeze(0) - cy) / fy * z
    return torch.stack([x, y, z], 1)  # (B,3,h,w)


@torch.no_grad()
def splat_prior(disp_prev, T_cur_prev, K8, fxb):
    """Forward-splat previous disparity into the current view (z-buffered).

    disp_prev  (B,1,h,w) full-res units on the 1/8 grid (detached)
    T_cur_prev (B,4,4)   optical-frame relative pose
    fxb        (B,)      fx_fullres * baseline
    Returns prior_disp (0 where invalid), prior_depth (+inf where invalid),
    mask (B,1,h,w) in {0,1}.
    """
    B, _, h, w = disp_prev.shape
    disp_prev = disp_prev.float()
    fxb_ = fxb.view(B, 1, 1, 1).float()
    depth = fxb_ / disp_prev.clamp(min=0.25)
    pts = _unproject(depth, K8.float()).view(B, 3, -1)
    R, t = T_cur_prev[:, :3, :3].float(), T_cur_prev[:, :3, 3:].float()
    pc = R @ pts + t
    z = pc[:, 2]
    u = K8[:, 0].view(-1, 1) * pc[:, 0] / z.clamp(min=1e-3) + K8[:, 2].view(-1, 1)
    v = K8[:, 1].view(-1, 1) * pc[:, 1] / z.clamp(min=1e-3) + K8[:, 3].view(-1, 1)
    ui, vi = u.round().long(), v.round().long()
    valid = (z > 0.2) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

    lin = vi.clamp(0, h - 1) * w + ui.clamp(0, w - 1)
    lin = lin + torch.arange(B, device=lin.device).view(B, 1) * (h * w)
    zsrc = torch.where(valid, z, torch.full_like(z, float("inf")))
    zbuf = torch.full((B * h * w,), float("inf"), device=z.device, dtype=z.dtype)
    zbuf.scatter_reduce_(0, lin.view(-1), zsrc.view(-1), reduce="amin", include_self=True)

    prior_depth = zbuf.view(B, 1, h, w)
    mask = torch.isfinite(prior_depth).float()
    prior_disp = torch.where(mask.bool(), fxb_ / prior_depth.clamp(min=1e-3),
                             torch.zeros_like(prior_depth))
    return prior_disp, prior_depth, mask


def warp_hidden(hidden_prev, prior_depth, mask, T_cur_prev, K8):
    """Backward-warp the previous hidden state to the current frame using the
    splatted depth: current pixel -> 3D -> previous frame -> bilinear sample."""
    B, _, h, w = prior_depth.shape
    depth = torch.where(mask.bool(), prior_depth, prior_depth.new_ones(1))
    pts = _unproject(depth.float(), K8.float()).view(B, 3, -1)
    R, t = T_cur_prev[:, :3, :3].float(), T_cur_prev[:, :3, 3:].float()
    pp = R.transpose(1, 2) @ (pts - t)  # into previous optical frame
    z = pp[:, 2].clamp(min=1e-3)
    u = K8[:, 0].view(-1, 1) * pp[:, 0] / z + K8[:, 2].view(-1, 1)
    v = K8[:, 1].view(-1, 1) * pp[:, 1] / z + K8[:, 3].view(-1, 1)
    gx = 2.0 * u / (w - 1) - 1.0
    gy = 2.0 * v / (h - 1) - 1.0
    grid = torch.stack([gx, gy], -1).view(B, h, w, 2).to(hidden_prev.dtype)
    warped = F.grid_sample(hidden_prev, grid, mode="bilinear",
                           padding_mode="zeros", align_corners=True)
    return warped * mask


# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------

class ConvGRU(nn.Module):
    def __init__(self, hidden_ch, in_ch):
        super().__init__()
        self.convz = nn.Conv2d(hidden_ch + in_ch, hidden_ch, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_ch + in_ch, hidden_ch, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_ch + in_ch, hidden_ch, 3, 1, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], 1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], 1)))
        return (1 - z) * h + z * q


class NarrowBand(nn.Module):
    """Sample right features at center+offset hypotheses, score with 2D convs,
    soft-select the disparity. Offsets are in full-res pixel units."""

    def __init__(self, offsets, stride, max_disp, feat_ch=32, groups=8,
                 extra_ch=0, hidden=48):
        super().__init__()
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32))
        self.stride = stride
        self.max_disp = max_disp
        self.groups = groups
        k = len(offsets)
        self.net = nn.Sequential(
            conv_bn_relu(k * groups + 1 + extra_ch, hidden),
            conv_bn_relu(hidden, hidden),
            nn.Conv2d(hidden, k, 3, 1, 1))

    def forward(self, fl, fr, center, extra=None):
        B, C, h, w = fl.shape
        cpg = C // self.groups
        ys, xs = torch.meshgrid(
            torch.arange(h, device=fl.device, dtype=torch.float32),
            torch.arange(w, device=fl.device, dtype=torch.float32), indexing="ij")
        gy = (2.0 * ys / (h - 1) - 1.0).expand(B, h, w)

        corrs = []
        for off in self.offsets:
            d_feat = (center[:, 0] + off) / self.stride       # feature-pixel units
            gx = 2.0 * (xs.unsqueeze(0) - d_feat) / (w - 1) - 1.0
            grid = torch.stack([gx, gy], -1).to(fl.dtype)
            fr_w = F.grid_sample(fr, grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=True)
            corrs.append((fl * fr_w).view(B, self.groups, cpg, h, w).mean(2))

        feats = [torch.cat(corrs, 1), center / self.max_disp]
        if extra is not None:
            feats.append(extra)
        scores = self.net(torch.cat(feats, 1))
        prob = F.softmax(scores, 1)
        hyps = center + self.offsets.view(1, -1, 1, 1)        # (B,K,h,w)
        disp = (prob * hyps).sum(1, keepdim=True).clamp(min=0.0)
        return disp, prob


class TempoBandNet(nn.Module):
    STRIDE = 8

    def __init__(self, max_disp=128, feat_ch=32, groups=8, hidden_ch=32, ctx_ch=16):
        super().__init__()
        assert max_disp % self.STRIDE == 0
        self.max_disp = max_disp
        self.num_disp8 = max_disp // self.STRIDE
        self.groups = groups
        self.hidden_ch = hidden_ch

        self.features = FeatureExtractor(feat_ch)
        self.aggregation = CostAggregation3D(groups)          # full sweep @1/8

        fuse_in = feat_ch + 4  # f8l + coarse + prior + mask + |coarse-prior|
        self.fuse_enc = conv_bn_relu(fuse_in, hidden_ch)
        self.gru = ConvGRU(hidden_ch, hidden_ch)
        self.gate = nn.Conv2d(hidden_ch, 1, 3, 1, 1)
        self.ctx = nn.Conv2d(hidden_ch, ctx_ch, 3, 1, 1)

        self.band8 = NarrowBand([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16],
                                stride=8, max_disp=max_disp, feat_ch=feat_ch,
                                groups=groups, extra_ch=ctx_ch)
        self.band4 = NarrowBand([-2, -1, 0, 1, 2], stride=4, max_disp=max_disp,
                                feat_ch=feat_ch, groups=groups, hidden=32)
        self.refine = DisparityRefinement(max_disp)

    def forward(self, left, right, K, fxb, state=None, rel_pose=None):
        """K (B,4) full-res [fx,fy,cx,cy]; fxb (B,); rel_pose (B,4,4) optical
        T_cur<-prev (required when state is not None)."""
        B = left.shape[0]
        f4l, f8l = self.features(left)
        f4r, f8r = self.features(right)
        h8, w8 = f8l.shape[-2:]
        K8 = scale_intrinsics(K, self.STRIDE)

        # 1) full-sweep coarse (cold-start & per-pixel fallback)
        vol = gwc_volume(f8l, f8r, self.num_disp8, self.groups)
        coarse = soft_argmin(self.aggregation(vol)) * self.STRIDE

        # 2) temporal prior
        if state is not None:
            prior, prior_depth, mask = splat_prior(state["disp8"], rel_pose, K8, fxb)
            h_warp = warp_hidden(state["hidden"], prior_depth, mask, rel_pose, K8)
        else:
            prior = torch.zeros_like(coarse)
            mask = torch.zeros_like(coarse)
            h_warp = f8l.new_zeros(B, self.hidden_ch, h8, w8)

        # 3) confidence-gated fusion
        x = self.fuse_enc(torch.cat(
            [f8l, coarse / self.max_disp, prior / self.max_disp, mask,
             (coarse - prior).abs() / self.max_disp], 1))
        h = self.gru(h_warp, x)
        gate = torch.sigmoid(self.gate(h)) * mask
        init = gate * prior + (1.0 - gate) * coarse

        # 4) narrow-band matching, coarse-to-fine
        disp8, _ = self.band8(f8l, f8r, init, extra=self.ctx(h))
        disp4, _ = self.band4(f4l, f4r, upsample_disp(disp8, 2))

        # 5) full-res refinement
        disp = self.refine(upsample_disp(disp4, 4), left)

        new_state = {"hidden": h,
                     "disp8": F.avg_pool2d(disp4.detach(), 2)}
        return {"disp": disp,
                "aux": [(coarse, 8), (disp8, 8), (disp4, 4)],
                "state": new_state, "gate": gate, "prior_mask": mask}
