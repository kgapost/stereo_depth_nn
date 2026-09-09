"""Training losses for the stereo depth models, selected with --loss.

All four options supervise *disparity*, and all of them apply deep supervision
to the intermediate scales a model returns in out["aux"] with the same 0.4/0.6
weighting the original smooth-L1 loss used - so switching loss never changes
which outputs are supervised, only how.

Why anything other than smooth-L1. The metrics that matter here are depth
metrics: train.py's metrics() reports depth MAE, evaluate.py reports abs-rel
bucketed by range. Smooth-L1 on disparity is a poor proxy for them, because
depth = fxb / disp is wildly non-linear: on TartanAir (fxb = 80 px*m) a 1 px
error at disp 80 is 3 mm of depth, while the same 1 px at disp 4 is 5 m. A
uniform penalty in disparity therefore spends nearly all of its gradient on the
near field, which was already easy.

Working in log space fixes that, and for stereo the correspondence is exact
rather than approximate:

    log depth = log(fxb) - log(disp)

so an error in log-disparity *is* an error in log-depth (up to sign), and a
loss on log-disparity spatial gradients is *identical* to one on log-depth
gradients - the constant cancels in the finite differences. That makes the
YOLO26 depth losses portable here verbatim, which is what depth26/logl1 do.

  smoothl1  the original: smooth-L1 on disparity. Default, so existing runs
            and checkpoints stay directly comparable.
  depth26   faithful port of ultralytics' DepthLoss26 (SILog + multi-scale
            gradient matching), applied to log-disparity.
  logl1     the same, but plain L1 in log space instead of SILog.
  hybrid    smoothl1 + w_log * logl1 term + w_grad * gradient term.

A note on depth26 and scale. SILog subtracts the mean log error, which makes it
scale-*invariant* - the right choice for monocular depth, where absolute scale
is unknowable, and the wrong one here, where fxb hands us true metric scale for
free. depth26 is included as the honest reference port and as an ablation, but
expect logl1/hybrid to beat it on the depth metrics. Its EPE in particular can
look bad while its structure is good, which is why the grid summary also tracks
depth MAE (see train_tartanair.py).

The gradient-matching term is the other half of DepthLoss26 and is worth having
on its own: nothing in the pointwise losses rewards edge alignment, so they are
perfectly happy to smear a thin obstacle into the background behind it - which
is precisely the failure that matters for obstacle avoidance.
"""

import torch
import torch.nn.functional as F

from models import upsample_disp

LOSS_CHOICES = ["smoothl1", "depth26", "logl1", "hybrid"]

# Floor applied before log(). Half a pixel of disparity is already past what
# the sensor can resolve (>160 m at TartanAir's fxb=80), so nothing real is
# clipped; DepthLoss26's own 1e-3 floor would let log() reach -7 on invalid
# pixels and swamp the term.
MIN_DISP = 0.5

# Number of average-pooled scales in the gradient-matching term (DepthLoss26
# uses 4).
GRAD_SCALES = 4


def _aux_weights(n):
    """Deep-supervision weights: the last (finest) aux scale gets 0.6, the
    coarser ones 0.4 - unchanged from the original disparity_loss."""
    return [0.4] * max(0, n - 1) + ([0.6] if n else [])


def _grad_l1(lp, lg, vf):
    """L1 between predicted and GT log-disparity spatial gradients (dx, dy).

    Each finite difference is zeroed unless *both* contributing pixels are
    valid, so edges are only matched where the ground truth actually defines
    one. Ported from DepthLoss26._grad_l1.
    """
    pdx = (lp[:, :, :, 1:] - lp[:, :, :, :-1]) * vf[:, :, :, 1:] * vf[:, :, :, :-1]
    gdx = (lg[:, :, :, 1:] - lg[:, :, :, :-1]) * vf[:, :, :, 1:] * vf[:, :, :, :-1]
    pdy = (lp[:, :, 1:, :] - lp[:, :, :-1, :]) * vf[:, :, 1:, :] * vf[:, :, :-1, :]
    gdy = (lg[:, :, 1:, :] - lg[:, :, :-1, :]) * vf[:, :, 1:, :] * vf[:, :, :-1, :]
    return F.l1_loss(pdx, gdx) + F.l1_loss(pdy, gdy)


def _multiscale_grad(pred, gt, vf, scales=GRAD_SCALES):
    """Gradient matching summed over `scales` average-pooled resolutions.

    Coarser levels catch large-scale structure (a wall leaning the wrong way)
    that a single-resolution gradient term is blind to.
    """
    lp = pred.clamp(min=MIN_DISP).log()
    lg = gt.clamp(min=MIN_DISP).log()
    total = _grad_l1(lp, lg, vf)
    for _ in range(1, max(scales, 1)):
        if lp.shape[-1] < 4 or lp.shape[-2] < 4:
            break
        vp = F.avg_pool2d(vf, 2)
        occupied = vp > 0
        # Keep descending only while each image's occupied cells are mostly
        # full. Contiguous padding stays full under pooling, scattered holes
        # (real-rig stereo, LiDAR GT) do not - and pooling those produces
        # meaningless gradients. Near-inert on dense sim depth.
        keep = (vp.sum(dim=(1, 2, 3)) > 0.7 * occupied.sum(dim=(1, 2, 3))).view(-1, 1, 1, 1)
        if not keep.any():
            break
        denom = vp.clamp(min=1e-6)
        lp = F.avg_pool2d(lp * vf, 2) / denom
        lg = F.avg_pool2d(lg * vf, 2) / denom
        vf = occupied.to(vf.dtype) * keep.to(vf.dtype)  # zeroed images cannot re-enter
        total = total + _grad_l1(lp, lg, vf)
    return total


def _silog(lp, lg, lam):
    """Scale-invariant log error over valid pixels (lam=1.0 fully invariant,
    0.0 reduces to log-RMSE). Centered-variance form from DepthLoss26: it is
    non-negative by construction and stable in fp16 near convergence."""
    d = lp - lg
    m = d.mean()
    return torch.sqrt(((d - m) ** 2).mean() + (1.0 - lam) * m ** 2 + 1e-6)


def add_loss_args(ap):
    """Register --loss and its weights on an ArgumentParser.

    Shared by train.py and train_tartanair.py so the two cannot drift apart -
    a grid sweep launches train_tartanair.py subprocesses that must accept
    exactly what the sweep forwards.
    """
    ap.add_argument("--loss", default="smoothl1", choices=LOSS_CHOICES,
                    help="training objective. smoothl1 is the original "
                         "disparity loss and stays the default so previous "
                         "runs remain comparable; depth26/logl1/hybrid add "
                         "log-space and gradient-matching terms that target "
                         "the depth metrics directly (see losses.py). Note "
                         "that train_loss in log.csv is only comparable "
                         "between runs using the same --loss; val EPE, D1 and "
                         "depth MAE stay comparable across all of them.")
    ap.add_argument("--loss-w-log", type=float, default=1.0,
                    help="weight on the log-space term (depth26/logl1/hybrid)")
    ap.add_argument("--loss-w-grad", type=float, default=0.5,
                    help="weight on the multi-scale gradient-matching term")
    ap.add_argument("--loss-silog-lambda", type=float, default=1.0,
                    help="SILog scale-invariance for --loss depth26: 1.0 is "
                         "fully scale-invariant (the DepthLoss26 default), 0.0 "
                         "reduces to log-RMSE and keeps absolute scale, which "
                         "is what stereo's known fxb actually gives you")


def build_loss(name, w_log=1.0, w_grad=0.5, silog_lambda=1.0):
    """Return criterion(out, disp_gt, valid) -> scalar loss tensor.

    `out` is a model's dict with "disp" (B,1,H,W) in full-resolution pixel
    units and an optional "aux" list of (disp, stride) pairs at coarser grids.
    `valid` is the dataset's float mask; invalid GT pixels hold disp 0.
    """
    if name not in LOSS_CHOICES:
        raise ValueError(f"unknown loss '{name}' (choices: {', '.join(LOSS_CHOICES)})")

    def criterion(out, disp_gt, valid):
        vb = valid > 0.5
        if vb.sum() == 0:
            # Stay attached: fit() calls backward() unconditionally.
            return out["disp"].sum() * 0.0
        vf = vb.to(disp_gt.dtype)

        def term(pred):
            if name == "smoothl1":
                return F.smooth_l1_loss(pred[vb], disp_gt[vb])

            lp = pred.clamp(min=MIN_DISP).log()[vb]
            lg = disp_gt.clamp(min=MIN_DISP).log()[vb]
            grad = _multiscale_grad(pred, disp_gt, vf)
            if name == "depth26":
                return w_log * _silog(lp, lg, silog_lambda) + w_grad * grad
            if name == "logl1":
                return w_log * F.l1_loss(lp, lg) + w_grad * grad
            # hybrid: near-field precision from smooth-L1, far-field relative
            # accuracy from the log term. The balance between the two is the
            # thing worth tuning with --loss-w-log.
            return (F.smooth_l1_loss(pred[vb], disp_gt[vb])
                    + w_log * F.l1_loss(lp, lg) + w_grad * grad)

        total = term(out["disp"])
        aux = out.get("aux", [])
        for w, (d, stride) in zip(_aux_weights(len(aux)), aux):
            total = total + w * term(upsample_disp(d, stride))
        return total

    return criterion
