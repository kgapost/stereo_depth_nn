"""Blind-spot ground truth derived from dense GT depth.

README Section 1.2 asks for a model that (a) *finds* the regions where stereo
matching has no usable signal and (b) *fills* them with something better than
copying the nearest pixel. Section 5's gap table claims that as this
project's contribution. Neither half was implementable before now, for a
simple reason: there was no ground truth for "blind spot". The dataset's
`valid` mask (stereo_datasets.py) is a depth-range filter -
`depth > 0.2 & depth < max_depth` - which says nothing about whether a pixel
is matchable.

This module derives that missing ground truth. It needs no new data
collection: everything here comes from dense GT depth that is already on
disk. Section 1.2 names three categories of blind spot, and they are not
equally derivable, so they are kept as separate channels rather than merged
into one vague "bad pixel" label:

  occluded     - geometric. The surface this left pixel sees is hidden behind
                 a nearer surface in the right view, so no correspondence
                 *exists*. Two independent derivations are provided, and
                 which one to trust was settled by measurement rather than by
                 argument (scripts/verify_occlusion_gt.py):

                   `lr_consistency_occlusion` asks whether the right camera's
                   own GT depth agrees about the surface at the corresponding
                   column. This is the DEFAULT and the one to use.
                   `geometric_occlusion` runs a z-buffer forward-warp
                   collision test on the left disparity alone.

                 On a controlled synthetic scene the two agree perfectly
                 (IoU 1.00). On real TartanAir frames they agree only to
                 IoU 0.63, and arbitrating the disagreement by SGBM error
                 shows the forward-warp test is the deficient one: pixels only
                 the LR test flags carry 5.44 px of matcher error (against a
                 0.99 px unoccluded baseline) and number ~9,900 per frame,
                 while pixels only the warp test flags carry 3.04 px and
                 number ~1,600. The warp test therefore misses roughly 38% of
                 genuine occlusions - thin occluders such as grass blades and
                 wave crests never trigger its collision test, because the
                 occluder is not well sampled on the left pixel grid. It is
                 kept only as a fallback for data with no right-camera depth.
  out_of_frame - geometric. The correspondence exists but falls outside the
                 right image, i.e. the left border strip of width ~max_disp.
                 Separated from `occluded` because it is a property of the rig,
                 not the scene, and a model should not be credited for
                 "detecting" a fixed border. SGBM declines to return a
                 disparity for essentially all of it, so it is trivially real.
  textureless  - photometric, OPT-IN, and deliberately not part of the
                 supervised target. A correspondence exists and is
                 unoccluded, but the local neighbourhood carries no gradient
                 to match on. This is a proxy (local structure tensor), not an
                 exact label, and measurement cut it down to almost nothing:
                 binning SGBM error by structure-tensor eigenvalue over
                 unoccluded pixels (120 frames, 3 environments) gives

                     lambda_min        MAE px   share   vs base
                     < 1e-7             4.455    0.1%    6.99x
                     1e-7 .. 1e-6       0.607    1.8%    0.95x
                     1e-6 .. 1e-5       0.378   14.2%    0.59x
                     1e-5 .. 1e-4       0.372   18.6%    0.58x
                     1e-4 .. 1e-3       0.544   33.9%    0.85x
                     1e-3 .. 1e-2       0.991   30.9%    1.56x
                     > 1e-2             1.142    0.6%    1.79x
                     base overall       0.637  100.0%    1.00x

                 A genuine textureless blind spot does exist - 7x the baseline
                 error - but it covers 0.1% of pixels, which is a rounding
                 error next to occlusion's 9.1%. Any threshold loose enough to
                 cover a useful area dilutes it to nothing (1.99% of pixels at
                 1e-6 scores 1.4x). Note also that the curve is U-shaped and
                 the *upper* end carries more total error than the lower: in
                 this data "high texture" means foliage and thin structure,
                 i.e. geometry, not texture. So the honest reading is that
                 occlusion is the blind-spot problem on this corpus, and a
                 regularised matcher's smoothness prior already handles plain
                 surfaces well - which is itself worth reporting, since
                 Section 1.2 assumed otherwise.

                 Two earlier drafts of this module got this wrong in opposite
                 directions (a 1e-3 threshold that flagged 70% of the image
                 and scored *below* baseline; then a 6 px "spike" at 1e-6 that
                 turned out to be 24-frame sampling noise and vanished at 120
                 frames). Both mistakes are why `scripts/verify_occlusion_gt.py`
                 reports pixel-pooled exclusive bins rather than thresholds.

The third category Section 1.2 mentions - shiny or repeating surfaces, where
the cost volume is multi-modal rather than flat - is deliberately NOT emitted
here. Detecting it properly needs the cost volume itself, which is a property
of a given model and not of the data, so a dataset-level label would be
circular. It is a documented limitation, not an oversight.

So the supervised blind-spot target is the two geometric channels:
`occluded` (9.1% of valid pixels, 8.1x the matcher error of a matchable
pixel) and `out_of_frame` (2.6%, which SGBM declines outright) - 11.7% of
the image, both derived exactly, both independently validated.

Why synthetic data makes the *filling* half trainable at all: GT depth is
dense everywhere, including inside the occluded regions. So for every pixel
this module labels as a blind spot, the correct depth is still known and can
supervise a fill directly. That is not true of any real-world stereo dataset
with LiDAR ground truth, where occluded pixels are usually just missing.

All functions are numpy, single-frame, and cheap enough to run inside a
DataLoader worker.
"""
import numpy as np

__all__ = ["geometric_occlusion", "lr_consistency_occlusion",
           "textureless_mask", "structure_tensor_min_eig",
           "blind_spot_channels"]


def geometric_occlusion(disp, valid=None, tol=0.5):
    """Left pixels whose surface is hidden by a nearer one in the right view.

    A left pixel at column x with disparity d corresponds to right column
    x - d. Two left pixels collide when they claim the same right column; the
    one with the larger disparity is nearer and wins, and the loser is
    occluded. That is the whole test, and on GT disparity it is exact.

    `tol` (px) is the slack before a collision counts, so that two samples of
    one continuous slanted surface - which legitimately land on the same right
    column - are not called occlusions of each other.

    Returns a bool array, True where occluded. Pixels whose correspondence
    falls off the left edge of the right image are NOT included here; they are
    `out_of_frame` (see `blind_spot_channels`).
    """
    disp = np.asarray(disp, dtype=np.float32)
    H, W = disp.shape
    if valid is None:
        valid = np.ones((H, W), dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    xs = np.broadcast_to(np.arange(W, dtype=np.float32), (H, W))
    xr = np.rint(xs - disp).astype(np.int64)
    inb = valid & (xr >= 0) & (xr < W)

    rows = np.broadcast_to(np.arange(H, dtype=np.int64)[:, None], (H, W))
    flat = (rows * W + xr).ravel()[inb.ravel()]
    d = disp.ravel()[inb.ravel()]

    # Scatter in ascending-disparity order, so the largest disparity claiming
    # each right column is the value that survives. Cheaper than
    # np.maximum.at, which is unbuffered and an order of magnitude slower.
    best = np.full(H * W, -np.inf, dtype=np.float32)
    order = np.argsort(d, kind="stable")
    best[flat[order]] = d[order]

    occ = np.zeros(H * W, dtype=bool)
    occ[inb.ravel()] = d < best[flat] - tol
    return occ.reshape(H, W)


def lr_consistency_occlusion(disp_left, disp_right, valid=None, tol=1.0):
    """Independent occlusion evidence, using the *right* camera's GT depth.

    TartanAir ships `depth_rcam_front` alongside `depth_lcam_front` (and this
    project's AirSim collection records every right camera too), but the
    loader has only ever read the left one. Converting the right depth to a
    right-view disparity map gives a second, geometrically independent test:
    a left pixel at x with disparity d is genuinely visible only if the right
    view agrees that the surface at column x - d sits at the same disparity.
    Where it disagrees, something nearer is in the way.

    This exists to *validate* `geometric_occlusion`, not to replace it: it
    needs an extra depth image, so it is not something a deployed model or a
    single-view dataset can rely on. Use it to confirm the cheap test is
    right, then use the cheap test.
    """
    disp_left = np.asarray(disp_left, dtype=np.float32)
    disp_right = np.asarray(disp_right, dtype=np.float32)
    H, W = disp_left.shape
    if valid is None:
        valid = np.ones((H, W), dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    xs = np.broadcast_to(np.arange(W, dtype=np.float32), (H, W))
    xr = np.rint(xs - disp_left).astype(np.int64)
    inb = valid & (xr >= 0) & (xr < W)

    rows = np.broadcast_to(np.arange(H, dtype=np.int64)[:, None], (H, W))
    sampled = np.zeros((H, W), dtype=np.float32)
    sampled[inb] = disp_right[rows[inb], xr[inb]]

    occ = np.zeros((H, W), dtype=bool)
    occ[inb] = np.abs(sampled[inb] - disp_left[inb]) > tol
    return occ


def textureless_mask(image, win=7, thresh=1e-6):
    """Unoccluded pixels with too little local gradient to match on.

    Uses the smaller eigenvalue of the windowed structure tensor - the same
    quantity Shi-Tomasi corner response thresholds on. Taking the *smaller*
    eigenvalue is the point: a region can have a strong gradient in one
    direction only (a horizontal edge, or a sky gradient) and still be
    hopeless for stereo, because matching happens along the epipolar line.
    A single gradient-magnitude threshold would wrongly call those matchable.

    `image` is HxW grayscale or HxWxC in [0, 1]. `thresh` is in units of
    (intensity/pixel)^2 and is the one genuinely tunable number in this
    module; `scripts/verify_occlusion_gt.py` bins measured matching error
    against it, so the 1e-5 default is calibrated rather than guessed. See
    this module's docstring for why larger thresholds are actively wrong
    here rather than merely loose.
    """
    return structure_tensor_min_eig(image, win) < thresh


def structure_tensor_min_eig(image, win=7):
    """Smaller eigenvalue of the windowed structure tensor, per pixel.

    Exposed separately from `textureless_mask` so that calibration scripts
    can bin against the raw quantity instead of re-deriving it and risking
    measuring something subtly different from what the mask thresholds.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)

    gy, gx = np.gradient(img)
    # Windowed sums of the structure-tensor entries, via a separable box
    # filter built from cumulative sums (no scipy dependency).
    a = _box(gx * gx, win)
    b = _box(gx * gy, win)
    c = _box(gy * gy, win)

    # Smaller eigenvalue of [[a, b], [b, c]], guarded against tiny negatives
    # from floating-point error inside the sqrt.
    tr, det = a + c, a * c - b * b
    disc = np.maximum(tr * tr / 4.0 - det, 0.0)
    return tr / 2.0 - np.sqrt(disc)


def _box(x, win):
    """Mean over a win x win window, edge-replicated, via cumulative sums."""
    r = win // 2
    p = np.pad(x, r + 1, mode="edge")
    cs = p.cumsum(0).cumsum(1)
    # Inclusive-exclusive corners of each window in the padded integral image.
    H, W = x.shape
    y0, x0 = np.arange(H), np.arange(W)
    ya, yb = y0, y0 + win
    xa, xb = x0, x0 + win
    out = (cs[np.ix_(yb, xb)] - cs[np.ix_(ya, xb)]
           - cs[np.ix_(yb, xa)] + cs[np.ix_(ya, xa)])
    return out / float(win * win)


def blind_spot_channels(disp, left_image=None, valid=None, disp_right=None,
                        tol=1.0, tex_win=7, tex_thresh=1e-7,
                        textureless=False):
    """The per-pixel blind-spot labels, as a dict of bool HxW arrays.

    Keys: `occluded`, `out_of_frame`, and `any` (their union). The channels
    are made mutually exclusive in that priority order, so they can be summed,
    reported, or stacked into a multi-channel target without double-counting a
    pixel.

    `textureless=True` adds a third channel (and includes it in `any`), which
    requires `left_image`. It is OFF by default, and that default is a
    measured decision rather than a convenience: see this module's docstring
    for the eigenvalue-binned numbers: the only threshold at which the channel
    is real covers 0.1% of pixels, so including it adds noise to the target
    without adding coverage. It stays available because the measurement was
    made on TartanAir, and a corpus with more plain man-made surfaces could
    plausibly behave differently - in which case re-run
    scripts/verify_occlusion_gt.py on that corpus before turning it on.

    Pass `disp_right` - the right camera's GT disparity - whenever it exists.
    It selects the left-right consistency test, which this module's docstring
    shows to be substantially more complete than the forward-warp fallback
    used when it is absent. TartanAir ships `depth_rcam_front` for every
    trajectory, and this project's AirSim collection records every right
    camera, so the fallback should be rare in practice.
    """
    disp = np.asarray(disp, dtype=np.float32)
    H, W = disp.shape
    if valid is None:
        valid = np.ones((H, W), dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    xs = np.broadcast_to(np.arange(W, dtype=np.float32), (H, W))
    out_of_frame = valid & (np.rint(xs - disp) < 0)

    if disp_right is not None:
        occluded = lr_consistency_occlusion(disp, disp_right, valid, tol=tol)
    else:
        occluded = geometric_occlusion(disp, valid, tol=tol)
    occluded &= ~out_of_frame

    ch = {"occluded": occluded, "out_of_frame": out_of_frame}
    ch["any"] = occluded | out_of_frame

    if textureless:
        if left_image is None:
            raise ValueError("textureless=True needs left_image")
        tex = (textureless_mask(left_image, tex_win, tex_thresh)
               & valid & ~occluded & ~out_of_frame)
        ch["textureless"] = tex
        ch["any"] = ch["any"] | tex
    return ch
