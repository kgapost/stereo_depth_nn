"""Establish that occlusion.py's blind-spot labels are correct, on real data.

The labels in occlusion.py are only worth building a paper on if they can be
shown to be right, so this script runs three independent checks and prints a
table for each:

 1. AGREEMENT. The cheap forward-warp occlusion test (left depth only) against
    the left-right consistency test (which also reads the right camera's GT
    depth, something the loader has never used). These are geometrically
    independent derivations, so high IoU on real scenes means the cheap test -
    the one that will actually run in the data pipeline - is trustworthy.

 2. PREDICTIVE POWER - the acid test. A real block matcher (OpenCV SGBM, a
    stand-in for the classic pipeline README Section 1.1 wants to replace) is
    run on the same frames, and its disparity error is reported per label
    category. If the labels mean anything, error must be far higher inside
    `occluded`/`textureless` than on matchable pixels. This is what turns
    "these masks are geometrically derived" into "these masks locate the
    regions that actually break stereo".

 3. CALIBRATION. `textureless_mask`'s threshold is the one tunable number in
    the module, so it is swept rather than guessed: for each candidate, the
    script reports how much of the image gets flagged and how much matcher
    error is captured. A good threshold flags few pixels and captures much
    error.

Usage:
    python scripts/verify_occlusion_gt.py --data ~/datasets/tartanair \
        --envs Ocean SeasonalForestAutumn --frames 40
"""
import argparse
import os
import sys
import glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_datasets import read_tartanair_depth          # noqa: E402
from occlusion import (blind_spot_channels, geometric_occlusion,   # noqa: E402
                       lr_consistency_occlusion,
                       structure_tensor_min_eig)

FXB = 80.0          # TartanAir: fx 320 px * 0.25 m baseline
MAX_DISP = 128


def find_frames(root, envs, per_env):
    """Evenly spaced frames from each environment's first trajectory."""
    out = []
    for env in envs:
        trajs = sorted(glob.glob(os.path.join(root, env, "Data_*", "P*")))
        if not trajs:
            print(f"  ! no trajectories under {env}", file=sys.stderr)
            continue
        for t in trajs[:1]:
            li = sorted(glob.glob(os.path.join(t, "image_lcam_front", "*.png")))
            ri = sorted(glob.glob(os.path.join(t, "image_rcam_front", "*.png")))
            ld = sorted(glob.glob(os.path.join(t, "depth_lcam_front", "*.png")))
            rd = sorted(glob.glob(os.path.join(t, "depth_rcam_front", "*.png")))
            n = min(len(li), len(ri), len(ld), len(rd))
            if n == 0:
                continue
            for i in np.linspace(0, n - 1, min(per_env, n)).astype(int):
                out.append((env, li[i], ri[i], ld[i], rd[i]))
    return out


def sgbm_disparity(left_bgr, right_bgr):
    """Classic SGBM disparity, in pixels. -1 where the matcher declines."""
    m = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=MAX_DISP, blockSize=5,
        P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = m.compute(left_bgr, right_bgr).astype(np.float32) / 16.0
    d[d <= 0] = -1.0
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/datasets/tartanair"))
    ap.add_argument("--envs", nargs="+",
                    default=["Ocean", "SeasonalForestAutumn", "Gascola"])
    ap.add_argument("--frames", type=int, default=20, help="frames per env")
    ap.add_argument("--tex-thresh", type=float, default=1e-6)
    ap.add_argument("--max-depth", type=float, default=95.0)
    a = ap.parse_args()

    frames = find_frames(a.data, a.envs, a.frames)
    if not frames:
        sys.exit("no frames found")
    print(f"verifying on {len(frames)} frames from {len(a.envs)} environments\n")

    agree = []                      # (iou, warp_frac, lr_frac)
    cov = {k: [] for k in ("occluded", "out_of_frame", "textureless", "any")}
    # matcher error accumulators: category -> [abs errors]
    err = {k: [] for k in ("matchable", "occluded", "out_of_frame", "textureless")}
    density = {k: [[], []] for k in err}     # [n_matcher_valid, n_total]

    # Every statistic below is POOLED over pixels (sum of errors / count of
    # pixels), never a mean of per-frame means. This matters more than it
    # looks: the textureless signal lives in ~4% of pixels concentrated on a
    # few frames, and averaging per-frame means dilutes it to nothing. An
    # earlier draft of this script did exactly that and reported textureless
    # as *better* than matchable.
    pool = {}                       # key -> [summed abs error, pixel count]

    def add(key, e, m):
        if m.any():
            s = pool.setdefault(key, [0.0, 0])
            s[0] += float(e[m].sum())
            s[1] += int(m.sum())

    # Exclusive structure-tensor bins. This is the trustworthy view of the
    # texture effect: the cumulative masks a threshold sweep produces blend
    # the sharp spike at the very bottom with the easiest band sitting right
    # above it, which is how 1e-3-scale thresholds end up looking harmless.
    tex_bins = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, float("inf")]

    for n, (env, lp, rp, ldp, rdp) in enumerate(frames, 1):
        left, right = cv2.imread(lp), cv2.imread(rp)
        dl, dr = read_tartanair_depth(ldp), read_tartanair_depth(rdp)

        valid = (dl > 0.2) & (dl < a.max_depth)
        disp = np.where(valid, FXB / np.maximum(dl, 1e-3), 0.0).astype(np.float32)
        valid &= disp < MAX_DISP
        disp_r = np.where((dr > 0.2) & (dr < a.max_depth),
                          FXB / np.maximum(dr, 1e-3), 0.0).astype(np.float32)

        # --- check 1: do the two independent occlusion tests agree? ---
        o_warp = geometric_occlusion(disp, valid)
        o_lr = lr_consistency_occlusion(disp, disp_r, valid)
        inter, union = (o_warp & o_lr).sum(), (o_warp | o_lr).sum()
        agree.append((inter / max(union, 1), o_warp.mean(), o_lr.mean()))

        # textureless=True here even though it is off by default in the
        # module: measuring it is the whole point of checks 3 and 4.
        ch = blind_spot_channels(disp, left / 255.0, valid,
                                 disp_right=disp_r, tex_thresh=a.tex_thresh,
                                 textureless=True)
        for k in cov:
            cov[k].append(ch[k][valid].mean() if valid.any() else 0.0)

        # --- check 2: does matcher error concentrate in the labels? ---
        sg = sgbm_disparity(left, right)
        got = (sg > 0) & valid
        matchable = valid & ~ch["any"]
        cats = {"matchable": matchable, "occluded": ch["occluded"],
                "out_of_frame": ch["out_of_frame"], "textureless": ch["textureless"]}
        for k, m in cats.items():
            mm = m & got
            if mm.any():
                err[k].append(np.abs(sg[mm] - disp[mm]))
            density[k][0].append(int(mm.sum()))
            density[k][1].append(int(m.sum()))

        # --- check 3: calibrate the textureless threshold ---
        # Base: matcher returned something, and the pixel is not already
        # explained by the two geometric channels. Error is attributed to
        # texture only on pixels where matching was otherwise possible.
        e = np.abs(sg - disp)
        base = got & ~ch["occluded"] & ~ch["out_of_frame"]
        add("base", e, base)
        lam = structure_tensor_min_eig(left / 255.0)
        for i in range(len(tex_bins) - 1):
            add(("texbin", i), e,
                base & (lam >= tex_bins[i]) & (lam < tex_bins[i + 1]))

        if n % 10 == 0 or n == len(frames):
            print(f"  {n}/{len(frames)} frames", flush=True)

    ag = np.array(agree)
    print("\n=== 1. forward-warp vs left-right-consistency occlusion ===")
    print(f"  IoU                     mean {ag[:,0].mean():.4f}   min {ag[:,0].min():.4f}")
    print(f"  occluded frac (warp)    mean {ag[:,1].mean()*100:.2f}%")
    print(f"  occluded frac (LR)      mean {ag[:,2].mean()*100:.2f}%")

    print("\n=== 2. blind-spot coverage (share of valid pixels) ===")
    for k in ("occluded", "out_of_frame", "textureless", "any"):
        v = np.array(cov[k]) * 100
        print(f"  {k:<14} {v.mean():6.2f}%   (min {v.min():5.2f}  max {v.max():5.2f})")

    print("\n=== 3. SGBM disparity error by category (the acid test) ===")
    print(f"  {'category':<14} {'MAE px':>9} {'p90 px':>9} {'matcher ret':>12}"
          f" {'px/frame':>10}")
    for k in ("matchable", "occluded", "out_of_frame", "textureless"):
        n_tot = sum(density[k][1]) / max(len(frames), 1)
        if not err[k]:
            # No matcher output at all in this category is itself the result -
            # out_of_frame is expected to land here, since SGBM cannot return
            # a disparity for the left border strip by construction.
            print(f"  {k:<14} {'n/a':>9} {'n/a':>9} {0.0:11.1f}% {n_tot:10.0f}")
            continue
        e = np.concatenate(err[k])
        ret = sum(density[k][0]) / max(sum(density[k][1]), 1)
        print(f"  {k:<14} {e.mean():9.3f} {np.percentile(e,90):9.3f}"
              f" {ret*100:11.1f}% {n_tot:10.0f}")

    print("\n=== 4. SGBM error by exclusive structure-tensor bin ===")
    print("  (base: matcher returned, unoccluded, in-frame -- so this isolates"
          "\n   texture from the geometric channels)")
    print(f"  {'lambda_min range':<24} {'MAE px':>9} {'share':>8} {'vs base':>9}")
    b_sum, b_n = pool.get("base", [0.0, 0])
    b_mae = b_sum / max(b_n, 1)
    for i in range(len(tex_bins) - 1):
        s = pool.get(("texbin", i))
        if not s:
            continue
        mae = s[0] / max(s[1], 1)
        lo, hi = tex_bins[i], tex_bins[i + 1]
        label = f"[{lo:.0e}, {hi:.0e})" if np.isfinite(hi) else f"[{lo:.0e}, inf)"
        print(f"  {label:<24} {mae:9.3f} {s[1]/max(b_n,1)*100:7.1f}%"
              f" {mae/max(b_mae,1e-9):8.2f}x")
    print(f"  {'(base overall)':<24} {b_mae:9.3f} {100.0:7.1f}% {1.0:8.2f}x")


if __name__ == "__main__":
    main()
