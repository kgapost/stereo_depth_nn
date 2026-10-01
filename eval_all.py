"""Health-check and evaluate every run under a grid-search output directory.

For each run directory (as named by train_tartanair.py's run_grid_search):

  1. Health check: did it reach the expected epoch count, and are both
     best.pth and last.pth present? (last.pth is written every epoch; best.pth
     only when val EPE improves - a run whose loss went NaN on epoch 0 reaches
     20/20 epochs but never gets a best.pth, see losses.py/train.py fit().)

  2. For each checkpoint that exists (best.pth, last.pth), evaluate on two
     held-out-environment splits built from the checkpoint's own recorded
     --data/--crop/--window/etc. (saved in ck["args"] by train.py's fit()),
     so every run is re-evaluated under exactly the settings it was trained
     with, not today's CLI defaults:

       "same"      - environment_split() with that run's own --val-frac /
                     --val-envs, i.e. the identical held-out environment(s)
                     it validated against during training. Note this is
                     "same" as in "reproduces the training split", not
                     "same seed" in the sense of environment_split depending
                     on --seed - it doesn't (see environment_split's
                     docstring in train_tartanair.py): which environments get
                     held out is a deterministic function of val_frac/
                     val_envs alone, seed only affects window subsampling.
                     Every run in a sweep therefore already validates
                     against the same held-out environment(s); this
                     reproduces that number as a sanity check on the
                     harness itself.
       "different" - alt_environment_split() below: the same by-environment
                     holdout *methodology*, but a differently-seeded (see
                     --diff-seed) random choice of which environment(s) to
                     hold out, deliberately excluding whatever "same" held
                     out so the two are guaranteed disjoint rather than
                     maybe-different-by-luck. This is a second, independent
                     generalization estimate - not a different dataset
                     (there is only one TartanAir root on this machine).

     Each split is capped to --max-eval-windows (fixed --cap-seed, so
     best.pth and last.pth of the same run see the identical subset) to keep
     a full-grid scan tractable - see run_grid_search's own --data-fraction
     for the same tradeoff at training time.

  3. Average inference time per window (ms), timed around the same
     run_window() call train.py's own validation loop uses - not a separate
     hand-rolled forward dispatch - so a future model addition to run_window
     is automatically picked up here too, the way evaluate.py's now-stale
     hand-rolled dispatch (missing fxb for several *_fxb models) was not.

  4. Overfit/underfit and "how training was left" diagnostics, read straight
     off the training curve already logged in each run's log.csv - no extra
     forward passes needed for this part. Val-only by design (not
     train_loss): train_loss is not comparable across --loss choices (see
     the --grid-loss help text in train_tartanair.py), while val EPE is.
     epochs_since_best is 0 when the run's last evaluation was also its best
     one, i.e. it was still improving when it stopped (more epochs/patience
     may help - underfit at the cutoff); a large positive overfit_gap_pct
     with epochs_since_best > 0 means val EPE got measurably worse after its
     peak, the textbook overfitting signature. See log_csv_fit_stats().

Writes one row per run to --out (default <runs-dir>/eval_all.csv).

Usage:
  python eval_all.py --runs-dir runs/tartanair_grid
  python eval_all.py --runs-dir runs/tartanair_grid --max-eval-windows 500 --models siam2d_2dun_fxb
  python eval_all.py --runs-dir runs/tartanair_grid --limit 3          # smoke test
"""

import argparse
import csv
import math
import os
import random
import time
import traceback

import torch
from torch.utils.data import DataLoader, Subset

from stereo_datasets import TartanAirDataset
from models import build_model
from train import run_window, criterion_from_args
from train_tartanair import _parse_run_name, _log_csv_epochs_done, environment_split

CSV_FIELDS = [
    "model", "batch_size", "lr", "loss", "run_dir",
    "epochs_done", "expected_epochs", "has_best", "has_last", "healthy",
    "best_epoch", "last_eval_epoch", "epochs_since_best",
    "val_epe_at_best", "val_epe_final", "overfit_gap_epe", "overfit_gap_pct",
    "fit_note",
    "held_out_env_same", "held_out_env_diff",
    "best_same_epe", "best_same_d1", "best_same_depth_mae", "best_same_n", "best_same_infer_ms",
    "best_diff_epe", "best_diff_d1", "best_diff_depth_mae", "best_diff_n", "best_diff_infer_ms",
    "last_same_epe", "last_same_d1", "last_same_depth_mae", "last_same_n", "last_same_infer_ms",
    "last_diff_epe", "last_diff_d1", "last_diff_depth_mae", "last_diff_n", "last_diff_infer_ms",
    "avg_infer_ms", "error",
]


def alt_environment_split(dataset, val_frac, exclude_envs, seed):
    """Same by-environment holdout as train_tartanair.environment_split, but
    a differently-seeded random choice of which environment(s) to hold out,
    excluding `exclude_envs` so the result is guaranteed disjoint from
    whatever the "same" split already held out (rather than merely likely
    to be, which a second call with a different seed alone would only give
    you most of the time). Returns (val_subset, held_out_env_names), or
    (None, []) if there is only one environment total and so nothing else to
    hold out.
    """
    envs = sorted({s["env"] for s in dataset.sequences})
    candidates = [e for e in envs if e not in exclude_envs]
    if not candidates:
        return None, []
    k = min(max(1, math.ceil(val_frac * len(envs))), len(candidates))
    rng = random.Random(seed)
    val = set(rng.sample(candidates, k))
    val_sids = {sid for sid, s in enumerate(dataset.sequences) if s["env"] in val}
    va = [pos for pos, (sid, _) in enumerate(dataset.index) if sid in val_sids]
    return Subset(dataset, va), sorted(val)


def cap_windows(subset, n, seed):
    """Deterministically take at most `n` windows from `subset`."""
    if subset is None or n is None or len(subset) <= n:
        return subset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(subset), generator=g)[:n].tolist()
    return Subset(subset, idx)


class _DatasetCache:
    """TartanAirDataset construction (walking the filesystem, reading every
    pose file) is the expensive part of evaluating a run - reuse one instance
    across every run that shares the same (data, camera, difficulty, envs,
    scale, window, frame_stride, crop, max_disp), which in a grid search is
    most of them."""

    def __init__(self):
        self._cache = {}

    def get(self, data, camera, difficulty, envs, scale, window, frame_stride,
            crop, max_disp):
        key = (tuple(data), camera, difficulty,
               tuple(envs) if envs else None, scale, window, frame_stride,
               crop, max_disp)
        if key not in self._cache:
            crop_hw = tuple(int(x) for x in crop.split("x")) if crop else None
            self._cache[key] = TartanAirDataset(
                data, camera=camera, difficulty=difficulty, envs=envs,
                scale=scale, window=window, frame_stride=frame_stride,
                crop=crop_hw, augment=True, max_disp=max_disp)
        return self._cache[key]


_FIT_STATS_EMPTY = {k: "" for k in (
    "best_epoch", "last_eval_epoch", "epochs_since_best",
    "val_epe_at_best", "val_epe_final", "overfit_gap_epe", "overfit_gap_pct",
    "fit_note")}


def log_csv_fit_stats(log_csv_path):
    """Overfit/underfit diagnostics and "how far past its best point did
    training run" from a run's log.csv alone - the same file summary.csv's
    best_val_epe/best_epoch already come from, but derived directly here so
    this holds even for a run --grid-summarize would call FAILED/unhealthy
    (any epoch that got as far as one evaluation still has something to say).

    best is the minimum logged val_epe (matches best.pth, see fit() in
    train.py); last_eval_epoch is the epoch of the LAST logged evaluation,
    which is not necessarily epochs_done - 1: a run cut short before its
    configured final epoch may never have evaluated its last completed epoch
    at all (fit()'s do_eval only forces an eval on --eval-every multiples and
    on the configured args.epochs - 1, not on whatever epoch a crash happened
    to leave off at).

    epochs_since_best = last_eval_epoch - best_epoch: 0 means the run was
    still improving (or at least not yet past its peak) at the point it
    stopped - more epochs/patience might have helped further. A large value
    means training continued well past its best point.

    overfit_gap_epe/_pct = val_epe_final - val_epe_at_best: positive means
    val EPE got worse after its best point, i.e. actual overfitting rather
    than a plateau. Deliberately val-only, not train_loss vs val_epe: the
    two are not on the same scale, and train_loss itself is not comparable
    across --loss choices (see --grid-loss's help text), while val EPE is.
    """
    if not os.path.exists(log_csv_path):
        return dict(_FIT_STATS_EMPTY)
    best = best_epoch = last_eval_epoch = final_epe = None
    with open(log_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get("val_epe", "")
            if v == "":
                continue
            v = float(v)
            epoch = int(row["epoch"])
            last_eval_epoch, final_epe = epoch, v
            if best is None or v < best:
                best, best_epoch = v, epoch
    if best is None:
        return dict(_FIT_STATS_EMPTY)

    since_best = last_eval_epoch - best_epoch
    gap = final_epe - best
    gap_pct = (100.0 * gap / best) if best > 0 else 0.0
    if since_best == 0:
        note = "still improving at cutoff (more epochs/patience may help)"
    elif gap_pct < 2.0:
        note = (f"plateaued {since_best} epoch(s) after its best "
                f"(epoch {best_epoch}), no real overfitting")
    else:
        note = (f"overfitting: val EPE {gap_pct:.1f}% worse than its best, "
                f"{since_best} epoch(s) after epoch {best_epoch}")
    return {"best_epoch": best_epoch, "last_eval_epoch": last_eval_epoch,
            "epochs_since_best": since_best,
            "val_epe_at_best": f"{best:.4f}", "val_epe_final": f"{final_epe:.4f}",
            "overfit_gap_epe": f"{gap:.4f}", "overfit_gap_pct": f"{gap_pct:.2f}",
            "fit_note": note}


def write_csv(out_path, rows):
    """Rewrite the whole CSV after every run, not just at the end - a scan
    over ~180 runs takes hours, and a crash or Ctrl-C partway through should
    not throw away everything scanned so far."""
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def default_window(model_name):
    return {"siam2d_egomotion_fxb": 4, "c3d_3dhg_10_fxb": 10,
            "c3d_3dhg_3_fxb": 3}.get(model_name, 1)


def evaluate_split(model, model_name, subset, device, eval_bs, workers):
    """Mean EPE / D1 / depth-MAE and avg inference ms/window over `subset`,
    using train.py's own run_window() so the per-model forward dispatch (and
    any future model added to it) is exercised exactly as at training time.
    """
    loader = DataLoader(subset, batch_size=eval_bs, shuffle=False,
                        num_workers=workers)
    ns = argparse.Namespace(model=model_name)
    criterion = criterion_from_args(ns)
    sums = [0.0] * 5
    total_time, total_windows = 0.0, 0
    model.eval()
    with torch.no_grad():
        # One untimed pass over the first batch: cuDNN's algorithm search and
        # CUDA's own lazy init otherwise land entirely on whichever batch
        # happens to run first, skewing that one measurement by an order of
        # magnitude relative to the rest (see the smoke-test run that first
        # surfaced this: 71ms vs ~35ms for identical batches).
        for warmup_batch in loader:
            run_window(model, ns, warmup_batch, device, train=False,
                      criterion=criterion)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            break
        for batch in loader:
            n = batch["left"].shape[0]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, m = run_window(model, ns, batch, device, train=False,
                              criterion=criterion)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            total_windows += n
            sums = [a + b for a, b in zip(sums, m)]
    epe = sums[0] / max(sums[3], 1)
    d1 = 100.0 * sums[1] / max(sums[3], 1)
    mae = sums[2] / max(sums[4], 1)
    infer_ms = (total_time / max(total_windows, 1)) * 1000.0
    return epe, d1, mae, total_windows, infer_ms


def load_model(model_name, ckpt_path, ck_args, device):
    model = build_model(model_name, max_disp=ck_args.get("max_disp", 128),
                        yolo_scale=ck_args.get("yolo_scale", "n")).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs/tartanair_grid")
    ap.add_argument("--data", nargs="+", default=None,
                    help="override the TartanAir root(s) recorded in each "
                         "checkpoint (default: use each checkpoint's own "
                         "--data, as saved in ck['args'])")
    ap.add_argument("--out", default=None,
                    help="default: <runs-dir>/eval_all.csv")
    ap.add_argument("--expected-epochs", type=int, default=20)
    ap.add_argument("--max-eval-windows", type=int, default=200,
                    help="cap per split so a full-grid scan stays tractable; "
                         "0 or negative disables the cap")
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--diff-seed", type=int, default=1337,
                    help="seed for choosing the 'different' split's held-out "
                         "environment(s)")
    ap.add_argument("--cap-seed", type=int, default=123,
                    help="seed for subsampling each split down to "
                         "--max-eval-windows; fixed (not per-run) so best.pth "
                         "and last.pth of the same run see identical windows")
    ap.add_argument("--models", nargs="+", default=None,
                    help="only evaluate these --model names")
    ap.add_argument("--only", default=None,
                    help="only evaluate run directories containing this substring")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many run directories (smoke test)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out or os.path.join(args.runs_dir, "eval_all.csv")
    max_win = args.max_eval_windows if args.max_eval_windows and args.max_eval_windows > 0 else None

    run_names = sorted(
        n for n in os.listdir(args.runs_dir)
        if os.path.isdir(os.path.join(args.runs_dir, n)) and _parse_run_name(n)
    )
    if args.models:
        run_names = [n for n in run_names if _parse_run_name(n)[0] in args.models]
    if args.only:
        run_names = [n for n in run_names if args.only in n]
    if args.limit:
        run_names = run_names[:args.limit]

    cache = _DatasetCache()
    rows = []
    for i, name in enumerate(run_names, 1):
        model_name, bs, lr, loss = _parse_run_name(name)
        run_dir = os.path.join(args.runs_dir, name)
        log_csv_path = os.path.join(run_dir, "log.csv")
        best_path = os.path.join(run_dir, "best.pth")
        last_path = os.path.join(run_dir, "last.pth")
        epochs_done = _log_csv_epochs_done(log_csv_path)
        has_best, has_last = os.path.exists(best_path), os.path.exists(last_path)
        healthy = epochs_done >= args.expected_epochs and has_best and has_last

        row = {"model": model_name, "batch_size": bs, "lr": lr, "loss": loss,
               "run_dir": name, "epochs_done": epochs_done,
               "expected_epochs": args.expected_epochs,
               "has_best": has_best, "has_last": has_last, "healthy": healthy,
               "error": ""}
        row.update(log_csv_fit_stats(log_csv_path))
        print(f"[{i}/{len(run_names)}] {name}  epochs={epochs_done}/"
              f"{args.expected_epochs}  best={has_best} last={has_last}"
              + ("" if healthy else "  ** UNHEALTHY **")
              + (f"  ({row['fit_note']})" if row["fit_note"] else ""))

        if not has_best and not has_last:
            rows.append(row)
            write_csv(out_path, rows)
            continue

        try:
            ck_probe_path = best_path if has_best else last_path
            ck_args = torch.load(ck_probe_path, map_location="cpu",
                                 weights_only=False).get("args", {})
            data = args.data or ck_args.get("data")
            if not data:
                raise RuntimeError("no --data override given and none found "
                                   "in the checkpoint's saved args")
            window = ck_args.get("window") or default_window(model_name)
            dataset = cache.get(
                data=data, camera=ck_args.get("camera", "front"),
                difficulty=ck_args.get("difficulty"), envs=ck_args.get("envs"),
                scale=ck_args.get("scale", 1.0), window=window,
                frame_stride=ck_args.get("frame_stride", 1),
                crop=ck_args.get("crop", "480x640"),
                max_disp=ck_args.get("max_disp", 128))

            _, val_same, held_same = environment_split(
                dataset, ck_args.get("val_frac", 0.15), ck_args.get("val_envs"))
            val_diff, held_diff = alt_environment_split(
                dataset, ck_args.get("val_frac", 0.15), held_same, args.diff_seed)

            val_same = cap_windows(val_same, max_win, args.cap_seed)
            val_diff = cap_windows(val_diff, max_win, args.cap_seed)
            row["held_out_env_same"] = ",".join(held_same) if held_same else ""
            row["held_out_env_diff"] = ",".join(held_diff) if held_diff else ""

            all_ms = []
            for tag, ckpt_path, has in (("best", best_path, has_best),
                                        ("last", last_path, has_last)):
                if not has:
                    continue
                model = load_model(model_name, ckpt_path, ck_args, device)
                for split_tag, subset in (("same", val_same), ("diff", val_diff)):
                    if subset is None or len(subset) == 0:
                        continue
                    epe, d1, mae, n, ms = evaluate_split(
                        model, model_name, subset, device, args.eval_bs,
                        args.workers)
                    row[f"{tag}_{split_tag}_epe"] = f"{epe:.4f}"
                    row[f"{tag}_{split_tag}_d1"] = f"{d1:.4f}"
                    row[f"{tag}_{split_tag}_depth_mae"] = f"{mae:.4f}"
                    row[f"{tag}_{split_tag}_n"] = n
                    row[f"{tag}_{split_tag}_infer_ms"] = f"{ms:.2f}"
                    all_ms.append(ms)
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            if all_ms:
                row["avg_infer_ms"] = f"{sum(all_ms) / len(all_ms):.2f}"
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  !!! {row['error']}")
            traceback.print_exc()

        rows.append(row)
        write_csv(out_path, rows)

    n_healthy = sum(1 for r in rows if r["healthy"])
    print(f"\n=== {len(rows)} runs scanned, {n_healthy} healthy, "
          f"written to {out_path} ===")


if __name__ == "__main__":
    main()
