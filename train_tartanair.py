"""Train the baseline (FastStereoNet) or temporal (TempoBandNet) model on TartanAir.

Companion to train.py, which trains on the AirSim sequences recorded by
collect_dataset.py. The optimisation is *identical* - loss, schedule, metrics
and checkpoint format are imported from train.py, so runs from the two scripts
are directly comparable. Only the data differs:

  * layout      - TartanAir V1 and V2 trees are discovered automatically
                  (see stereo_datasets.TartanAirDataset)
  * calibration - fixed by the dataset (fx = fy = 320, baseline 0.25 m), not
                  read from a per-sequence calib.json
  * validation  - held out by *environment*, not by sequence. Trajectories
                  inside one TartanAir environment revisit the same geometry
                  and textures, so a by-trajectory split leaks.

Getting the data (V2, https://tartanair.org):
    pip install tartanair
    python -c "import tartanair as ta; ta.init('$HOME/datasets/tartanair'); \
        ta.download(env=['AbandonedFactory'], difficulty=['easy'], \
        modality=['image','depth'], camera_name=['lcam_front','rcam_front'], unzip=True)"

Examples:
    # baseline, single frames, held-out environments for validation
    python train_tartanair.py --model baseline --data ~/datasets/tartanair \
        --crop 480x640 --out runs/tartanair_baseline

    # temporal, windows of 4 frames with simulated VIO noise
    python train_tartanair.py --model temporal --window 4 --bs 4 \
        --data ~/datasets/tartanair --crop 480x640 --pose-noise 0.3 \
        --out runs/tartanair_temporal

    # pre-train here, then fine-tune on the AirSim rig with train.py --resume
    python train_tartanair.py --model baseline --data ~/datasets/tartanair \
        --out runs/tartanair_pre
    python train.py --model baseline --data ~/datasets/airsim_stereo \
        --out runs/finetune --init runs/tartanair_pre/best.pth

    # yolo (a YOLO26 encoder/decoder around the same cost volume) with the
    # log-space objective; --crop must be a multiple of 32 for this model
    python train_tartanair.py --model yolo --loss logl1 --crop 480x640 \
        --data ~/datasets/tartanair --out runs/tartanair_yolo

    # grid search: sweep --grid-models x --grid-bs x --grid-lr x --grid-loss,
    # each combo a fresh subprocess (so a CUDA OOM in one can't corrupt the
    # next), skipping combos already completed by an earlier interrupted sweep
    # and resuming ones that stopped part-way. The defaults are the full
    # comparison - 5 models x bs {8,16} x lr {1e-3,3e-3,1e-2} x 4 losses,
    # minus the combos measured not to fit in 24GB - so this is the whole
    # command:
    python train_tartanair.py --grid-search --data ~/datasets/tartanair \
        --amp --out runs/tartanair_grid

    # compare objectives across the existing models *before* comparing
    # architectures, so a later result cannot confound the two
    python train_tartanair.py --grid-search --data ~/datasets/tartanair \
        --grid-models baseline temporal mobilenet anynet \
        --grid-loss smoothl1 depth26 logl1 hybrid \
        --grid-bs 8 --grid-lr 3e-4 --grid-epochs 10 --crop 480x640 --amp \
        --out runs/loss_ablation

A note on the domain gap. TartanAir's 0.25 m baseline gives fx*b = 80 px*m,
against roughly 23 px*m for this project's 6.3 cm rig at 640x480 / 83 deg FOV -
TartanAir disparities are ~3.5x larger at the same depth. The network only ever
sees images, so it learns whatever disparity statistics it is fed. Use --scale
to shrink fx (and disparity with it) toward the target rig: --scale 0.29 lands
fx*b near 23, at the cost of resolution. Leaving --scale 1 is the right choice
when TartanAir is a pre-training or benchmarking corpus rather than a stand-in
for the real camera.
"""

import argparse
import csv
import itertools
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from stereo_datasets import TartanAirDataset
from losses import LOSS_CHOICES, add_loss_args
from train import fit, sequence_split, log_exception, _now
from models import build_model, count_parameters

MODEL_CHOICES = ["baseline", "stereoconv", "stereoconv3d", "stereoconv3d_fast",
                 "temporal", "mobilenet", "anynet", "yolo"]
# depth26 stays a valid --loss/--grid-loss choice (LOSS_CHOICES, losses.py),
# but is no longer swept by default: every combo tried so far gave far worse
# EPE and depth MAE than the other three losses (its SILog term is
# scale-invariant, the wrong choice when stereo's fxb already gives true
# metric scale - see losses.py and PAPER.md Section 14). Pass
# `--grid-loss ... depth26` explicitly to include it again.
GRID_LOSS_DEFAULT = [l for l in LOSS_CHOICES if l != "depth26"]
DEFAULT_LOSS = "smoothl1"


def environment_split(dataset, val_frac, val_envs=None):
    """Split window indices by environment.

    Returns (train_subset, val_subset, held_out_env_names). With a single
    environment there is nothing to hold out, so this falls back to train.py's
    by-sequence split and reports no held-out environments.
    """
    envs = sorted({s["env"] for s in dataset.sequences})
    if val_envs:
        unknown = set(val_envs) - set(envs)
        if unknown:
            raise SystemExit(f"--val-envs not present in the data: {sorted(unknown)}\n"
                             f"available: {envs}")
        val = set(val_envs)
    elif len(envs) > 1:
        val = set(envs[-max(1, math.ceil(val_frac * len(envs))):])
    else:
        train_set, val_set = sequence_split(dataset, val_frac)
        return train_set, val_set, []

    val_sids = {sid for sid, s in enumerate(dataset.sequences) if s["env"] in val}
    tr, va = [], []
    for pos, (sid, _) in enumerate(dataset.index):
        (va if sid in val_sids else tr).append(pos)
    return Subset(dataset, tr), Subset(dataset, va), sorted(val)


def set_seed(seed, deterministic=False):
    """Seed every RNG this training path touches, so a config is reproducible.

    Called once at the start of each run - and each grid-search combo is its
    own subprocess, so every combo starts from the identical RNG state and
    differences between them are attributable to the config rather than to
    luck. Covers python's `random` (dataset augmentation), numpy (used inside
    stereo_datasets) and torch CPU+CUDA (weight init, dropout, the subsample
    permutations). Returns a torch.Generator for the DataLoader so shuffling
    is reproducible too.

    cudnn.benchmark is left off deliberately: its autotuner picks algorithms by
    timing, which varies run to run and would reintroduce nondeterminism.

    Scope of the guarantee, measured rather than assumed: with this seeding the
    data subset, sample order, augmentation and weight init are identical
    across runs, and on CPU the results are bit-identical. On CUDA they are
    not - bilinear interpolate's backward and the cost-volume scatter both
    accumulate with atomicAdd, whose summation order varies between runs, so
    two identically seeded GPU runs drift slightly. `deterministic=True` asks
    torch to refuse nondeterministic kernels instead; it makes runs bit-exact
    where every op has a deterministic implementation and raises a clear error
    naming the op where one does not, which is why it is opt-in.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic:
        # cuBLAS needs this to make its reductions reproducible; it must be set
        # before the first CUDA context use to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    """Re-seed each DataLoader worker deterministically from torch's per-worker
    seed, which set_seed's base seed makes reproducible."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def subsample(subset, n, seed, label):
    """Deterministically take `n` windows from `subset` (no-op if already
    smaller). Same seed on every combo => every run in a sweep trains and
    validates on exactly the same data, which is what makes the comparison
    fair rather than a lottery over different slices."""
    full_n = len(subset)
    if n is None or full_n <= n:
        return subset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(full_n, generator=g)[:n].tolist()
    out = Subset(subset, idx)
    print(f"  capped {label}: {full_n} -> {len(out)} windows (fixed seed {seed})")
    return out


def describe_calibration(dataset, max_disp):
    """Print the disparity scale the model is about to be trained on."""
    calib = dataset.sequences[0]["calib"]
    w, h = dataset.sequences[0]["size"]
    fxb = calib["fx"] * calib["baseline_m"]
    print(f"images {w}x{h}  fx={calib['fx']:.1f}  baseline={calib['baseline_m']}m"
          f"  ->  fx*b={fxb:.1f} px*m")
    print(f"  disparity at 2m / 10m / 30m: {fxb / 2:.1f} / {fxb / 10:.1f} / "
          f"{fxb / 30:.2f} px   (max-disp {max_disp} clips below "
          f"{fxb / max_disp:.2f} m)")


# --------------------------------------------------------------------------
# Grid search: sweeps --grid-models x --grid-bs x --grid-lr, each combo run
# as its own `python train_tartanair.py ...` subprocess (not in-process) so a
# CUDA OOM or crash in one combo can't corrupt the CUDA context of the next -
# the same isolation the old grid_search_tartanair.sh shell script relied on.
# --------------------------------------------------------------------------

# (dest on args, CLI flag, kind) for settings shared by every combo in a
# sweep - forwarded to each subprocess exactly as the user passed them.
_GRID_PASSTHROUGH_ARGS = [
    ("data", "--data", "list"),
    # note: --epochs is NOT forwarded from args.epochs; the sweep sends
    # args.grid_epochs instead (see run_grid_search).
    ("window", "--window", "value"),
    ("frame_stride", "--frame-stride", "value"),
    ("crop", "--crop", "value"),
    ("max_disp", "--max-disp", "value"),
    ("scale", "--scale", "value"),
    ("camera", "--camera", "value"),
    ("difficulty", "--difficulty", "value"),
    ("envs", "--envs", "list"),
    ("val_envs", "--val-envs", "list"),
    ("val_frac", "--val-frac", "value"),
    ("pose_noise", "--pose-noise", "value"),
    ("pose_noise_trans", "--pose-noise-trans", "value"),
    ("workers", "--workers", "value"),
    ("amp", "--amp", "flag"),
    ("print_every", "--print-every", "value"),
    ("eval_every", "--eval-every", "value"),
    ("patience", "--patience", "value"),
    ("max_train_windows", "--max-train-windows", "value"),
    ("max_val_windows", "--max-val-windows", "value"),
    ("data_fraction", "--data-fraction", "value"),
    ("seed", "--seed", "value"),
    # --loss itself is NOT here: like --epochs it varies per combo and is sent
    # explicitly by run_grid_search. Its weights are shared by the whole sweep.
    ("loss_w_log", "--loss-w-log", "value"),
    ("loss_w_grad", "--loss-w-grad", "value"),
    ("loss_silog_lambda", "--loss-silog-lambda", "value"),
    ("yolo_scale", "--yolo-scale", "value"),
]

# depth MAE rides alongside EPE because they can disagree: a scale-invariant
# objective (--loss depth26) can be structurally good while its absolute
# disparity, and so its EPE, drifts. Ranking on EPE alone would drop it for the
# wrong reason.
_SUMMARY_FIELDS = ["model", "batch_size", "lr", "loss", "status", "best_val_epe",
                   "final_val_epe", "best_val_depth_mae", "seconds"]


def _passthrough_argv(args):
    argv = []
    for dest, flag, kind in _GRID_PASSTHROUGH_ARGS:
        val = getattr(args, dest, None)
        if val is None:
            continue
        if kind == "flag":
            if val:
                argv.append(flag)
        elif kind == "list":
            if val:
                argv += [flag, *[str(v) for v in val]]
        else:
            argv += [flag, str(val)]
    return argv


def _log_csv_epochs_done(log_csv_path):
    if not os.path.exists(log_csv_path):
        return 0
    with open(log_csv_path) as f:
        return max(0, sum(1 for _ in f) - 1)  # rows minus the header


def _log_csv_epes(log_csv_path):
    """(best_epe, final_epe, best_depth_mae) from a run's log.csv, each "" if
    unavailable - blank sorts last in the summary, same convention the shell
    version used.

    `best` is the minimum over all epochs, matching what best.pth holds.
    `final` is the last epoch that actually evaluated; fit() forces an eval on
    the final epoch regardless of --eval-every, so for a run that finished
    this is the end-of-training number rather than whatever the last
    --eval-every multiple happened to be.

    best_depth_mae is tracked separately rather than read off the best-EPE
    epoch: the two metrics need not bottom out together, and reporting each
    one's own best is what makes the two columns independently readable.
    """
    if not os.path.exists(log_csv_path):
        return "", "", ""
    best = final = best_mae = None
    with open(log_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mae = row.get("val_depth_mae_30m", "")
            if mae != "":
                mae = float(mae)
                best_mae = mae if best_mae is None else min(best_mae, mae)
            v = row.get("val_epe", "")
            if v == "":
                continue
            v = float(v)
            best = v if best is None else min(best, v)
            final = v  # rows are written in epoch order, so the last wins
    return ("" if best is None else f"{best:.4f}",
            "" if final is None else f"{final:.4f}",
            "" if best_mae is None else f"{best_mae:.4f}")


def _update_summary(summary_path, model, bs, lr, loss, row):
    """Rewrite summary.csv with `row` replacing any existing row for this
    (model,bs,lr,loss), so re-running a sweep updates in place instead of
    piling up duplicate rows across restarts. Returns the full row list.

    Rows written before --loss existed have no `loss` column; they are read as
    the default so an older summary.csv still parses and still matches.
    """
    rows = []
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            for r in csv.DictReader(f):
                r["loss"] = r.get("loss") or DEFAULT_LOSS
                if (r["model"] == model and r["batch_size"] == str(bs)
                        and r["lr"] == lr and r["loss"] == loss):
                    continue  # superseded by `row`
                rows.append(r)
    rows.append(row)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def _lan_ip():
    """Best-effort LAN-facing IP (no packets sent - a UDP connect just picks
    the outbound interface) so a tensorboard --bind_all URL can be printed."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        try:
            s.close()
        except Exception:
            pass


# Measured peak VRAM per sample at 480x640 with --amp (see PAPER/benchmarks):
# baseline 0.34, anynet 0.33, mobilenet 0.60, yolo ~0.39, temporal 1.41 GB.
# temporal is ~4x the rest purely because window=4 backprops through 4 frames.
_GB_PER_SAMPLE = {"baseline": 0.34, "anynet": 0.33, "mobilenet": 0.60,
                  "yolo": 0.39, "temporal": 1.41}
# The log-space losses build multi-scale gradient tensors on top of the model's
# own activations; measured to push temporal bs=16 from 22.5GB over the 24GB
# edge, so they carry a surcharge when deciding feasibility.
_LOG_LOSSES = {"depth26", "logl1", "hybrid"}
_LOSS_VRAM_SURCHARGE = 1.12
# Calibrated against measurements on a 23.5GB-usable 4090: temporal bs=16
# smoothl1 peaks at 22.5GB and does run, while the same config on logl1
# (est. 25.3GB) OOMs. 23.0 is the threshold that reproduces both outcomes.
# It is deliberately tight rather than safe - a combo that squeaks in and then
# OOMs is caught and reported by the run loop, whereas one wrongly excluded
# here is silently never measured at all.
_VRAM_BUDGET_GB = 23.0


def combo_fits(model, bs, loss, budget_gb=_VRAM_BUDGET_GB):
    """Predict whether a combo fits in VRAM, from measured per-sample cost.

    Cheap and deterministic, and it beats discovering the same OOM every sweep:
    a combo that cannot fit is never scheduled, so it does not burn a slot or
    clutter the summary with a FAILED row that carries no information.
    """
    per = _GB_PER_SAMPLE.get(model)
    if per is None:
        return True  # unknown model: let it run and report honestly if it OOMs
    est = per * bs * (_LOSS_VRAM_SURCHARGE if loss in _LOG_LOSSES else 1.0)
    return est <= budget_gb


def build_combos(args):
    """Expand the sweep axes into an explicit, ordered list of runnable combos.

    The axes are still a convenient way to *say* what you want, but the list is
    what actually runs: combos predicted not to fit in VRAM are dropped up
    front rather than scheduled and left to OOM. Returns (combos, skipped).
    """
    combos, skipped = [], []
    for c in itertools.product(args.grid_models, args.grid_bs, args.grid_lr,
                               args.grid_loss):
        model, bs, _, loss = c
        if args.grid_skip_infeasible and not combo_fits(model, bs, loss):
            skipped.append(c)
        else:
            combos.append(c)
    return combos, skipped


def _run_hit_oom(stdout_path):
    """Did this run die of CUDA OOM? Decides whether resuming it could ever
    succeed: an OOM will simply recur, anything else might be transient."""
    try:
        with open(stdout_path, errors="replace") as f:
            tail = f.read()[-8000:]
    except OSError:
        return False
    return "OutOfMemoryError" in tail or "CUDA out of memory" in tail


def _cleanup_failed_run(run_out, drop_checkpoints, slog=print):
    """Tidy up after a failed combo so it cannot poison the next one.

    Each combo is a subprocess, so its VRAM is already returned by the OS when
    it dies - this handles what the OS does not: a partial checkpoint that
    auto-resume would otherwise pick up and re-crash on, and the wait for
    memory to actually show as free before the next run starts.
    """
    if drop_checkpoints:
        # An OOM recurs on resume, so the checkpoint is worse than useless:
        # keep it as .failed for inspection rather than letting resume find it.
        for name in ("last.pth", "best.pth"):
            p = os.path.join(run_out, name)
            if os.path.exists(p):
                try:
                    shutil.move(p, p + ".failed")
                except OSError:
                    pass
    free = _free_vram_gb()
    if free is not None:
        slog(f"    cleanup: {free:.1f}GB VRAM free before next run")


def _free_vram_gb():
    try:
        free, _ = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return None


def _start_tensorboard(out_root, port):
    """Launch tensorboard for the sweep as a child process.

    Returns (proc, url) or (None, None). Started as `sys.executable -m
    tensorboard.main` rather than the `tensorboard` console script: the usual
    way to launch a sweep is `path/to/venv/bin/python3 train_tartanair.py`
    *without* activating the venv, in which case the console script is not on
    PATH but the module is importable by this very interpreter.

    --host 0.0.0.0 listens on every IPv4 interface rather than just localhost,
    which is what lets another machine on the LAN reach it; it is used in
    preference to --bind_all because --bind_all also wants an IPv6 socket and
    dies with "could not bind to unsupported address family ::" on hosts with
    IPv6 disabled. Failure here is never fatal: the sweep is the point, the
    dashboard is a convenience.
    """
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("note: tensorboard not installed in this environment; skipping "
              "dashboard (pip install tensorboard)")
        return None, None
    err_path = os.path.join(out_root, "tensorboard.log")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", os.path.abspath(out_root), "--host", "0.0.0.0",
             "--port", str(port)],
            stdout=open(err_path, "w"), stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"note: could not start tensorboard ({e}); continuing without it")
        return None, None

    time.sleep(3)  # give it a moment to bind, so a failure surfaces here
    if proc.poll() is not None:
        reason = ""
        try:
            with open(err_path) as f:
                errs = [l.strip() for l in f if l.strip().startswith("ERROR")]
            reason = f": {errs[-1]}" if errs else ""
        except OSError:
            pass
        print(f"note: tensorboard exited immediately{reason}\n"
              f"      (full log: {err_path}); continuing without it")
        return None, None
    ip = _lan_ip()
    return proc, f"http://{ip or 'localhost'}:{port}"


def _sorted_rows(rows):
    """Rows ranked by depth MAE, best first; no-metric rows last.

    Depth MAE rather than EPE on purpose: the log-space losses deliberately
    stop spending gradient on the near field, where disparity error is cheap,
    so they can score worse on EPE while being better at the depth accuracy
    obstacle avoidance actually needs (losses.py). Ranking a loss comparison
    by EPE would systematically pick the wrong winner.
    """
    def key(r):
        try:
            return float(r.get("best_val_depth_mae", ""))
        except (TypeError, ValueError):
            return float("inf")
    return sorted(rows, key=key)


def _winner(rows):
    """Best completed run, or None if nothing finished with a metric."""
    for r in _sorted_rows(rows):
        if r.get("status") in ("OK", "SKIPPED") and r.get("best_val_depth_mae"):
            return r
    return None


def _final_run_command(win, args, out_root):
    """The command to train the winning config properly: all the data, a long
    schedule, and early stopping so it stops when it stops improving rather
    than burning the full 100 epochs regardless."""
    if win is None:
        return None
    name = f"{win['model']}_bs{win['batch_size']}_lr{win['lr']}_{win['loss']}_final"
    # Absolute paths for both: the interpreter because it is usually a venv
    # that is not on PATH, and the script so the command runs from any cwd.
    parts = [
        sys.executable,
        os.path.abspath(__file__),
        f"--data {' '.join(args.data)}",
        f"--model {win['model']}", f"--bs {win['batch_size']}",
        f"--lr {win['lr']}", f"--loss {win['loss']}",
        "--data-fraction 1.0",           # the sweep's fraction is a shortcut; the real run uses everything
        f"--epochs {args.final_epochs}",
        f"--patience {args.final_patience}",
        f"--eval-every {args.eval_every}",
        f"--seed {args.seed}",
        f"--workers {args.workers}",
    ]
    if args.crop:
        parts.append(f"--crop {args.crop}")
    if args.amp:
        parts.append("--amp")
    parts.append(f"--out {os.path.join(os.path.dirname(out_root) or '.', name)}")
    return " \\\n    ".join(parts)


def write_report(rows, skipped, args, out_root, elapsed_s, tb_url=None):
    """Write report.md next to the runs: the ranked table, what was skipped and
    why, and the command to train the winner for real. Markdown so it can be
    pasted into a PR or a lab notebook without reformatting."""
    rows = _sorted_rows(rows)
    win = _winner(rows)
    path = os.path.join(out_root, "report.md")
    # SKIPPED means "already complete from an earlier sweep", so it counts as
    # a finished run with results, not as something that did not happen.
    ok = [r for r in rows if r.get("status") in ("OK", "SKIPPED")]
    failed = [r for r in rows if r.get("status") in ("FAILED", "OOM")]

    L = []
    L.append(f"# Grid search report: {os.path.basename(os.path.normpath(out_root))}")
    L.append("")
    L.append(f"- generated: {_now()}")
    L.append(f"- wall clock: {_fmt_hms(elapsed_s)}")
    L.append(f"- data: `{' '.join(args.data)}`")
    L.append(f"- epochs/combo: {args.grid_epochs}   eval every: {args.eval_every}"
             f"   seed: {args.seed}")
    L.append(f"- data fraction: {args.data_fraction}   crop: {args.crop}"
             f"   amp: {bool(args.amp)}   workers: {args.workers}")
    L.append(f"- axes: models={args.grid_models} bs={args.grid_bs} "
             f"lr={args.grid_lr} loss={args.grid_loss}")
    L.append(f"- {len(ok)} completed, {len(failed)} failed, {len(skipped)} "
             f"skipped as infeasible")
    L.append("")
    L.append("Ranked by **val depth MAE < 30 m** (lower is better). EPE is shown "
             "too but is not the ranking metric: the log-space losses trade "
             "near-field disparity accuracy for far-field depth accuracy, so "
             "EPE would favour the wrong objective (see losses.py).")
    L.append("")
    hdr = ["#", "model", "bs", "lr", "loss", "status", "depth MAE (m)",
           "best EPE (px)", "final EPE (px)", "sec"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join("---" for _ in hdr) + "|")
    for i, r in enumerate(rows, 1):
        L.append("| " + " | ".join([
            str(i), r.get("model", ""), r.get("batch_size", ""), r.get("lr", ""),
            r.get("loss", ""), r.get("status", ""),
            r.get("best_val_depth_mae", "") or "-",
            r.get("best_val_epe", "") or "-", r.get("final_val_epe", "") or "-",
            r.get("seconds", ""),
        ]) + " |")
    L.append("")

    if skipped:
        L.append("## Skipped as infeasible")
        L.append("")
        L.append("Predicted to exceed the VRAM budget from measured per-sample "
                 "cost, so never scheduled (`--no-grid-skip-infeasible` to try "
                 "them anyway):")
        L.append("")
        for model, bs, lr, loss in skipped:
            est = _GB_PER_SAMPLE.get(model, 0) * bs * (
                _LOSS_VRAM_SURCHARGE if loss in _LOG_LOSSES else 1.0)
            L.append(f"- `{model} bs={bs} lr={lr} loss={loss}` - ~{est:.1f} GB "
                     f"needed > {_VRAM_BUDGET_GB} GB budget")
        L.append("")

    if failed:
        L.append("## Failed")
        L.append("")
        for r in failed:
            L.append(f"- `{r['model']} bs={r['batch_size']} lr={r['lr']} "
                     f"loss={r['loss']}` - {r['status']} "
                     f"(see `{r['model']}_bs{r['batch_size']}_lr{r['lr']}"
                     f"{'' if r['loss'] == DEFAULT_LOSS else '_' + r['loss']}"
                     f"/stdout.log`)")
        L.append("")

    L.append("## Winner")
    L.append("")
    if win is None:
        L.append("No run completed with a validation metric, so there is no "
                 "winner to report.")
    else:
        L.append(f"**{win['model']}**, bs={win['batch_size']}, lr={win['lr']}, "
                 f"loss={win['loss']} - depth MAE "
                 f"{win['best_val_depth_mae']} m, best EPE "
                 f"{win['best_val_epe']} px.")
        L.append("")
        L.append(f"Train it properly: all the data (`--data-fraction 1.0`, vs "
                 f"the sweep's {args.data_fraction}), up to "
                 f"{args.final_epochs} epochs, stopping early after "
                 f"{args.final_patience} evaluations without improvement.")
        L.append("")
        L.append("```bash")
        L.append(_final_run_command(win, args, out_root))
        L.append("```")
        L.append("")
        L.append("A caveat worth keeping: the sweep ranked configs on "
                 f"{args.data_fraction} of the data over {args.grid_epochs} "
                 "epochs. That is enough to separate clearly different configs, "
                 "but nothing here establishes the run-to-run noise floor, so "
                 "treat near-ties as ties rather than as a ranking.")
    L.append("")
    if tb_url:
        L.append(f"Curves: {tb_url}")
        L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L))
    return path, win


def _fmt_hms(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _print_summary(rows, out_root, tb_url=None):
    # Ranked by depth MAE, not EPE. The log-space losses deliberately stop
    # spending gradient on the near field, where disparity error is cheap, so
    # they can score worse on EPE while being better at the depth accuracy
    # obstacle avoidance actually needs (see losses.py). Ranking a loss
    # comparison by EPE would systematically pick the wrong winner.
    def sort_key(r):
        try:
            return float(r.get("best_val_depth_mae", ""))
        except (TypeError, ValueError):
            return float("inf")  # FAILED / no-metric-yet rows sort last

    rows = sorted(rows, key=sort_key)
    print("\n=== summary, sorted by best_val_depth_mae (lower = better) ===")
    widths = [max(len(h), max((len(str(r.get(h, ""))) for r in rows), default=0))
             for h in _SUMMARY_FIELDS]
    def fmt(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    print(fmt(_SUMMARY_FIELDS))
    for r in rows:
        print(fmt([r.get(h, "") for h in _SUMMARY_FIELDS]))

    print("\n=== tensorboard ===")
    if tb_url:
        print(f"  {tb_url}")
    else:
        print(f"  tensorboard --logdir {out_root} --bind_all")
        ip = _lan_ip()
        if ip:
            print(f"  then from another PC on the LAN: http://{ip}:6006")


def run_grid_search(args):
    """Orchestrate the sweep: build the combo list, skip already-completed
    combos, run the rest as subprocesses, keep summary.csv updated as it
    goes, and always print a sorted summary on the way out (including on
    Ctrl-C) so an interrupted sweep still leaves a readable record.
    """
    for d in args.data:
        if not os.path.isdir(d):
            raise SystemExit(f"error: TartanAir data dir not found: {d}")

    out_root = args.out
    log_dir = args.log_dir if args.log_dir != "logs" else os.path.join(out_root, "logs")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    summary_path = os.path.join(out_root, "summary.csv")
    sweep_log = open(os.path.join(out_root, "sweep.log"), "a")

    def slog(msg=""):
        print(msg)
        sweep_log.write(msg + "\n")
        sweep_log.flush()

    passthrough = _passthrough_argv(args)
    combos, skipped = build_combos(args)
    total = len(combos)
    rows = []
    sweep_t0 = time.time()

    tb_proc, tb_url = (None, None)
    if not args.no_tensorboard:
        tb_proc, tb_url = _start_tensorboard(out_root, args.tensorboard_port)

    slog(f"=== sweep: {total} runs, epochs={args.grid_epochs}, data={args.data}, out={out_root} ===")
    slog(f"models={args.grid_models}  batch_sizes={args.grid_bs}  lrs={args.grid_lr}  "
         f"losses={args.grid_loss}  seed={args.seed}")
    for model, bs, lr, loss in skipped:
        est = _GB_PER_SAMPLE.get(model, 0) * bs * (
            _LOSS_VRAM_SURCHARGE if loss in _LOG_LOSSES else 1.0)
        slog(f"  skipped {model} bs={bs} lr={lr} loss={loss}: ~{est:.1f}GB needed "
             f"> {_VRAM_BUDGET_GB}GB budget (--no-grid-skip-infeasible to try anyway)")
    if tb_url:
        slog(f"=== tensorboard live at: {tb_url}  (open this in a browser) ===")

    try:
        for i, (model, bs, lr, loss) in enumerate(combos, 1):
            # The loss suffix is omitted for the default, so run directories
            # from sweeps that predate --loss keep matching and are still
            # skipped/resumed rather than silently retrained from scratch.
            run_name = f"{model}_bs{bs}_lr{lr}"
            if loss != DEFAULT_LOSS:
                run_name += f"_{loss}"
            run_out = os.path.join(out_root, run_name)
            log_csv_path = os.path.join(run_out, "log.csv")

            resume_from = None
            if not args.grid_force_rerun:
                done = _log_csv_epochs_done(log_csv_path)
                if done >= args.grid_epochs:
                    slog(f"\n=== [{i}/{total}] {run_name} -> already complete "
                         f"({done}/{args.grid_epochs} epochs), skipping ===")
                    best_epe, final_epe, best_mae = _log_csv_epes(log_csv_path)
                    row = {"model": model, "batch_size": str(bs), "lr": lr,
                           "loss": loss, "status": "SKIPPED",
                           "best_val_epe": best_epe, "final_val_epe": final_epe,
                           "best_val_depth_mae": best_mae, "seconds": "0"}
                    rows = _update_summary(summary_path, model, bs, lr, loss, row)
                    continue
                # Partially trained (interrupted sweep, machine reboot, a
                # transient CUDA failure): pick up from the checkpoint rather
                # than throwing away the epochs already paid for.
                last_ckpt = os.path.join(run_out, "last.pth")
                if done > 0 and os.path.exists(last_ckpt):
                    resume_from = last_ckpt
                    slog(f"\n=== [{i}/{total}] {run_name} -> resuming at epoch "
                         f"{done}/{args.grid_epochs} from last.pth ===")

            try:
                os.makedirs(run_out, exist_ok=True)
            except OSError as e:
                slog(f"!!! could not create {run_out}: {e}, skipping")
                row = {"model": model, "batch_size": str(bs), "lr": lr,
                       "loss": loss, "status": "FAILED", "best_val_epe": "",
                       "final_val_epe": "", "best_val_depth_mae": "",
                       "seconds": "0"}
                rows = _update_summary(summary_path, model, bs, lr, loss, row)
                continue

            if resume_from is None:
                slog(f"\n=== [{i}/{total}] model={model} bs={bs} lr={lr} "
                     f"loss={loss} -> {run_out} ===")
            t0 = time.time()

            argv = [sys.executable, os.path.abspath(__file__), *passthrough,
                    "--epochs", str(args.grid_epochs),
                    "--model", model, "--bs", str(bs), "--lr", lr,
                    "--loss", loss,
                    "--out", run_out, "--log-dir", log_dir]
            if resume_from:
                argv += ["--resume", resume_from]
            with open(os.path.join(run_out, "stdout.log"),
                      "a" if resume_from else "w") as run_log:
                proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    print(line, end="")
                    run_log.write(line)
                    sweep_log.write(line)
                status = proc.wait()
            sweep_log.flush()
            dt = time.time() - t0

            if status != 0:
                oom = _run_hit_oom(os.path.join(run_out, "stdout.log"))
                slog(f"!!! run failed (exit {status}"
                     f"{', CUDA OOM' if oom else ''}), see {run_out}/stdout.log "
                     f"and {log_dir}/{run_name}.txt - continuing sweep")
                # Clean up before the next config. The run was its own process,
                # so the kernel has already reclaimed its VRAM - but a crash can
                # leave orphaned dataloader workers holding memory, and a
                # half-written checkpoint would otherwise be picked up by
                # auto-resume and fail again the same way.
                _cleanup_failed_run(run_out, drop_checkpoints=oom, slog=slog)
                row = {"model": model, "batch_size": str(bs), "lr": lr,
                       "loss": loss, "status": "OOM" if oom else "FAILED",
                       "best_val_epe": "", "final_val_epe": "",
                       "best_val_depth_mae": "", "seconds": f"{dt:.0f}"}
                rows = _update_summary(summary_path, model, bs, lr, loss, row)
                continue

            best_epe, final_epe, best_mae = _log_csv_epes(log_csv_path)
            row = {"model": model, "batch_size": str(bs), "lr": lr, "loss": loss,
                   "status": "OK", "best_val_epe": best_epe,
                   "final_val_epe": final_epe, "best_val_depth_mae": best_mae,
                   "seconds": f"{dt:.0f}"}
            rows = _update_summary(summary_path, model, bs, lr, loss, row)
    finally:
        if os.path.exists(summary_path):
            with open(summary_path, newline="") as f:
                rows = list(csv.DictReader(f))
        _print_summary(rows, out_root, tb_url)
        # Written even on Ctrl-C, so an interrupted sweep still leaves a
        # readable record of everything that did finish.
        try:
            report_path, win = write_report(rows, skipped, args, out_root,
                                            time.time() - sweep_t0, tb_url)
            print(f"\n=== report written: {report_path} ===")
            if win is not None:
                print(f"\nWinner: {win['model']} bs={win['batch_size']} "
                      f"lr={win['lr']} loss={win['loss']}  "
                      f"(depth MAE {win['best_val_depth_mae']} m, "
                      f"EPE {win['best_val_epe']} px)")
                print("\nTrain it for real - all the data, long schedule, early "
                      "stopping:\n")
                print(_final_run_command(win, args, out_root))
                print()
        except Exception as e:  # a reporting bug must never sink a finished sweep
            print(f"note: could not write report ({type(e).__name__}: {e})")
        sweep_log.close()
        if tb_proc is not None:
            # Keep the dashboard up briefly so the final numbers can be read,
            # then don't leave an orphaned server bound to the port.
            print(f"\ntensorboard still serving {tb_url} - press Ctrl-C to stop it")
            try:
                tb_proc.wait()
            except KeyboardInterrupt:
                pass
            finally:
                tb_proc.terminate()
                try:
                    tb_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    tb_proc.kill()


def main():
    ap = argparse.ArgumentParser(
        description="Train the stereo depth models on TartanAir (V1 or V2).")
    ap.add_argument("--data", nargs="+", required=True,
                    help="TartanAir root(s); trajectories are found recursively")
    ap.add_argument("--model", default="baseline", choices=MODEL_CHOICES)
    ap.add_argument("--out", default="runs/tartanair")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--window", type=int, default=None,
                    help="temporal window (default: 1 baseline, 4 temporal)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="gap between window frames (TartanAir already moves a "
                         "long way between its 10 Hz frames, hence 1)")
    ap.add_argument("--crop", default="480x640",
                    help="HxW random crop (multiple of 16, or of 32 for --model "
                         "yolo, whose backbone reaches 1/32). The default "
                         "480x640 crops TartanAir's 640x640 frames vertically "
                         "only: it matches the deployment camera's 4:3 aspect "
                         "ratio, fits V1 and V2, and satisfies both multiple-of "
                         "constraints. Full width is kept on purpose - "
                         "disparity is a horizontal offset, so narrowing the "
                         "image would discard the very context matching needs "
                         "and truncate the search range at the left edge. Pass "
                         "'' to train on whole uncropped frames.")
    ap.add_argument("--max-disp", type=int, default=128)
    add_loss_args(ap)
    ap.add_argument("--yolo-scale", default="n", choices=["n", "s", "m", "l", "x"],
                    help="YOLO26 compound scale for --model yolo. 'n' is 2.6M "
                         "params against baseline's 0.3M, so a comparison "
                         "table should say so; 's' is 9.4M.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize factor; scales fx and disparity linearly "
                         "(target_fxb / 80 matches another rig's disparity range)")
    ap.add_argument("--camera", default="front",
                    choices=["front", "back", "left", "right", "top", "bottom"],
                    help="V2 stereo pair to train on (only 'front' is verified "
                         "against the NED pose convention)")
    ap.add_argument("--difficulty", default=None, choices=["easy", "hard"],
                    help="restrict to Data_easy / Data_hard (default: both)")
    ap.add_argument("--envs", nargs="+", default=None,
                    help="restrict to these environment names")
    ap.add_argument("--val-envs", nargs="+", default=None,
                    help="hold out these environments (default: last --val-frac)")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of *environments* held out for validation")
    ap.add_argument("--pose-noise", type=float, default=0.0,
                    help="train-time rotation noise on rel poses, deg (temporal)")
    ap.add_argument("--pose-noise-trans", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader workers. 8 measured: 1 worker caps the "
                         "whole pipeline at ~18 frames/s and starves the GPU, "
                         "8 reaches ~70 win/s which is ~88%% of the GPU-only "
                         "ceiling, so there is little left to gain above it.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="continue a run: restores weights, optimiser and epoch")
    ap.add_argument("--init", default=None,
                    help="warm-start weights only, e.g. from an AirSim run")
    ap.add_argument("--log-dir", default="logs",
                    help="folder collecting one <run-name>.txt per experiment")
    ap.add_argument("--print-every", type=int, default=10,
                    help="update the same console line every N batches with "
                         "progress and an ETA")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: give up after N consecutive "
                         "evaluations with no improvement in val EPE (0 = "
                         "never stop early). Counted in evaluations, not "
                         "epochs, so it means the same thing at any "
                         "--eval-every.")
    ap.add_argument("--eval-every", type=int, default=4,
                    help="run validation every N epochs. The final epoch is "
                         "always evaluated regardless, so a run never ends "
                         "without a result. Note --patience counts "
                         "evaluations, not epochs, so raising this stretches "
                         "how long early stopping waits in wall-clock terms.")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds python/numpy/torch/cuda and the DataLoader at "
                         "the start of every run, so a config is reproducible "
                         "and every grid combo starts from identical RNG state")
    ap.add_argument("--deterministic", action="store_true",
                    help="refuse nondeterministic CUDA kernels so identically "
                         "seeded runs are bit-exact. Off by default because "
                         "some ops these models use (bilinear interpolate "
                         "backward, the cost-volume scatter) have no "
                         "deterministic CUDA kernel and will raise instead.")
    ap.add_argument("--data-fraction", type=float, default=0.3,
                    help="train and validate on this deterministic fraction "
                         "(0-1] of each split. Defaults to 0.3 to keep sweeps "
                         "tractable; pass --data-fraction 1.0 for the "
                         "full-length run on the config a sweep picks, or an "
                         "explicit --max-train-windows/--max-val-windows to "
                         "give either half an absolute count instead.")
    ap.add_argument("--max-train-windows", type=int, default=None,
                    help="cap the training set to N windows, randomly "
                         "subsampled with a fixed seed so every run in a grid "
                         "search trains on the same subset. Disabled by "
                         "default: training on everything is the right thing "
                         "for a real run, and capping is a sweep-only "
                         "shortcut you should have to ask for. Ignored if "
                         "--data-fraction is given.")
    ap.add_argument("--max-val-windows", type=int, default=None,
                    help="cap validation to N windows, same fixed-seed "
                         "subsampling as --max-train-windows. Validation runs "
                         "every --eval-every epochs, so on a large dataset it "
                         "can dominate sweep wall-clock time far more than "
                         "training does - this bounds it.")
    ap.add_argument("--grid-search", action="store_true",
                    help="sweep --grid-models x --grid-bs x --grid-lr instead "
                         "of a single run; --model/--bs/--lr are ignored, --out "
                         "becomes the sweep's output root. Each combo runs as "
                         "its own subprocess (isolates a CUDA OOM to one run) "
                         "and is skipped on a re-run if already complete.")
    ap.add_argument("--grid-models", nargs="+", default=MODEL_CHOICES,
                    choices=MODEL_CHOICES,
                    help="default is all eight models, yolo included. yolo "
                         "needs ultralytics installed (imported lazily, only "
                         "for this model); pass --grid-models without it to "
                         "exclude it, e.g. on a machine without that "
                         "dependency")
    ap.add_argument("--grid-bs", nargs="+", type=int, default=[8, 16],
                    help="ascending, so the cheapest (and least OOM-prone) "
                         "combos report first. Measured peak VRAM at 480x640 "
                         "with --amp: baseline/anynet ~2.7/5.5/11GB, mobilenet "
                         "~4.8/9.5/19GB, temporal ~11/22.5GB and OOM at 32 "
                         "(its window=4 backprop is ~4x the activations); yolo "
                         "3.1GB at bs=8 measured, so ~6/12GB at 16/32 on the "
                         "linear scaling the other rows show.")
    ap.add_argument("--grid-lr", nargs="+", default=["1e-3", "3e-3"],
                    help="log-spaced ~3x apart, biased high: on capped data a "
                         "short sweep is step-starved, so under-fitting is the "
                         "real risk rather than divergence (AdamW + grad-clip "
                         "1.0 + cosine keeps even 3e-3 tractable). 3e-4 was "
                         "dropped after losing to 1e-3 in every trial run. "
                         "1e-2 was added after the first full sweep "
                         "(runs/tartanair_grid) came back with 3e-3 winning "
                         "outright - top rank overall (temporal, bs=8, "
                         "logl1) and the best score for several other "
                         "model/loss combos - with 3e-3 sitting at the top "
                         "of the range tested, and a winner on the boundary "
                         "means the optimum was never bracketed. It was then "
                         "dropped again once actually measured: across every "
                         "combo tried, 1e-2 had the worst average EPE and "
                         "depth MAE of the three rates and placed in zero of "
                         "the top 10 results by depth MAE - the extra ~3x "
                         "step turned out to be past the optimum, not "
                         "short of it. Kept as literal strings (not parsed "
                         "to float) so run directory names match exactly "
                         "what you typed")
    ap.add_argument("--grid-loss", nargs="+",
                    default=GRID_LOSS_DEFAULT,
                    choices=LOSS_CHOICES,
                    help="fourth sweep axis: smoothl1, logl1, hybrid by "
                         "default. smoothl1 is the control (every earlier "
                         "run is comparable to it), logl1 the log-space "
                         "objective aimed at the depth metrics, hybrid the "
                         "combination of the two - so if smoothl1 wins the "
                         "near field and logl1 the far field, hybrid is "
                         "where that shows up. depth26 (a faithful port of "
                         "the loss YOLO26-depth trains with: SILog + "
                         "multi-scale gradient matching) is excluded by "
                         "default: every combo tried gave far worse EPE and "
                         "depth MAE than the other three losses, confirming "
                         "its scale-invariant SILog term is the wrong choice "
                         "here, where stereo's fxb already gives true metric "
                         "scale (losses.py). Pass `--grid-loss ... depth26` "
                         "explicitly to include it anyway. The same losses "
                         "are applied to every model, so a model comparison "
                         "is not confounded by which objective each one "
                         "happened to get. Runs using a non-default loss get "
                         "a '_<loss>' directory suffix. train_loss is not "
                         "comparable across losses, but val EPE / D1 / depth "
                         "MAE are.")
    ap.add_argument("--grid-epochs", type=int, default=20,
                    help="epochs per combo in a sweep, kept separate from "
                         "--epochs (which stays long for the real run you do "
                         "once the sweep has picked a winner)")
    ap.add_argument("--final-epochs", type=int, default=100,
                    help="epochs in the follow-up command the report prints "
                         "for training the winning config on all the data")
    ap.add_argument("--final-patience", type=int, default=8,
                    help="early-stopping patience, in evaluations, for that "
                         "follow-up command")
    ap.add_argument("--grid-force-rerun", action="store_true",
                    help="redo combos even if a previous sweep already "
                         "completed them (default: finished combos are "
                         "skipped, partly-trained ones resume from last.pth)")
    ap.add_argument("--grid-skip-infeasible", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="drop combos predicted to exceed VRAM before running "
                         "them (default on). --no-grid-skip-infeasible runs "
                         "them anyway, e.g. to re-measure after a model change.")
    ap.add_argument("--no-tensorboard", action="store_true",
                    help="don't launch a tensorboard server for the sweep")
    ap.add_argument("--tensorboard-port", type=int, default=6006)
    args = ap.parse_args()
    args.dataset = "tartanair"  # recorded in the checkpoint, as train.py does

    if args.grid_search:
        run_grid_search(args)
        return

    if args.window is None:
        args.window = {"temporal": 4, "stereoconv3d": 10,
                       "stereoconv3d_fast": 3}.get(args.model, 1)
    crop = tuple(int(x) for x in args.crop.split("x")) if args.crop else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Seed before anything random happens: dataset construction, the subsample
    # permutations, weight init and the loader's shuffle all draw from these.
    gen = set_seed(args.seed, deterministic=args.deterministic)
    print(f"seed={args.seed}")

    try:
        dataset = TartanAirDataset(args.data, camera=args.camera,
                                   difficulty=args.difficulty, envs=args.envs,
                                   scale=args.scale, window=args.window,
                                   frame_stride=args.frame_stride, crop=crop,
                                   augment=True, max_disp=args.max_disp)
        if crop is not None:
            w, h = dataset.sequences[0]["size"]
            if crop[0] > h or crop[1] > w:
                raise SystemExit(f"--crop {crop[0]}x{crop[1]} does not fit the "
                                 f"{w}x{h} images (after --scale {args.scale}).")

        train_set, val_set, held_out = environment_split(dataset, args.val_frac,
                                                         args.val_envs)
        envs = sorted({s["env"] for s in dataset.sequences})
        print(f"{len(envs)} environments, {len(dataset.sequences)} trajectories -> "
              f"{len(train_set)} train / {len(val_set)} val windows "
              f"(window={args.window})")
        # --data-fraction applies to train *and* val: validation runs every
        # --eval-every epochs and on a large set it dominates wall-clock, so
        # shrinking only the training half would leave most of the cost. An
        # explicitly given absolute count wins over the fraction for that half
        # - otherwise, with --data-fraction carrying a default, the count flags
        # could never take effect at all.
        n_train, n_val = args.max_train_windows, args.max_val_windows
        if args.data_fraction is not None:
            if not 0 < args.data_fraction <= 1:
                raise SystemExit(f"--data-fraction must be in (0, 1], got "
                                 f"{args.data_fraction}")
            print(f"  --data-fraction {args.data_fraction}"
                  + ("  (overridden for train by --max-train-windows)"
                     if n_train is not None else "")
                  + ("  (overridden for val by --max-val-windows)"
                     if n_val is not None else ""))
            if n_train is None:
                n_train = max(1, int(round(len(train_set) * args.data_fraction)))
            if n_val is None:
                n_val = max(1, int(round(len(val_set) * args.data_fraction)))
        # Distinct seeds for the two so the subsets are drawn independently.
        train_set = subsample(train_set, n_train, args.seed, "training set")
        val_set = subsample(val_set, n_val, args.seed + 1, "validation set")
        print("validation: " + (f"held-out environments {held_out}" if held_out else
                                "by sequence (only one environment available)"))
        describe_calibration(dataset, args.max_disp)

        train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True,
                                  num_workers=args.workers, drop_last=True, pin_memory=True,
                                  generator=gen, worker_init_fn=seed_worker,
                                  persistent_workers=args.workers > 0)
        val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True,
                                worker_init_fn=seed_worker,
                                persistent_workers=args.workers > 0)

        model = build_model(args.model, max_disp=args.max_disp,
                            yolo_scale=args.yolo_scale).to(device)
        print(f"model={args.model}  params={count_parameters(model) / 1e6:.2f}M  "
              f"device={device}  loss={args.loss}")
        if args.init:
            if args.resume:
                raise SystemExit("--init and --resume are mutually exclusive")
            ck = torch.load(args.init, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            print(f"warm-started from {args.init} (epoch {ck.get('epoch', '?')})")

        fit(model, train_loader, val_loader, args, device)
    except Exception:
        log_exception(args)
        raise


if __name__ == "__main__":
    main()
