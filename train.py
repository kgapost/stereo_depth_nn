"""Train the baseline (FastStereoNet) or temporal (TempoBandNet) model.

Examples:
  # baseline, single frames
  python train.py --model baseline --data ~/datasets/airsim_stereo --out runs/baseline

  # temporal, windows of 4 frames with ego-motion
  python train.py --model temporal --window 4 --bs 4 --data ~/datasets/airsim_stereo \
      --out runs/temporal --pose-noise 0.3

  # log-space objective aimed at the depth metrics rather than disparity EPE
  python train.py --model yolo --loss logl1 --data ~/datasets/airsim_stereo \
      --out runs/yolo_logl1

Validation is held out by *sequence* (never by frame) to avoid leakage.

The training objective is selected with --loss; see losses.py for what each
option optimises and why. Metrics are unaffected by the choice, so runs using
different losses stay directly comparable on val EPE / D1 / depth MAE.
"""

import argparse
import csv
import math
import os
import time
import traceback
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

from stereo_datasets import build_dataset
from losses import add_loss_args, build_loss
from models import build_model, count_parameters


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _run_log_path(args):
    """Path to this run's plaintext log: one <run-name>.txt per experiment,
    collected in args.log_dir so a whole grid search is browsable in one
    folder instead of digging through each run's own --out directory."""
    log_dir = getattr(args, "log_dir", None) or "logs"
    os.makedirs(log_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.out))
    return os.path.join(log_dir, f"{run_name}.txt")


def log_exception(args):
    """Best-effort: append a timestamped traceback to this run's text log.

    Called from the outer try/except in main() so crashes that happen before
    fit() ever opens its log file (bad data path, OOM building the model,
    ...) still land somewhere instead of only flashing past on the console.
    """
    try:
        with open(_run_log_path(args), "a") as f:
            f.write(f"[{_now()}] FAILED\n")
            f.write(traceback.format_exc() + "\n")
    except Exception:
        pass  # logging the failure must never mask the original exception


def sequence_split(dataset, val_frac):
    """Split window indices by sequence id: last sequences go to validation."""
    n_seq = len(dataset.sequences)
    n_val = max(1, math.ceil(val_frac * n_seq)) if n_seq > 1 else 0
    val_sids = set(range(n_seq - n_val, n_seq))
    tr, va = [], []
    for pos, (sid, _) in enumerate(dataset.index):
        (va if sid in val_sids else tr).append(pos)
    return Subset(dataset, tr), Subset(dataset, va)


def perturb_poses(rel_pose, rot_std_deg, trans_std_frac):
    """Simulate VIO noise: small random rotation + proportional translation noise."""
    B = rel_pose.shape[0]
    dev = rel_pose.device
    w = torch.randn(B, 3, device=dev) * math.radians(rot_std_deg)
    theta = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k = (w / theta).view(B, 3)
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = torch.eye(3, device=dev).expand(B, 3, 3)
    s, c = torch.sin(theta).view(B, 1, 1), torch.cos(theta).view(B, 1, 1)
    dR = I + s * K + (1 - c) * (K @ K)
    out = rel_pose.clone()
    out[:, :3, :3] = dR @ rel_pose[:, :3, :3]
    t = rel_pose[:, :3, 3]
    out[:, :3, 3] = t + torch.randn_like(t) * (t.norm(dim=1, keepdim=True) * trans_std_frac + 1e-3)
    return out


def criterion_from_args(args):
    """Build the --loss criterion, defaulting to the original smooth-L1 so a
    caller that predates the flag (or an older args namespace restored from a
    checkpoint) keeps the previous behaviour exactly."""
    return build_loss(getattr(args, "loss", "smoothl1"),
                      w_log=getattr(args, "loss_w_log", 1.0),
                      w_grad=getattr(args, "loss_w_grad", 0.5),
                      silog_lambda=getattr(args, "loss_silog_lambda", 1.0))


@torch.no_grad()
def metrics(pred, gt, valid, fxb, max_eval_depth=30.0):
    """Returns (epe_sum, d1_sum, mae_depth_sum, n_valid, n_depth) for aggregation."""
    vb = valid > 0.5
    err = (pred - gt).abs()
    epe = err[vb].sum()
    d1 = ((err > 3.0) & (err > 0.05 * gt) & vb).sum().float()
    depth_gt = fxb.view(-1, 1, 1, 1) / gt.clamp(min=1e-3)
    depth_pred = fxb.view(-1, 1, 1, 1) / pred.clamp(min=1e-3)
    db = vb & (depth_gt < max_eval_depth)
    mae = (depth_pred - depth_gt).abs()[db].sum()
    return epe.item(), d1.item(), mae.item(), vb.sum().item(), db.sum().item()


def run_window(model, args, batch, device, train=True, criterion=None):
    """Run one (B,T,...) window; returns (loss, metric tuple of final frame).

    `criterion` is the --loss function from losses.build_loss; built on demand
    when a caller does not supply one.
    """
    if criterion is None:
        criterion = criterion_from_args(args)
    for k in batch:
        batch[k] = batch[k].to(device, non_blocking=True)
    T = batch["left"].shape[1]

    if args.model in ("stereoconv3d", "stereoconv3d_fast"):
        # One forward call over the whole stacked window, not a per-frame
        # loop: the model consumes all T frames at once and predicts a
        # single disparity map for the last (current) one.
        out = model(batch["left"], batch["right"], batch["fxb"])
        gt, valid = batch["disp"][:, -1], batch["valid"][:, -1]
        loss = criterion(out, gt, valid)
        m = metrics(out["disp"], gt, valid, batch["fxb"])
        return loss, m

    state, loss, m = None, 0.0, None
    for t in range(T):
        left, right = batch["left"][:, t], batch["right"][:, t]
        if args.model == "temporal":
            rel = batch["rel_pose"][:, t]
            if train and args.pose_noise > 0 and t > 0:
                rel = perturb_poses(rel, args.pose_noise, args.pose_noise_trans)
            out = model(left, right, batch["K"], batch["fxb"],
                        state=state, rel_pose=rel)
            state = out["state"]
        elif args.model == "stereoconv":
            out = model(left, right, batch["fxb"])
        else:
            out = model(left, right)
        loss = loss + criterion(out, batch["disp"][:, t], batch["valid"][:, t])
        m = metrics(out["disp"], batch["disp"][:, t], batch["valid"][:, t], batch["fxb"])
    return loss / T, m


def fit(model, train_loader, val_loader, args, device):
    """Optimise `model`, logging to args.out/log.csv and saving last/best.pth.

    Split out of main() so train_tartanair.py trains under exactly the same
    schedule, loss and checkpoint format - only the data differs.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    criterion = criterion_from_args(args)

    start_epoch, best_epe = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_epe = ck["epoch"] + 1, ck.get("best_epe", best_epe)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    log_path = os.path.join(args.out, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_epe", "val_d1",
                                    "val_depth_mae_30m", "lr", "seconds"])

    run_name = os.path.basename(os.path.normpath(args.out))
    txt = open(_run_log_path(args), "a")

    def tlog(msg):
        print(msg)
        txt.write(msg + "\n")
        txt.flush()

    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=args.out)
    except Exception as e:
        tlog(f"[{_now()}] tensorboard unavailable ({e}); continuing without it")

    tlog("=" * 88)
    tlog(f"[{_now()}] {'resuming' if args.resume else 'starting'} run '{run_name}'  "
         f"model={args.model} bs={args.bs} lr={args.lr} epochs={args.epochs} "
         f"window={args.window} loss={getattr(args, 'loss', 'smoothl1')}")
    tlog(f"  data={args.data}  out={args.out}")

    print_every = max(1, getattr(args, "print_every", 10))
    eval_every = max(1, getattr(args, "eval_every", 1))
    patience = getattr(args, "patience", 0) or 0
    evals_since_best = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0, running = time.time(), 0.0
            n_steps = len(train_loader)
            for step, batch in enumerate(train_loader):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.split(":")[0], enabled=args.amp):
                    loss, m = run_window(model, args, batch, device, train=True,
                                         criterion=criterion)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                running += loss.item()
                if (step + 1) % print_every == 0 or step + 1 == n_steps:
                    epe = m[0] / max(m[3], 1)
                    elapsed = time.time() - t0
                    avg = elapsed / (step + 1)
                    remaining = (n_steps - step - 1) + (args.epochs - epoch - 1) * n_steps
                    eta = avg * remaining
                    pct = 100.0 * (step + 1) / n_steps
                    print(f"\r  e{epoch} batch {step + 1}/{n_steps} ({pct:5.1f}%) "
                          f"loss={loss.item():.3f} epe(last)={epe:.2f}px "
                          f"| elapsed {_fmt_duration(elapsed)} eta {_fmt_duration(eta)}   ",
                          end="", flush=True)
            print()  # end the overwritten progress line before the epoch summary
            train_loss = running / max(n_steps, 1)

            do_eval = (epoch + 1) % eval_every == 0 or epoch == args.epochs - 1
            val_epe = val_d1 = val_mae = None
            is_best = False
            if do_eval:
                model.eval()
                sums = [0.0] * 5
                with torch.no_grad():
                    for batch in val_loader:
                        _, m = run_window(model, args, batch, device, train=False,
                                          criterion=criterion)
                        sums = [a + b for a, b in zip(sums, m)]
                val_epe = sums[0] / max(sums[3], 1)
                val_d1 = 100.0 * sums[1] / max(sums[3], 1)
                val_mae = sums[2] / max(sums[4], 1)
                is_best = val_epe < best_epe
            sched.step()
            dt = time.time() - t0

            if do_eval:
                tlog(f"[{_now()}] epoch {epoch}: loss={train_loss:.3f}  "
                     f"val EPE={val_epe:.2f}px D1={val_d1:.2f}% "
                     f"depthMAE<30m={val_mae:.2f}m  ({dt:.0f}s)"
                     + ("  *new best*" if is_best else ""))
            else:
                tlog(f"[{_now()}] epoch {epoch}: loss={train_loss:.3f}  "
                     f"(no eval this epoch, next at epoch "
                     f"{min(epoch + eval_every - (epoch + 1) % eval_every, args.epochs - 1)}"
                     f")  ({dt:.0f}s)")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, f"{train_loss:.4f}",
                    f"{val_epe:.4f}" if do_eval else "",
                    f"{val_d1:.4f}" if do_eval else "",
                    f"{val_mae:.4f}" if do_eval else "",
                    f"{sched.get_last_lr()[0]:.2e}", f"{dt:.0f}"])
            if tb is not None:
                tb.add_scalar("train/loss", train_loss, epoch)
                if do_eval:
                    tb.add_scalar("val/epe_px", val_epe, epoch)
                    tb.add_scalar("val/d1_pct", val_d1, epoch)
                    tb.add_scalar("val/depth_mae_30m", val_mae, epoch)
                tb.add_scalar("lr", sched.get_last_lr()[0], epoch)
                tb.add_scalar("epoch_seconds", dt, epoch)
                tb.flush()

            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "epoch": epoch,
                  "best_epe": min(best_epe, val_epe) if do_eval else best_epe,
                  "args": vars(args)}
            torch.save(ck, os.path.join(args.out, "last.pth"))
            if is_best:
                prev_best = best_epe
                best_epe = val_epe
                torch.save(ck, os.path.join(args.out, "best.pth"))
                tlog(f"  >>> NEW BEST MODEL: val EPE {val_epe:.2f}px "
                     f"(previous best {prev_best:.2f}px) - saved to best.pth <<<"
                     if prev_best != float("inf") else
                     f"  >>> NEW BEST MODEL: val EPE {val_epe:.2f}px - saved to best.pth <<<")

            # Early stopping. Counted in *evaluations* rather than epochs, so
            # --patience means the same thing regardless of --eval-every.
            if do_eval and patience:
                evals_since_best = 0 if is_best else evals_since_best + 1
                if evals_since_best >= patience:
                    tlog(f"[{_now()}] early stop: {evals_since_best} evaluations "
                         f"({evals_since_best * eval_every} epochs) with no "
                         f"improvement on val EPE {best_epe:.2f}px "
                         f"(--patience {patience})")
                    break
        tlog(f"[{_now()}] finished run '{run_name}': best val EPE {best_epe:.2f}px")
    finally:
        if tb is not None:
            tb.close()
        txt.close()
    return best_epe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--dataset", default="airsim", choices=["airsim", "tartanair"])
    ap.add_argument("--model", default="baseline",
                    choices=["baseline", "stereoconv", "stereoconv3d",
                            "stereoconv3d_fast", "temporal", "mobilenet",
                            "anynet", "yolo"])
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--window", type=int, default=None,
                    help="temporal window (default: 1 baseline, 4 temporal)")
    ap.add_argument("--frame-stride", type=int, default=2,
                    help="gap between window frames (2 @10FPS => 0.2s of motion)")
    ap.add_argument("--crop", default=None,
                    help="HxW random crop (multiple of 16; 32 for --model yolo)")
    ap.add_argument("--max-disp", type=int, default=128)
    add_loss_args(ap)
    ap.add_argument("--yolo-scale", default="n", choices=["n", "s", "m", "l", "x"],
                    help="YOLO26 compound scale for --model yolo")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--pose-noise", type=float, default=0.0,
                    help="train-time rotation noise on rel poses, deg (temporal)")
    ap.add_argument("--pose-noise-trans", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="continue a run: restores weights, optimiser and epoch")
    ap.add_argument("--init", default=None,
                    help="warm-start weights only, e.g. from a TartanAir run")
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
    args = ap.parse_args()

    if args.window is None:
        args.window = {"temporal": 4, "stereoconv3d": 10,
                       "stereoconv3d_fast": 3}.get(args.model, 1)
    crop = tuple(int(x) for x in args.crop.split("x")) if args.crop else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    try:
        dataset = build_dataset(args.dataset, args.data, window=args.window,
                                frame_stride=args.frame_stride, crop=crop,
                                augment=True, max_disp=args.max_disp)
        train_set, val_set = sequence_split(dataset, args.val_frac)
        print(f"{len(dataset.sequences)} sequences -> {len(train_set)} train / "
              f"{len(val_set)} val windows (window={args.window})")

        train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True,
                                  num_workers=args.workers, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

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
