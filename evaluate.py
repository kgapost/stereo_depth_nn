"""Evaluate a checkpoint (accuracy per depth range) and benchmark latency.

  python evaluate.py --ckpt runs/temporal/best.pth --model temporal \
      --data ~/datasets/airsim_stereo_test --dataset airsim
  python evaluate.py --model baseline --bench          # latency only, no data
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from stereo_datasets import build_dataset
from models import build_model, count_parameters

DEPTH_RANGES = [(0.0, 10.0), (10.0, 30.0), (30.0, 95.0)]


@torch.no_grad()
def evaluate(model, loader, args, device):
    is_temporal = args.model == "temporal"
    is_stereoconv3d = args.model in ("stereoconv3d", "stereoconv3d_fast")
    sums = {"epe": 0.0, "d1": 0.0, "n": 0}
    range_sums = {r: [0.0, 0] for r in DEPTH_RANGES}  # abs-rel depth error

    def accumulate(pred, gt, valid, fxb):
        vb = valid > 0.5
        err = (pred - gt).abs()
        sums["epe"] += err[vb].sum().item()
        sums["d1"] += ((err > 3.0) & (err > 0.05 * gt) & vb).sum().item()
        sums["n"] += vb.sum().item()

        fxb = fxb.view(-1, 1, 1, 1)
        depth_gt = fxb / gt.clamp(min=1e-3)
        depth_pred = fxb / pred.clamp(min=1e-3)
        for r in DEPTH_RANGES:
            rb = vb & (depth_gt >= r[0]) & (depth_gt < r[1])
            if rb.any():
                rel = ((depth_pred - depth_gt).abs() / depth_gt)[rb]
                range_sums[r][0] += rel.sum().item()
                range_sums[r][1] += rb.sum().item()

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        if is_stereoconv3d:
            # One forward call over the whole stacked window, scored only
            # against the last (current) frame - same as training.
            out = model(batch["left"], batch["right"], batch["fxb"])
            accumulate(out["disp"], batch["disp"][:, -1], batch["valid"][:, -1],
                      batch["fxb"])
            continue

        T = batch["left"].shape[1]
        state = None
        for t in range(T):
            if is_temporal:
                out = model(batch["left"][:, t], batch["right"][:, t],
                            batch["K"], batch["fxb"], state=state,
                            rel_pose=batch["rel_pose"][:, t])
                state = out["state"]
            elif args.model == "stereoconv":
                out = model(batch["left"][:, t], batch["right"][:, t], batch["fxb"])
            else:
                out = model(batch["left"][:, t], batch["right"][:, t])
            accumulate(out["disp"], batch["disp"][:, t], batch["valid"][:, t],
                      batch["fxb"])

    print(f"EPE      : {sums['epe'] / max(sums['n'], 1):.3f} px")
    print(f"D1 (>3px): {100.0 * sums['d1'] / max(sums['n'], 1):.2f} %")
    for r, (s, n) in range_sums.items():
        if n:
            print(f"abs-rel depth {r[0]:>4.0f}-{r[1]:<4.0f} m: {s / n:.4f}")


@torch.no_grad()
def benchmark(model, args, device):
    h, w = (int(x) for x in args.size.split("x"))
    is_stereoconv3d = args.model in ("stereoconv3d", "stereoconv3d_fast")
    if is_stereoconv3d:
        left = torch.randn(1, args.window, 3, h, w, device=device)
        right = torch.randn(1, args.window, 3, h, w, device=device)
    else:
        left = torch.randn(1, 3, h, w, device=device)
        right = torch.randn(1, 3, h, w, device=device)
    K = torch.tensor([[w / 2.0, w / 2.0, w / 2.0, h / 2.0]], device=device)
    fxb = torch.tensor([w / 2.0 * 0.4], device=device)
    if args.fp16 and device != "cpu":
        model = model.half()
        left, right, K, fxb = left.half(), right.half(), K, fxb

    def run(state):
        if args.model == "temporal":
            rel = torch.eye(4, device=device).unsqueeze(0)
            return model(left, right, K, fxb, state=state, rel_pose=rel)
        if args.model == "stereoconv" or is_stereoconv3d:
            return model(left, right, fxb)
        return model(left, right)

    state = run(None).get("state") if args.model == "temporal" else None
    for _ in range(10):  # warmup (steady-state for temporal)
        out = run(state)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        out = run(state)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = torch.tensor(times)
    print(f"latency {h}x{w} ({device}{', fp16' if args.fp16 else ''}): "
          f"{times.mean():.1f} ± {times.std():.1f} ms "
          f"({1000.0 / times.mean():.1f} FPS)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default="baseline",
                    choices=["baseline", "stereoconv", "stereoconv3d",
                            "stereoconv3d_fast", "temporal", "mobilenet",
                            "anynet", "yolo"])
    ap.add_argument("--yolo-scale", default="n", choices=["n", "s", "m", "l", "x"],
                    help="YOLO26 compound scale for --model yolo; must match "
                         "the scale --ckpt was trained with")
    ap.add_argument("--data", nargs="+", default=None)
    ap.add_argument("--dataset", default="airsim", choices=["airsim", "tartanair"])
    ap.add_argument("--window", type=int, default=None,
                    help="window length for temporal/stereoconv3d evaluation "
                         "(default: 8 temporal, 10 stereoconv3d, 3 "
                         "stereoconv3d_fast, 1 otherwise)")
    ap.add_argument("--max-disp", type=int, default=128)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--bench", action="store_true", help="run latency benchmark")
    ap.add_argument("--size", default="256x448", help="HxW for --bench")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    if args.window is None:
        args.window = {"temporal": 8, "stereoconv3d": 10,
                       "stereoconv3d_fast": 3}.get(args.model, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, max_disp=args.max_disp,
                        yolo_scale=args.yolo_scale).to(device).eval()
    print(f"model={args.model}  params={count_parameters(model) / 1e6:.2f}M")
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.ckpt} (epoch {ck.get('epoch', '?')})")

    if args.bench:
        benchmark(model, args, device)
    if args.data:
        window = args.window if args.model in (
            "temporal", "stereoconv3d", "stereoconv3d_fast") else 1
        ds = build_dataset(args.dataset, args.data, window=window,
                           frame_stride=1, window_stride=window,
                           augment=False, max_disp=args.max_disp)
        loader = DataLoader(ds, batch_size=args.bs, num_workers=4)
        print(f"evaluating on {len(ds)} windows from {len(ds.sequences)} sequences")
        evaluate(model, loader, args, device)


if __name__ == "__main__":
    main()
