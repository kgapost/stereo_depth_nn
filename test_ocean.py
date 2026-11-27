"""Final test-set pass: evaluate every run under --runs-dir on the held-out
Ocean environment (excluded from train/val via --test-envs). Reuses
eval_all's dataset cache, load_model and evaluate_split."""
import argparse, csv, os, torch
from torch.utils.data import Subset
from eval_all import (_DatasetCache, _parse_run_name, cap_windows,
                      evaluate_split, load_model)
from eval_all import default_window

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default="runs_v2")
ap.add_argument("--env", default="Ocean")
ap.add_argument("--max-windows", type=int, default=0, help="0 = all")
ap.add_argument("--eval-bs", type=int, default=8)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
cache, rows = _DatasetCache(), []
for name in sorted(os.listdir(a.runs_dir)):
    p = _parse_run_name(name)
    d = os.path.join(a.runs_dir, name)
    if not p or not os.path.isdir(d):
        continue
    for tag in ("best", "last"):
        ckp = os.path.join(d, f"{tag}.pth")
        if not os.path.exists(ckp):
            continue
        ck_args = torch.load(ckp, map_location="cpu", weights_only=False).get("args", {})
        window = ck_args.get("window") or default_window(p[0])
        ds = cache.get(data=ck_args["data"], camera=ck_args.get("camera", "front"),
                       difficulty=ck_args.get("difficulty"), envs=[a.env],
                       scale=ck_args.get("scale", 1.0), window=window,
                       frame_stride=ck_args.get("frame_stride", 1),
                       crop=ck_args.get("crop", "480x640"),
                       max_disp=ck_args.get("max_disp", 128))
        sub = Subset(ds, list(range(len(ds))))
        sub = cap_windows(sub, a.max_windows or None, 123)
        model = load_model(p[0], ckp, ck_args, device)
        epe, d1, mae, n, ms = evaluate_split(model, p[0], sub, device, a.eval_bs, a.workers)
        print(f"{name} {tag}: EPE={epe:.3f}px D1={d1:.2f}% depthMAE={mae:.3f}m n={n} {ms:.1f}ms", flush=True)
        rows.append(dict(model=p[0], run=name, ckpt=tag, env=a.env, epe=f"{epe:.4f}",
                         d1=f"{d1:.4f}", depth_mae=f"{mae:.4f}", n=n, infer_ms=f"{ms:.2f}"))
        del model; torch.cuda.empty_cache()
out = os.path.join(a.runs_dir, f"test_{a.env.lower()}.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", out)
