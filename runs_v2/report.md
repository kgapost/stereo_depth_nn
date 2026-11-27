# Grid search report: runs_v2

- generated: 2026-10-06 00:37:23
- wall clock: 18h 03m 30s
- data: `/home/kapost/datasets/tartanair`
- epochs/combo: 40   eval every: 4   seed: 42
- data fraction: 0.5   crop: 480x640   amp: True   workers: 2
- axes: models=['siam2d_egomotion_fxb'] bs=[8] lr=['1e-3'] loss=['logl1']
- 6 completed, 0 failed, 0 skipped as infeasible

Ranked by **val depth MAE < 30 m** (lower is better). EPE is shown too but is not the ranking metric: the log-space losses trade near-field disparity accuracy for far-field depth accuracy, so EPE would favour the wrong objective (see losses.py).

| # | model | bs | lr | loss | status | depth MAE (m) | best EPE (px) | best epoch | final EPE (px) | sec | infer ms/win (est) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | siam2d_2dun_fxb | 8 | 1e-3 | logl1 | OK | 1.1832 | 2.2298 | 31 | 2.2487 | 14660 | 11.2 |
| 2 | siam2d_egomotion_fxb | 8 | 1e-3 | logl1 | OK | 1.3000 | 2.5790 | 35 | 2.6830 | 65007 | 122.2 |
| 3 | pyr2d_casc2d_fxb | 8 | 1e-3 | logl1 | OK | 1.4287 | 3.0796 | 31 | 3.1045 | 19787 | 10.7 |
| 4 | yolo2d_3dhg_fxb | 8 | 1e-3 | logl1 | OK | 1.4396 | 2.7369 | 31 | 2.7787 | 23972 | 11.2 |
| 5 | siam2d_3dhg_fxb | 8 | 1e-3 | logl1 | OK | 1.4869 | 2.8649 | 35 | 2.8667 | 18383 | 11.5 |
| 6 | mobile2d_3dhg_fxb | 8 | 1e-3 | logl1 | OK | 1.5159 | 2.8079 | 19 | 2.8203 | 18346 | 10.2 |

## Winner

**siam2d_2dun_fxb**, bs=8, lr=1e-3, loss=logl1 - depth MAE 1.1832 m, best EPE 2.2298 px.

Train it properly: all the data (`--data-fraction 1.0`, vs the sweep's 0.5), up to 100 epochs, stopping early after 8 evaluations without improvement.

```bash
/home/kapost/stereo_depth_nn/stereo_env/bin/python \
    /home/kapost/stereo_depth_nn/train_tartanair.py \
    --data /home/kapost/datasets/tartanair \
    --model siam2d_2dun_fxb \
    --bs 8 \
    --lr 1e-3 \
    --loss logl1 \
    --data-fraction 1.0 \
    --epochs 100 \
    --patience 8 \
    --eval-every 4 \
    --seed 42 \
    --workers 2 \
    --crop 480x640 \
    --amp \
    --out ./siam2d_2dun_fxb_bs8_lr1e-3_logl1_final
```

A caveat worth keeping: the sweep ranked configs on 0.5 of the data over 40 epochs. That is enough to separate clearly different configs, but nothing here establishes the run-to-run noise floor, so treat near-ties as ties rather than as a ranking.

Curves: http://160.40.53.49:6006
