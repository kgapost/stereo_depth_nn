#!/usr/bin/env bash
# Round 3: baseline consolidation ahead of the blind-spot method paper.
#
# Four jobs, strictly sequential (one RTX 4090, 24 GB).
#
# Settings are pinned to Round 2's (--data-fraction 0.5, --grid-epochs 40,
# bs 8, lr 1e-3, logl1, --crop 480x640, --amp, Ocean held out) wherever a
# result needs to be directly comparable to a Round 2 row. Job 2 is the
# deliberate exception: it is the full-data run.
#
# --patience defaults to 0 (never stop early), so jobs 1/3/4 run all 40 epochs
# exactly as the Round 2 sweep did. Only job 2 sets it.
#
# RESTART BEHAVIOUR. This script is safe to re-run, and is meant to be: a
# reboot on 2026-11-26 killed job 2 in epoch 56 of 100 and took jobs 3-4 with
# it. The first version guarded with `[ -f last.pth ] && skip`, which reads a
# run killed mid-flight as a finished one -- re-running it would have left the
# headline baseline frozen at epoch 55 and silently moved on. A job is now
# finished only if it said so, and an unfinished one with a checkpoint is
# resumed, which is what train.py's --resume is for (it restores weights,
# optimiser, scheduler and epoch, and appends to the existing log.csv):
#
#   .queue_done present                -> SKIP  (exited 0, early stopping incl.)
#   log.csv already has --epochs rows   -> SKIP  (and backfill the sentinel)
#   last.pth present, fewer rows        -> RESUME from last.pth
#   neither                             -> START fresh
#
# A job that keeps dying is capped at MAX_ATTEMPTS resumes so an @reboot entry
# cannot spin on it forever.

set -u -o pipefail

# Overridable so the START/RESUME/SKIP logic above can be exercised against a
# throwaway ROOT with a stub PY, instead of being debugged on live runs.
PY=${PY:-/home/kapost/stereo_depth_nn/stereo_env/bin/python}
TRAIN=${TRAIN:-/home/kapost/stereo_depth_nn/train_tartanair.py}
DATA=${DATA:-/home/kapost/datasets/tartanair}
ROOT=${ROOT:-/home/kapost/stereo_depth_nn/runs_v3}
# --workers 8, not 2. This was pinned at 2 for the first launch and that was a
# mistake on two counts, found 2026-11-27 while checking why the GPU was idle:
#
#  * SPEED. At 2 workers both loader processes sit at 101% CPU while the GPU
#    idles in a third of all samples (measured: 27/0/0/28/53/63/44/0/0/0/43/76
#    over 12 s). The box has 32 cores and 54 GB free, so this was using 1/16th
#    of it. Round 2 ran the same model at 8 workers in 366 s/epoch at half
#    data, i.e. ~733 s/epoch at full data; job 2 was taking 1545 s. The GPU,
#    not the loader, should be the limit -- README 15.1 measures 8 workers at
#    ~88% of the GPU-only ceiling.
#  * COMPARABILITY, which matters more. Augmentation (random crop and
#    photometric jitter) is drawn inside the workers from per-worker seeds, so
#    the worker count is part of the data stream, not just a speed dial. Every
#    Round 2 run used 8 workers. Jobs 3-4 exist to measure seed-to-seed noise
#    against Round 2's seed 42 -- at 2 workers that comparison would have
#    mixed seed variance with a different augmentation stream and measured
#    neither cleanly.
COMMON="--data $DATA --bs 8 --lr 1e-3 --loss logl1 --crop 480x640 --amp \
        --workers 8 --eval-every 4"
MAX_ATTEMPTS=5

mkdir -p "$ROOT/logs"

# Only one queue at a time. Without this, a manual launch and the @reboot entry
# (or two impatient launches) would put two trainings on the same 24 GB GPU and
# the same output directory. The lock is held on fd 9 for the whole run and is
# released by the kernel however the script dies, reboot included.
exec 9> "$ROOT/.queue.lock"
if ! flock -n 9; then
    echo "[$(date -Is)] another Round 3 queue already holds $ROOT/.queue.lock" \
         "-- this one is exiting without touching anything"
    exit 0
fi

# Rows in log.csv, not counting its header: epochs actually completed.
epochs_done () {
    local csv="$1"
    [ -f "$csv" ] || { echo 0; return; }
    local n
    n=$(grep -c '^[0-9]' "$csv" 2>/dev/null || echo 0)
    echo "${n:-0}"
}

run_job () {
    local name="$1"; shift
    local out="$ROOT/$name"
    local csv="$out/log.csv"
    local done_flag="$out/.queue_done"
    local attempts_file="$out/.queue_attempts"

    # --epochs is read back out of the job's own arguments rather than passed
    # separately, so the completion test cannot drift away from the setting it
    # is testing against.
    local target=0 prev=""
    for a in "$@"; do
        [ "$prev" = "--epochs" ] && target="$a"
        prev="$a"
    done

    local done_n
    done_n=$(epochs_done "$csv")

    if [ -f "$done_flag" ]; then
        echo "[$(date -Is)] SKIP  $name (finished: $(cat "$done_flag"))"
        return 0
    fi
    if [ "$target" -gt 0 ] && [ "$done_n" -ge "$target" ]; then
        echo "[$(date -Is)] SKIP  $name (log.csv already has $done_n/$target epochs)"
        echo "complete at $done_n/$target epochs (backfilled $(date -Is))" > "$done_flag"
        return 0
    fi

    local resume=()
    local mode="START" redirect=">"
    if [ "$done_n" -gt 0 ] && [ -f "$out/last.pth" ]; then
        # The cap counts *fruitless* resumes, not resumes. A job that gains
        # epochs each time is working, however many reboots it has survived;
        # one that dies at the same epoch over and over is broken.
        local n_att=0 last_ep=-1
        [ -f "$attempts_file" ] && read -r n_att last_ep < "$attempts_file"
        [ "$done_n" -gt "$last_ep" ] && n_att=0
        if [ "$n_att" -ge "$MAX_ATTEMPTS" ]; then
            echo "[$(date -Is)] GIVE UP $name ($n_att resumes with no progress" \
                 "past epoch $done_n/$target) -- investigate before retrying"
            return 1
        fi
        echo "$((n_att + 1)) $done_n" > "$attempts_file"
        resume=(--resume "$out/last.pth")
        mode="RESUME"
        echo "[$(date -Is)] RESUME $name at epoch $done_n/$target" \
             "(attempt $((n_att + 1))/$MAX_ATTEMPTS)"
    else
        mkdir -p "$out"
        echo "[$(date -Is)] START $name (0/$target epochs)"
    fi
    echo "[$(date -Is)]   args: $* ${resume[*]:-}"

    # Append, never truncate: a resumed job's earlier log is evidence too.
    "$PY" "$TRAIN" --out "$out" "$@" "${resume[@]}" \
        >> "$ROOT/logs/$name.log" 2>&1
    local rc=$?

    done_n=$(epochs_done "$csv")
    if [ "$rc" -eq 0 ]; then
        echo "finished rc=0 at $done_n/$target epochs on $(date -Is)" > "$done_flag"
    fi
    echo "[$(date -Is)] END   $name rc=$rc ($done_n/$target epochs)"
    return $rc
}

echo "[$(date -Is)] ===== Round 3 queue starting ====="

# ---------------------------------------------------------------------------
# Job 1 (~16 h, DONE 2026-11-25: 40/40 epochs, rc=0): TempoBandNet at the frame
# stride its design actually calls for. Both earlier rounds ran it at
# --frame-stride 1 (0.1 s between frames), but README 15.1 specifies stride 2
# (0.2 s) as the setting that gives its ego-motion path "real motion to learn
# from". Everything else matches its Round 2 row, so the delta isolates the
# stride.
#
# Result: 1.3017 m depth MAE / 2.5723 px val EPE, against stride 1's
# 1.3000 / 2.5790. The stride is worth nothing in final accuracy -- it only
# converges faster. So this job did NOT settle the backbone question it was run
# to settle; that choice now has to rest on mechanism, not on this number.
# ---------------------------------------------------------------------------
run_job siam2d_egomotion_fxb_stride2_seed42 \
    --model siam2d_egomotion_fxb $COMMON \
    --window 4 --frame-stride 2 \
    --data-fraction 0.5 --epochs 40 --seed 42

# ---------------------------------------------------------------------------
# Job 2 (~20 h remaining from epoch 56): the full, properly-trained run of
# Round 2's winner. This is the command report.md has been generating since
# October and that nobody ran. It is the anchor number the method paper's
# single-frame baseline is quoted from -- every ranking so far is a 0.5-data,
# 40-epoch proxy, and 75% of Round 1's configs were still improving at cutoff.
#
# Already worth more than the proxy it replaces: by epoch 55 it reached
# 1.1595 m depth MAE (ep 47) and 2.1263 px val EPE (ep 55), against Round 2's
# 1.1832 / 2.2298 for the same architecture.
#
# Caveat to carry into the paper's methods note: epochs 0-55 of this run were
# trained with --workers 2 and 56+ with --workers 8, so its augmentation stream
# changes at epoch 56 (see the COMMON note above). Harmless for a baseline
# anchor -- no result is compared against it at matched augmentation -- but it
# means an exact replay of this particular run would have to repeat the switch.
# Jobs 3-4 are unaffected: they had not started when the setting changed.
# ---------------------------------------------------------------------------
run_job siam2d_2dun_fxb_full_seed42 \
    --model siam2d_2dun_fxb $COMMON \
    --data-fraction 1.0 --epochs 100 --patience 8 --seed 42

# ---------------------------------------------------------------------------
# Jobs 3-4 (~19 h total): the noise floor. Across Rounds 1 and 2, zero of the
# 197+6 configs were ever repeated at a second seed, so every number in the
# project is n=1 -- including the 1.18 vs 1.30 m gap that decides which
# architecture is called the winner.
#
# Budget note: a full 2-extra-seed repeat of all six Round 2 models would be
# ~86 h, dominated by siam2d_egomotion_fxb at 15.8 h/run. Restricted here to
# the two architectures where a claim actually rests on the gap:
#   - siam2d_2dun_fxb   : is the top-two margin larger than run-to-run noise?
#   - pyr2d_casc2d_fxb  : is the Ocean domain-shift collapse (1.43 -> 2.32 m)
#                         a property of the architecture or one unlucky seed?
#                         This is now a reported finding, so it needs error bars.
# TempoBandNet's own seeds are deliberately deferred; job 1 was the better use
# of that GPU time.
# ---------------------------------------------------------------------------
for seed in 43 44; do
    run_job "siam2d_2dun_fxb_seed${seed}" \
        --model siam2d_2dun_fxb $COMMON \
        --data-fraction 0.5 --epochs 40 --seed "$seed"
done

for seed in 43 44; do
    run_job "pyr2d_casc2d_fxb_seed${seed}" \
        --model pyr2d_casc2d_fxb $COMMON \
        --data-fraction 0.5 --epochs 40 --seed "$seed"
done

echo "[$(date -Is)] ===== Round 3 queue finished ====="
