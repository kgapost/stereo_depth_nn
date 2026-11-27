#!/usr/bin/env bash
# Re-enter the Round 3 training queue after a reboot.
#
# Installed as a cron @reboot entry. The reason it exists: a reboot on
# 2026-11-26 08:32 killed the queue mid-epoch and nobody noticed for six hours,
# which cost more GPU time than the reboot itself. run_round3.sh is resumable
# and single-instance (flock), so simply re-running it at boot is safe --
# finished jobs are skipped, an interrupted one resumes from its checkpoint,
# and if a queue is somehow already running this exits without touching it.
#
# Remove with:  crontab -l | grep -v round3_boot | crontab -
#
# Cron gives almost no environment, so everything here is absolute.

QUEUE=/home/kapost/stereo_depth_nn/run_round3.sh
LOG=/home/kapost/stereo_depth_nn/runs_v3_queue.log
DATA=/home/kapost/datasets/tartanair
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

note () { echo "[$(date -Is)] boot: $*" >> "$LOG"; }

note "@reboot fired (up $(cut -d' ' -f1 /proc/uptime)s), waiting for GPU and data"

# The driver is not always up when cron fires, and an autofs/NFS dataset mount
# may lag further behind. Poll both for up to 10 minutes rather than failing
# fast and leaving the GPU idle until someone logs in.
ready=0
for _ in $(seq 1 60); do
    if nvidia-smi -L > /dev/null 2>&1 && [ -d "$DATA" ]; then
        ready=1
        break
    fi
    sleep 10
done

if [ "$ready" -ne 1 ]; then
    note "gave up after 10 min: nvidia-smi=$(nvidia-smi -L > /dev/null 2>&1 && echo ok || echo fail)" \
         "data=$([ -d "$DATA" ] && echo ok || echo missing). Queue NOT started."
    exit 1
fi

note "GPU and data ready, entering queue"
exec /usr/bin/env bash "$QUEUE" >> "$LOG" 2>&1
