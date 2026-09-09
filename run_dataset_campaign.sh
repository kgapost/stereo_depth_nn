#!/bin/bash
# Orchestrates a full-factorial stereo-depth dataset campaign by repeatedly
# invoking collect_dataset.py across (env, weather, time-of-day, speed, run)
# combinations, pausing after each ride so the operator can reposition the
# drone / sanity-check the sim before the next capture starts.
#
# collect_dataset.py always launches its own AirSim binary now (no --launch
# flag), so every single ride kills any previously-running instance and
# starts a fresh one for its --env - simpler/more isolated per ride, at the
# cost of a relaunch (~5-10s) on every ride instead of once per environment.
#
# Usage:
#   ./run_dataset_campaign.sh [output_dir]
#   DRY_RUN=1 ./run_dataset_campaign.sh          # print the resolved grid, run nothing
#
# Edit the arrays below to prune the grid before a long run.
set -uo pipefail

# ===== Campaign grid =====
ENVS=(AirSimNH AbandonedPark Africa_Savannah Blocks LandscapeMountains ZhangJiajie TrapCam)
WEATHERS=(good bad)
TIMES=(morning noon dusk)
SPEEDS=(slow normal fast)
RUN_NOS=(1 2)

OUT="${1:-$HOME/datasets/airsim_stereo}"
DURATION=6
FPS=10
DRY_RUN="${DRY_RUN:-0}"
# Countdown (seconds) before a run_no repeat (same env/weather/time/speed,
# just a different run number) - the one axis with no other reason for the
# scene to differ, so it gets a dedicated "go move somewhere else" delay
# instead of an instant Enter-press pause.
REPOSITION_COUNTDOWN="${REPOSITION_COUNTDOWN:-10}"
# AirSim's depth capture has been observed to hang its RPC indefinitely,
# intermittently - collect_dataset.py has no safe way to recover from this
# itself (see fetch_image's docstring: a threaded timeout+reconnect was
# tried and made things worse, crashing instead of recovering). So recovery
# happens here instead: kill the whole ride process if it runs far longer
# than a normal ride ever should, then retry once with a fresh AirSim launch.
RIDE_TIMEOUT="${RIDE_TIMEOUT:-$((DURATION + 60))}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT="$SCRIPT_DIR/collect_dataset.py"
PROJECT_VENV_PYTHON="$SCRIPT_DIR/../obs_det_env/bin/python3"
# Default to the project's own venv (airsim/cv2/pynput live there via
# --system-site-packages) rather than trusting whatever "python3" happens to
# be first on PATH - a plain system/conda python3 is missing at least pynput,
# which silently breaks keyboard flight control (or crashes after takeoff).
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$PROJECT_VENV_PYTHON" ]; then
    PYTHON="$PROJECT_VENV_PYTHON"
  else
    PYTHON="python3"
  fi
fi
echo "Using interpreter: $PYTHON"

# time-of-day labels -> concrete AirSim datetimes. Date itself is arbitrary
# (only the clock time drives sun angle here) but held fixed so every ride
# is comparable.
declare -A TOD=(
  [morning]="2026-06-21 08:00:00"
  [noon]="2026-06-21 12:00:00"
  [dusk]="2026-06-21 19:30:00"
)

# speed labels -> concrete m/s / deg/s, scaled off utils_airsim's defaults
# (SPEED=5, Z_SPEED=5, TURN_SPEED=15).
declare -A SPEED_MAP=( [slow]=2.5 [normal]=5 [fast]=9 )
declare -A ZSPEED_MAP=( [slow]=2.5 [normal]=5 [fast]=8 )
declare -A TURN_MAP=( [slow]=8 [normal]=15 [fast]=28 )

randf() { "$PYTHON" -c "import random; print(round(random.uniform($1, $2), 2))"; }

# Runs one ride under a process-level timeout, retrying once (with a fresh
# AirSim launch) if it hangs. timeout's exit code 124 means SIGTERM didn't
# finish it in time; --kill-after=10 escalates to SIGKILL after 10 more
# seconds for a process stuck deep in a blocking C call that ignores TERM.
run_ride() {
  timeout --kill-after=10 "$RIDE_TIMEOUT" "$@"
  local status=$?
  if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "  [warn] ride timed out after ${RIDE_TIMEOUT}s (status=$status) - " \
         "killing AirSim and retrying once with a fresh launch..."
    pkill -f "AirSim_Binary/" 2>/dev/null || true
    sleep 2
    timeout --kill-after=10 "$RIDE_TIMEOUT" "$@"
    status=$?
  fi
  return $status
}

mkdir -p "$OUT"
LOG="$OUT/campaign_log.csv"
[ -f "$LOG" ] || echo "timestamp,env,weather,bad_subtype,time_label,speed_label,run_no,altitude_m,wind_mps,intensity,tag,status" > "$LOG"

total=$(( ${#ENVS[@]} * ${#WEATHERS[@]} * ${#TIMES[@]} * ${#SPEEDS[@]} * ${#RUN_NOS[@]} ))
count=0
interrupted=0
trap 'interrupted=1' INT

echo "Campaign: $total rides ($((total * DURATION / 60)) min of raw recording) -> $OUT"
[ "$DRY_RUN" = "1" ] && echo "DRY RUN: printing resolved commands only, nothing will be executed."
echo

for env in "${ENVS[@]}"; do
  [ "$interrupted" = "1" ] && break

  for weather in "${WEATHERS[@]}"; do
    for time_label in "${TIMES[@]}"; do
      for speed_label in "${SPEEDS[@]}"; do
        for run_no in "${RUN_NOS[@]}"; do
          [ "$interrupted" = "1" ] && break 5
          count=$((count + 1))
          tag="${env}_${weather}_${time_label}_${speed_label}_${run_no}"

          # Resumability: skip only if a matching dir has an actual first
          # saved frame (left/000000.png). calib.json alone doesn't prove
          # that - it's now written as a one-time pre-recording step before
          # any per-frame data is ever saved, so a ride whose depth capture
          # hangs for its entire duration still gets a calib.json with zero
          # real frames. A dir without a first frame (Ctrl+C, a hang killed
          # by run_ride, a stuck depth worker for the whole ride, or a crash)
          # is incomplete and is removed so this combination gets
          # re-collected instead of silently skipped.
          mapfile -t existing_dirs < <(compgen -G "$OUT"/seq_*_"${tag}" 2>/dev/null || true)
          already_collected=0
          for d in "${existing_dirs[@]:-}"; do
            [ -z "$d" ] && continue
            if [ -f "$d/left/000000.png" ]; then
              already_collected=1
            else
              echo "  Removing incomplete previous attempt: $d"
              rm -rf "$d"
            fi
          done
          if [ "$already_collected" = "1" ]; then
            echo "[$count/$total] SKIP (already collected): $tag"
            continue
          fi

          # Resolve weather -> concrete flags. "bad" picks one of four
          # equally-likely sub-conditions, including wind (a physics setting,
          # simSetWind, independent of the visual --weather effects).
          wind_mps=0
          intensity=0
          bad_subtype="-"
          if [ "$weather" = "good" ]; then
            weather_flag="clear"
          else
            sub=$(( RANDOM % 4 ))
            case $sub in
              0) weather_flag="rain"; bad_subtype="rain"; intensity=$(randf 0.4 0.8) ;;
              1) weather_flag="fog";  bad_subtype="fog";  intensity=$(randf 0.4 0.8) ;;
              2) weather_flag="dust"; bad_subtype="dust"; intensity=$(randf 0.4 0.8) ;;
              3) weather_flag="clear"; bad_subtype="wind"; wind_mps=$(randf 5 12) ;;
            esac
          fi
          altitude=$(randf 5 20)

          # collect_dataset.py always launches its own AirSim instance now,
          # so kill any previously-running one first - two instances would
          # otherwise fight over the same RPC port. Matched broadly on
          # "AirSim_Binary/" (not this ride's own $env) since the process
          # still running could be a *different* environment left over from
          # the previous ride's env switch.
          if [ "$DRY_RUN" != "1" ]; then
            pkill -f "AirSim_Binary/" 2>/dev/null || true
            sleep 2
          fi

          echo "=== [$count/$total] $tag ==="
          echo "    weather=$weather_flag bad_subtype=$bad_subtype intensity=$intensity wind=${wind_mps}m/s"
          echo "    time=${TOD[$time_label]} altitude=${altitude}m speed=$speed_label(${SPEED_MAP[$speed_label]}/${ZSPEED_MAP[$speed_label]}/${TURN_MAP[$speed_label]})"

          cmd=("$PYTHON" "$COLLECT"
               --out "$OUT" --tag "$tag"
               --fps "$FPS" --duration "$DURATION"
               --env "$env"
               --weather "$weather_flag" --weather-intensity "$intensity"
               --wind "$wind_mps"
               --time-of-day "${TOD[$time_label]}"
               --altitude "$altitude"
               --speed "${SPEED_MAP[$speed_label]}"
               --z-speed "${ZSPEED_MAP[$speed_label]}"
               --turn-speed "${TURN_MAP[$speed_label]}")

          printf '    %q ' "${cmd[@]}"; echo; echo

          if [ "$DRY_RUN" = "1" ]; then
            status="dry-run"
          else
            run_ride "${cmd[@]}"
            status=$?
          fi

          echo "$(date -Iseconds),$env,$weather,$bad_subtype,$time_label,$speed_label,$run_no,$altitude,$wind_mps,$intensity,$tag,$status" >> "$LOG"

          if [ "$DRY_RUN" != "1" ] && [ "$count" -lt "$total" ]; then
            if [ "$run_no" != "${RUN_NOS[-1]}" ]; then
              echo "Ride done (status=$status). Move the drone to a NEW spot for the next run_no repeat" \
                   "(same env/weather/time/speed) - recording resumes automatically."
              for ((s = REPOSITION_COUNTDOWN; s > 0; s--)); do
                [ "$interrupted" = "1" ] && break
                printf "\r  starting in %2ds... (Ctrl+C to stop the campaign)" "$s"
                sleep 1
              done
              printf "\r  starting now!                                          \n"
            else
              read -rp "Ride done (status=$status). Reposition the drone if needed, then press Enter to continue (Ctrl+C to stop the campaign)... "
            fi
          fi
        done
      done
    done
  done
done

echo
if [ "$interrupted" = "1" ]; then
  echo "Campaign interrupted after $count/$total rides -> $OUT (log: $LOG)"
  echo "Re-run this script with the same output dir to resume; already-collected tags are skipped."
else
  echo "Campaign complete: $count/$total rides -> $OUT (log: $LOG)"
fi
