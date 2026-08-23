#!/bin/bash
# Heartbeat + completion sentinel for a cloud run (user rule 2026-08-21:
# every long task gets a heartbeat). Exits with a single line:
#   DONE <run>            games_final.pt appeared
#   STALE <run> age=<s>   train_log.json not updated for > STALE_MIN minutes
#   MISSING <run>         train_log.json never appeared within GRACE_MIN
# Usage: watch_run.sh <run_name> [STALE_MIN=40] [GRACE_MIN=30] [POLL_S=600]
set -u
RUN="$1"; STALE_MIN="${2:-40}"; GRACE_MIN="${3:-30}"; POLL="${4:-600}"
GCS=gs://llm-mahjong-experiments
T0=$(date +%s)
while true; do
  if timeout 60 gsutil -q stat "$GCS/$RUN/games_final.pt" 2>/dev/null; then
    echo "DONE $RUN"; exit 0
  fi
  TS=$(timeout 60 gsutil ls -l "$GCS/$RUN/train_log.json" 2>/dev/null | head -1 | awk '{print $2}')
  if [ -n "$TS" ]; then
    AGE=$(( $(date +%s) - $(date -d "$TS" +%s) ))
    if [ "$AGE" -gt $(( STALE_MIN * 60 )) ]; then
      echo "STALE $RUN age=${AGE}s"; exit 2
    fi
  elif [ $(( $(date +%s) - T0 )) -gt $(( GRACE_MIN * 60 )) ]; then
    echo "MISSING $RUN"; exit 3
  fi
  sleep "$POLL"
done
