#!/usr/bin/env bash
# Background watcher: polls MLX export status every 60s and appends to a watch log.
# Run with: nohup bash scripts/watch_mlx_export.sh &
# Stop with: kill $(cat logs/mlx_watch.pid)
set -euo pipefail

KITCHEN="/Volumes/Data NVME/GLM-5.2-kitchen"
OUT_DIR="/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-mixed-mlx"
WATCH_LOG="$KITCHEN/logs/mlx_export_watch.log"
echo $$ > "$KITCHEN/logs/mlx_watch.pid"

echo "Watch started at $(date)" >> "$WATCH_LOG"

while true; do
    PID=$(cat "$KITCHEN/logs/mlx_export.pid" 2>/dev/null || echo "?")
    if [ "$PID" = "?" ]; then
        echo "$(date '+%H:%M:%S') | no PID file" >> "$WATCH_LOG"
        sleep 60
        continue
    fi

    if ! kill -0 "$PID" 2>/dev/null; then
        SAF=$(find "$OUT_DIR" -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
        OUT_GB=$(du -sk "$OUT_DIR" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
        INDEX=$([ -f "$OUT_DIR/model.safetensors.index.json" ] && echo "YES" || echo "NO")
        echo "$(date '+%H:%M:%S') | PROCESS DONE | shards=$SAF size=${OUT_GB}GB index=$INDEX" >> "$WATCH_LOG"
        echo "Watch ended at $(date) — process finished" >> "$WATCH_LOG"
        break
    fi

    SAF=$(find "$OUT_DIR" -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
    OUT_GB=$(du -sk "$OUT_DIR" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
    AVAIL=$(df -g "/Volumes/Data NVME" | tail -1 | awk '{print $4}')
    printf "%s | PID=%s shards=%s size=%sGB disk=%sGi\n" \
        "$(date '+%H:%M:%S')" "$PID" "$SAF" "$OUT_GB" "$AVAIL" >> "$WATCH_LOG"
    sleep 60
done
