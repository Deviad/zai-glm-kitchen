#!/usr/bin/env bash
# Check status of the MLX export conversion.
# Usage: bash scripts/check_mlx_export_status.sh
set -euo pipefail

KITCHEN="/Volumes/Data NVME/GLM-5.2-kitchen"
OUT_DIR="/Volumes/Data NVME/GLM-5.2-MLX/GLM-5.2-shortgpt-pruned-mixed-mlx"
PID_FILE="$KITCHEN/logs/mlx_export.pid"

echo "════════════════════════════════════════════════════════════"
echo "  GLM-5.2 MLX Export Status  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

# Process status
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        MEM=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.1f", $1/1024/1024}')
        CPU=$(ps -o %cpu= -p "$PID" 2>/dev/null | tr -d ' ')
        ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
        echo "  Process:  RUNNING (PID $PID)"
        echo "  Elapsed:  $ELAPSED"
        echo "  CPU:      ${CPU}%    RSS: ${MEM} GB"
    else
        echo "  Process:  FINISHED or CRASHED (PID $PID not running)"
    fi
else
    echo "  Process:  PID file not found"
fi

# Log tail
LOG=$(ls -t "$KITCHEN"/logs/mlx_export_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
    echo "  Log:      $LOG"
    echo "─── Last 12 log lines ───"
    tail -12 "$LOG" 2>/dev/null
    echo "───"
fi

# Output directory stats
if [ -d "$OUT_DIR" ]; then
    SAF_COUNT=$(find "$OUT_DIR" -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
    OUT_BYTES=$(du -sk "$OUT_DIR" 2>/dev/null | awk '{print $1}')
    OUT_GB=$(echo "scale=2; $OUT_BYTES / 1048576" | bc)
    echo "  Output:   $OUT_DIR"
    echo "  Shards:   $SAF_COUNT safetensors files"
    echo "  Size:     ${OUT_GB} GB  (expected ~209 GB)"

    # Check for completion markers
    if [ -f "$OUT_DIR/model.safetensors.index.json" ]; then
        echo "  Status:   ✅ INDEX WRITTEN — conversion likely complete"
    elif [ -f "$OUT_DIR/config.json" ]; then
        echo "  Status:   ✅ CONFIG WRITTEN — finalizing"
    else
        echo "  Status:   🔄 IN PROGRESS"
    fi
else
    echo "  Output:   directory not yet created"
fi

# Disk space
echo "─── Disk space ───"
df -h "/Volumes/Data NVME" | tail -1 | awk '{printf "  Total: %s  Used: %s  Avail: %s  (%s)\n", $2, $3, $4, $5}'
echo "════════════════════════════════════════════════════════════"
