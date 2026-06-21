#!/usr/bin/env bash
# Download the prebuilt REAP37 + MLX 4-bit GLM-5.2 model into a separate folder.
set -euo pipefail

REPO="${REPO:-pipenetwork/GLM-5.2-REAP37-MLX-4bit}"
DEST="${DEST:-/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit}"

mkdir -p "$DEST"

echo "==> repo: $REPO"
echo "==> dest: $DEST"
echo "==> this is ~265 GB; download resumes if interrupted"

uv run --with huggingface_hub python - <<PY
from huggingface_hub import snapshot_download
repo = "$REPO"
dest = "$DEST"
path = snapshot_download(
    repo_id=repo,
    local_dir=dest,
    resume_download=True,
)
print("downloaded_to=", path)
PY

echo "==> done"
du -sh "$DEST"
find "$DEST" -maxdepth 1 -type f | wc -l | awk '{print "files=" $1}'
