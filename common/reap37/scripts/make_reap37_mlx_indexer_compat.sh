#!/usr/bin/env bash
# Create a separate MLX compatibility folder for REAP37 by materializing missing
# IndexShare indexer tensors. Existing large files are hardlinked; only one small
# additional safetensors shard is written.
set -euo pipefail

SRC="${SRC:-/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit}"
DEST="${DEST:-/Volumes/Data NVME/GLM-5.2-REAP37-MLX-4bit-indexer-compat}"
EXTRA="model-indexshare-compat.safetensors"

if [[ ! -f "$SRC/config.json" || ! -f "$SRC/model.safetensors.index.json" ]]; then
  echo "FATAL: source model missing config/index: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST"

echo "==> src:  $SRC"
echo "==> dest: $DEST"
echo "==> hardlinking existing files"
python3 - <<PY
import os, shutil
from pathlib import Path
src=Path("$SRC")
dst=Path("$DEST")
for p in src.iterdir():
    if p.name == ".cache" or p.name == "$EXTRA":
        continue
    q = dst / p.name
    if p.is_file():
        if q.exists():
            q.unlink()
        # The compat script modifies the index, so never hardlink it to source.
        if p.name == 'model.safetensors.index.json':
            shutil.copy2(p, q)
            continue
        try:
            os.link(p, q)
        except OSError:
            # Fallback if filesystems differ.
            if p.stat().st_size > 1024**3:
                os.symlink(p, q)
            else:
                shutil.copy2(p, q)
PY

echo "==> creating compat indexer shard"
uv run --with mlx python - <<PY
import json
from collections import defaultdict
from pathlib import Path
import mlx.core as mx

src=Path("$SRC")
dst=Path("$DEST")
extra_name="$EXTRA"
config=json.loads((src/'config.json').read_text())
index=json.loads((src/'model.safetensors.index.json').read_text())
weight_map=index['weight_map']
indexer_types=config.get('indexer_types')
if not indexer_types:
    raise SystemExit('config has no indexer_types; cannot build compat shard')

# Full indexer layers are the layers with actual indexer tensors.
full_layers=sorted({int(k.split('.')[2]) for k in weight_map if '.self_attn.indexer.' in k})
print('full indexer layers:', full_layers)

# Build dst_key -> src_key copies, then group by source shard so each large
# safetensors file is loaded at most once. MLX is used because numpy safetensors
# does not understand BF16 tensors.
copies=[]
last_full=None
for layer, typ in enumerate(indexer_types):
    if typ == 'full':
        last_full = layer
        continue
    if last_full is None:
        raise SystemExit(f'shared layer {layer} before any full indexer layer')
    src_prefix=f'model.layers.{last_full}.self_attn.indexer.'
    dst_prefix=f'model.layers.{layer}.self_attn.indexer.'
    src_keys=sorted(k for k in weight_map if k.startswith(src_prefix))
    if not src_keys:
        raise SystemExit(f'no source indexer tensors for full layer {last_full}')
    for sk in src_keys:
        dk=dst_prefix + sk[len(src_prefix):]
        if dk not in weight_map:
            copies.append((weight_map[sk], sk, dk))

if not copies:
    print('no missing compat tensors needed')
else:
    compat={}
    by_file=defaultdict(list)
    for fn, sk, dk in copies:
        by_file[fn].append((sk, dk))
    print('source shards to read:', len(by_file))
    for fn, pairs in sorted(by_file.items()):
        tensors = mx.load(str(src / fn))
        for sk, dk in pairs:
            compat[dk] = tensors[sk]
        del tensors
    mx.save_safetensors(str(dst/extra_name), compat)
    for k in compat:
        weight_map[k]=extra_name
    meta=index.setdefault('metadata', {})
    meta['indexshare_compat_note']='Duplicated previous full-layer indexer tensors for shared layers so stock mlx-lm can load REAP37.'
    meta['indexshare_compat_added_tensors']=len(compat)
    meta['total_size']=int(meta.get('total_size',0)) + (dst/extra_name).stat().st_size
    (dst/'model.safetensors.index.json').write_text(json.dumps(index, indent=2, sort_keys=True))
    print('added compat tensors:', len(compat))
    print('extra shard:', dst/extra_name, (dst/extra_name).stat().st_size/1024**2, 'MiB')
PY

echo "==> done"
du -sh "$DEST"
ls -lh "$DEST/$EXTRA"
