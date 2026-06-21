#!/usr/bin/env bash
# Build a fresh llama.cpp from master with the glm-dsa arch + quantize tool.
# The homebrew llama.cpp (v9200) predates GLM-5.2 support and cannot load glm-dsa.
set -euo pipefail

SRC="${LLAMA_SRC:-$HOME/projects/llama.cpp}"
BUILD="$SRC/build-metal"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"

echo "==> llama.cpp source: $SRC"
if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
else
  echo "==> updating existing clone to master"
  git -C "$SRC" fetch --depth 1 origin master
  git -C "$SRC" checkout master
  git -C "$SRC" reset --hard origin/master
fi

# Verify the clone actually has GLM-5.2 support before building.
if ! grep -rq "GlmMoeDsa" "$SRC/conversion/glm.py" 2>/dev/null; then
  echo "FATAL: $SRC/conversion/glm.py has no GlmMoeDsa converter." >&2
  echo "       GLM-5.2 support is missing. Check your checkout." >&2
  exit 1
fi
echo "==> GlmMoeDsa converter present in source ✓"

echo "==> configuring cmake (Metal enabled, release) -> $BUILD"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF

echo "==> building llama-quantize + llama-gguf-split ($JOBS jobs)"
cmake --build "$BUILD" --config Release -j "$JOBS" \
  --target llama-quantize llama-gguf-split

LQ="$BUILD/bin/llama-quantize"
echo "==> built: $LQ"
"$LQ" --help 2>&1 | head -1

# Sanity: confirm this binary recognizes glm-dsa.
if strings "$LQ" 2>/dev/null | grep -qi "glm.moe.dsa\|glm-dsa"; then
  echo "==> glm-dsa arch recognized in binary ✓"
else
  echo "WARNING: glm-dsa string not found in binary — quantize may still work" >&2
fi

echo "==> DONE. llama-quantize ready at: $LQ"
