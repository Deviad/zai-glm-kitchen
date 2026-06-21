#!/usr/bin/env bash
# Build llama.cpp with the glm-dsa arch + quantize tool from the kitchen's
# vendored fork submodule (or an explicit LLAMA_SRC). The homebrew llama.cpp
# (v9200) predates GLM-5.2 support and cannot load glm-dsa.
#
# Default source path is the git submodule at $ROOT/vendor/llama.cpp (Deviad
# fork, branch feature/patch_used_to_create_mixed_quantization_of_glm5.2).
# When building from the submodule, git fetch/checkout/reset steps are
# skipped so the pinned feature-branch state is preserved.
#
# Override:
#   LLAMA_SRC  — alternate llama.cpp checkout path (e.g. $HOME/projects/llama.cpp)
#   JOBS       — parallel build jobs (default: hw.ncpu)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

SRC="${LLAMA_SRC:-$ROOT/vendor/llama.cpp}"
BUILD="$SRC/build-metal"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"

echo "==> llama.cpp source: $SRC"

# Detect submodule vs. standalone vs. empty:
#   - $SRC/.git is a file  -> git submodule (work tree managed by parent repo).
#                             Do NOT touch git state; build from pinned state.
#   - $SRC/.git is a dir   -> standalone clone. Update to upstream master.
#   - $SRC/.git missing    -> not cloned yet. Clone from upstream master.
if [[ -f "$SRC/.git" ]]; then
  BRANCH="$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'detached')"
  echo "==> $SRC is a git submodule (branch: $BRANCH) — building from pinned state"
  echo "    (skipping fetch/checkout/reset to preserve the submodule's branch)"
elif [[ ! -d "$SRC/.git" ]]; then
  echo "==> cloning upstream llama.cpp (shallow, master) into $SRC"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
else
  echo "==> updating existing standalone clone to master"
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

echo "==> building llama-cli + llama-quantize + llama-trace-moe + llama-gguf-split ($JOBS jobs)"
cmake --build "$BUILD" --config Release -j "$JOBS" \
  --target llama-cli llama-quantize llama-trace-moe llama-gguf-split

LQ="$BUILD/bin/llama-quantize"
echo "==> built: $LQ"
"$LQ" --help 2>&1 | head -1

# Sanity: confirm this binary recognizes glm-dsa.
if strings "$LQ" 2>/dev/null | grep -qi "glm.moe.dsa\|glm-dsa"; then
  echo "==> glm-dsa arch recognized in binary ✓"
else
  echo "WARNING: glm-dsa string not found in binary — quantize may still work" >&2
fi

echo "==> DONE. Patched llama.cpp binaries ready at: $BUILD/bin/"
echo "    llama-cli        : $BUILD/bin/llama-cli"
echo "    llama-quantize   : $BUILD/bin/llama-quantize"
echo "    llama-trace-moe  : $BUILD/bin/llama-trace-moe"
echo "    llama-gguf-split : $BUILD/bin/llama-gguf-split"
