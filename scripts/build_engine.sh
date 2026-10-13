#!/usr/bin/env bash
# Build the pinned llama.cpp engine agent-vitals binds to (tag b9910), as shared
# libraries, with CUDA. The binding looks for vendor/llama.cpp/build/bin/libllama.so
# relative to the working directory, or wherever VITALS_LLAMA_LIB points.
#
#   scripts/build_engine.sh            # CUDA build for the local GPU
#   CUDA=OFF scripts/build_engine.sh   # CPU-only build
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=b9910
CUDA=${CUDA:-ON}
ARCH=${CUDA_ARCH:-native}
if [ ! -d vendor/llama.cpp ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp vendor/llama.cpp
fi
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DGGML_CUDA="$CUDA" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build -j"$(nproc)"
echo "built vendor/llama.cpp/build/bin/libllama.so @ $TAG"
