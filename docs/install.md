# Installation

## Requirements

- Linux, Python 3.11 or newer
- A GGUF model of a [supported family](models.md)
- CMake and a C/C++ toolchain to build the engine
- A CUDA toolkit for GPU replay (a CPU build works for small models and tests)

## Package

```bash
git clone https://github.com/zwycl/agent-vitals && cd agent-vitals
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[server,gpu]"
```

| Extra | Adds | Needed for |
|---|---|---|
| (none) | numpy, pandas, pyarrow, scipy, zarr, gguf, jinja2, pydantic | `import`, `replay`, `report`, `annotate`, `monitor`, `sections`, `gate`, `calibrate` |
| `server` | FastAPI, Uvicorn | `serve`, `dashboard` |
| `gpu` | PyTorch | `ledger`, `probe train`, `pipeline`, `explain`, `replay --linear-reading`, write shares, on-device pooling during replay |
| `gepa` | GEPA | `vitals.addons.gepa` |
| `dev` | `server`, `gpu`, `gepa`, plus pytest, ruff, httpx | development |

A command that needs a missing extra stops with the `pip install` line that adds it.

## Engine

The binding targets one pinned llama.cpp release (`b9910`, `vitals.engine.PINNED_TAG`). Build it and check that the
library exposes every symbol the binding uses:

```bash
scripts/build_engine.sh            # CUDA build into vendor/llama.cpp
CUDA=OFF scripts/build_engine.sh   # CPU build
vitals engine doctor
```

The library is found at `$VITALS_LLAMA_LIB`, else under `vendor/llama.cpp/build/bin` of the working directory, its
parents, or the checkout the package is installed from. Muse Glimmer and GLM-5.3-Flash need newer builds, selected
with `VITALS_LLAMA_LIB` or `--lib` (see [supported models](models.md)). Another llama.cpp build works only if it
keeps the expected ABI; `vitals engine doctor --lib PATH` checks the symbols.

## Tests

```bash
pytest -q
```

Tests that need the engine or a model skip when those are missing; the rest run on the CPU in a few seconds. To
run the model tests, set:

| Variable | Enables |
|---|---|
| `VITALS_TEST_GGUF` | engine, replay, server, ledger, probe and pipeline tests on Qwen3.5-0.8B (default: the Hugging Face cache path of `unsloth/Qwen3.5-0.8B-GGUF`) |
| `VITALS_LLAMA_LIB` | the engine library the tests load (default: the pinned build under `vendor/`) |
| `VITALS_TEST_MUSE_LIB`, `VITALS_TEST_MUSE_GGUF` | Muse Glimmer tests (`VITALS_TEST_MUSE_GGUF` is a glob over the shards) |
| `VITALS_TEST_GLM_LIB`, `VITALS_TEST_GLM_GGUF` | GLM-5.3-Flash tests (glob over the shards) |

`ruff check .` lints the repository.
