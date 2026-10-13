# Environment variables

## Engine

| Variable | Effect |
|---|---|
| `VITALS_LLAMA_LIB` | path of the `libllama` shared library (default: the pinned build under `vendor/llama.cpp/build/bin`); `--lib` overrides it per command |
| `VITALS_LLAMA_CTX_ABI` | `v1` or `v2`: force the llama.cpp context-parameter layout instead of detecting it |
| `VITALS_ALLOW_CUDA_GRAPHS` | `1` keeps CUDA graphs on with newer llama.cpp builds (they are turned off there because attention capture with graphs faults on long contexts) |

## Replay

| Variable | Effect |
|---|---|
| `VITALS_REPLAY_CHUNK` | forced tokens per engine call (default 64); `vitals replay --chunk` overrides it |
| `VITALS_REPLAY_HEAD_MEAN` | `0` copies per-head attention to the host instead of averaging heads on the GPU (same measurements, slower) |
| `VITALS_REPLAY_DEVICE_POOL` | `0` pools attention into section shares on the host one token at a time (same measurements, slower) |
| `VITALS_HEAD_WEIGHTS` | a JSON `{layer: [weight per head]}`: replay pools attention with these weights instead of the plain head mean (`vitals.engine.head_weights`). This changes the stored read maps and is not recorded in the store, so set it for a whole store or not at all |

## Caches

Results computed outside a store live under one root, `$VITALS_CACHE_DIR`, else `$XDG_CACHE_HOME/vitals`, else
`~/.cache/vitals` (`vitals.core.paths`). Each kind has its own directory and variable; the `--out-dir`,
`--ledger-dir`, `--explain-dir` and `--alarm-dir` options override them per command. Inside each, a store's results
sit under a 12-character key of the store's resolved path, so moving a store leaves its cached results behind.

| Variable | Default | Written by |
|---|---|---|
| `VITALS_ALARM_DIR` | `<root>/alarms` | `vitals monitor`, `vitals pipeline` (flags, explanations of flags) |
| `VITALS_LEDGER_DIR` | `<root>/ledger` | `vitals ledger`, `vitals pipeline` |
| `VITALS_EXPLAIN_DIR` | `<root>/explanations` | `vitals explain` |
| `VITALS_READMAP_CACHE` | `readmap.parquet` beside each episode when writable, else `<root>/readmaps` | read maps (`vitals report`, the dashboard, the pipeline) |

## Commands

| Variable | Effect |
|---|---|
| `VITALS_GGUF` | default `--gguf` of `vitals monitor` (which reads only the tokenizer) |
| `VITALS_DEBUG` | `1`: full tracebacks and debug logging instead of one-line errors |

## Add-ons and tests

The Meta-Harness add-on reads `VITALS_MH_REFERENCE_GGUF`, `VITALS_MH_DETOK_GGUF`, `VITALS_MH_ROOT`,
`VITALS_MH_HANDBOOK_TOOLS_JSON` and `VITALS_MH_CONFIG` ([add-on guide](../../vitals/addons/metaharness/README.md)),
plus `VITALS_GPU_SLOTS` and `VITALS_LOCK_DIR` for its GPU-slot locks. The tests read `VITALS_TEST_GGUF`,
`VITALS_TEST_MUSE_LIB`, `VITALS_TEST_MUSE_GGUF`, `VITALS_TEST_GLM_LIB` and `VITALS_TEST_GLM_GGUF`
([installation](../install.md)).
