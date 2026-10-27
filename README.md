# agent-vitals

Section-level interpretability and monitoring for LLM agent harnesses.

An agent harness assembles the model's context from parts: instructions, rules, tool schemas, examples, user turns,
tool results. `agent-vitals` measures how an open-weight model used each part. It teacher-forces logged agent
calls through a local GGUF copy of the model (or captures them live), and reports for every **section** and turn
where the model looked, which sections drove a decision, and which turns look wrong.

- **Read maps.** For each section and turn: read lift, peak lift with the hottest line, consult ratio and contrast.
  Did the agent revisit the completion checklist near the end? Does it read one line of a long policy and skip the
  rest? Does it consult the label list when it is unsure?
- **Monitoring.** Label-free checks and behavior probes trained in seconds on labeled decisions, and for every flag an
  explanation that shows the read map and the probe split by section side by side.
- **Explanations.** 

Sections are the unit of everything reported. The harness can declare them, `vitals annotate` derives them from a
prompt file, and without either, automatic sections cover the system prompt by heading, each tool schema, each user
turn and each tool result. See [declaring sections](docs/sections.md).

These are measurements of attention and hidden states. A section the model reads heavily did not necessarily
cause its behavior. The ablation in `vitals explain` is the causal check.

## Install

Requires Python 3.11+, Linux, a C/C++ toolchain with CMake, and a CUDA toolkit for GPU replay.

```bash
git clone https://github.com/zwycl/agent-vitals && cd agent-vitals
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[server,gpu]"
scripts/build_engine.sh          # builds the pinned llama.cpp release (CUDA=OFF for a CPU build)
vitals engine doctor
```

| Extra | Adds | Needed for |
|---|---|---|
| (none) | numpy, pandas, pyarrow, scipy, zarr, gguf | `import`, `replay`, `report`, `annotate`, `monitor`, `sections` |
| `server` | FastAPI, Uvicorn | `serve`, `dashboard` |
| `gpu` | PyTorch | `ledger`, `probe`, `pipeline`, `explain`, `--linear-reading`, write shares |
| `gepa` | GEPA | the GEPA add-on |
| `dev` | all of the above, pytest, ruff | development |

Other llama.cpp builds and the test variables: [installation](docs/install.md).

## Quickstart

The repository ships a five-call log of an airline support agent (`examples/calls.jsonl`, OpenAI chat format with
tool calls). A Qwen3.5 GGUF is the reference model. Qwen3.5-0.8B runs the whole quickstart in a few minutes on one
GPU.

```bash
GGUF=/path/to/Qwen3.5-0.8B-BF16.gguf

vitals import  --log examples/calls.jsonl --store runs/quickstart --gguf $GGUF   # 2 episodes, 5 turns
vitals replay  --store runs/quickstart --gguf $GGUF                              # attention, next-token statistics
vitals report  --store runs/quickstart                                           # read map per turn and section
vitals monitor --store runs/quickstart --gguf $GGUF                              # label-free checks
vitals dashboard runs/quickstart                                                 # http://127.0.0.1:8040/
```

`vitals monitor` flags the reservation id that `session-b` passes to `get_reservation`: the agent wrote it without
it appearing anywhere in the context. `vitals pipeline --store runs/quickstart --gguf $GGUF` runs replay, ledger,
checks and explanations in one pass and keeps them current with `--watch`. [Monitoring](docs/monitoring.md) adds
behavior probes and a webhook.

## Workflows

| Start from | Commands | Guide |
|---|---|---|
| Logged OpenAI-style calls of any agent | `vitals import`, then `vitals replay` | [your own harness](docs/own_harness.md) |
| A model you serve locally | `vitals serve` (records every call) | [live capture](docs/serve.md) |
| A trace store | `vitals report`, `vitals dashboard` | [replay](docs/replay.md), [dashboard](docs/dashboard.md) |
| Monitoring a store | `vitals monitor`, `ledger`, `probe`, `pipeline` | [monitoring](docs/monitoring.md) |
| One decision to explain | `vitals explain` | [explanations](docs/explain.md) |
| A harness optimizer | `vitals metaharness`, `vitals.addons.gepa` | [Meta-Harness](vitals/addons/metaharness/README.md), [GEPA](vitals/addons/gepa/README.md) |

From Python, `vitals` exports `TraceStore`, `LlamaCppBackend`, `record_turn`, `replay_episode`, `replay_store` and
`episode_readmap`.

## Documentation

- [Concepts and terms](docs/concepts.md): sections, read lift, peaks, consult, footings, ledger, normals, probes.
- [Declaring sections](docs/sections.md) and [your own harness](docs/own_harness.md).
- [Replay](docs/replay.md), [live capture](docs/serve.md), [dashboard](docs/dashboard.md).
- [Monitoring](docs/monitoring.md) and [explanations](docs/explain.md).
- [Supported models](docs/models.md), [linear-attention layers](docs/linear_layers.md),
  [section diagnostics](docs/section_diagnostics.md), [output sensitivity](docs/output_sensitivity.md).
- Reference: [command line](docs/reference/cli.md), [environment variables](docs/reference/env.md),
  [trace store](docs/reference/store.md), [monitoring outputs](docs/reference/outputs.md).

## Supported models

| Family | Captured for read maps | llama.cpp build |
|---|---|---|
| Qwen3.5 / Qwen3.6, 0.8B to 27B | the softmax-attention layers (1 in 4) | pinned `b9910` |
| Muse Glimmer 30B | 13 global attention layers | master |
| GLM-5.3-Flash | 11 sparse attention layers | master from 2026-09-30 |

Each architecture needs a verified graph-node map (`vitals/engine/nodes.py`). See [supported models](docs/models.md).

## Repository layout

| Path | Contents |
|---|---|
| `vitals/core` | trace store, schemas, section names, messages, cache paths, atomic file writes, segmentation |
| `vitals/engine`, `vitals/backends`, `vitals/loop` | llama.cpp binding, graph-node maps, prefill and decoding, the generation loop |
| `vitals/replay` | recording logged calls (`record_turn`, `vitals import`) and teacher-forced replay |
| `vitals/analysis` | read maps, consult measures, drift, phases, section diagnostics, calibration fits |
| `vitals/probes` | GPU measurements: per-head reading, section writes, ablation, attribution, write shares, linear reading |
| `vitals/monitor` | turn records, checks, the ledger, behavior probes, explanations of flags, the pipeline |
| `vitals/explain.py` | ablation-based explanation of one decision (`vitals explain`) |
| `vitals/server`, `vitals/dashboard` | OpenAI-compatible server with capture, and the local dashboard |
| `vitals/cli` | the `vitals` command |
| `vitals/addons` | Meta-Harness and GEPA integrations |
| `docs`, `examples`, `integrations`, `scripts`, `tests` | documentation, the quickstart log, client setups, the engine build, tests |

## Status

Pre-alpha research software!
