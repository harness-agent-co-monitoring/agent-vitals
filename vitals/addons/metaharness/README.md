# Vitals for Meta-Harness

This add-on connects `agent-vitals` to Meta-Harness. It converts benchmark logs
into Vitals trace stores, replays selected episodes, and writes section-level
digests that a harness optimizer can inspect.

The add-on supports three Meta-Harness task families:

- text classification
- Terminal-Bench
- HANDBOOK.md tasks evaluated from OpenHands completion logs

Run the add-on through:

```bash
vitals metaharness <command>
```

Use `vitals metaharness --help` to list commands. Every subcommand also accepts
`--help`.

## Data flow

A typical optimization step with Vitals feedback has four stages.

1. A candidate harness declares the sections it adds to the worker's context.
2. Meta-Harness evaluates the candidate and saves the trajectory.
3. Vitals replays a fixed subset of episodes through the reference model.
4. A CPU-only digest summarizes the read map for the next proposer step.

The benchmark score remains the optimization objective. The digest is
additional evidence and is not incorporated into the score.

## Section-level measures

Every digest uses the same four measures as the main package.

| Measure | Interpretation |
|---|---|
| **Read lift** | Attention per section token relative to an average context token. `1.0` is average. |
| **Peak lift** | Average lift of the eight most-read content tokens in a section, plus the line that contains most of their attention. Sections shorter than 32 tokens do not receive this value. |
| **Consult ratio** | Entropy-weighted attention share divided by plain attention share. Values above `1.0` indicate more reading when the next token is uncertain. |
| **Consult contrast** | Attention share at the most uncertain 10% of steps divided by the share at the least uncertain 10%. |

The implementation lives in `vitals/analysis/readmap.py` and
`vitals/analysis/reading.py`.

## Add-on structure

| Component | Path | Responsibility |
|---|---|---|
| Section contract | `contracts/` | Loads and validates `prompt_sections()` and `feedback_sections()`. Provides structural fallbacks. |
| Ingestion | `ingest/` | Converts classification `val.json`, Harbor trajectories, and OpenHands completion logs into Vitals episodes. |
| Replay | `replay/` | Replays ingested episodes through the reference model, including the one-shot in-loop replay. |
| Digests | `digest/` | Produces proposer-facing section summaries and candidate-versus-parent deltas. |
| Analysis | `analysis/` | Section roles, Terminal-Bench read maps (matrices, atlases, trajectories, events), and read-map rows for chat episodes. |

Another optimizer needs the same three interfaces: a section contract, a log
ingester, and a post-benchmark replay-and-digest hook.

## Install and configure

Install `agent-vitals` from the repository root.

```bash
python -m pip install -e .
vitals metaharness config
```

`config.py` resolves paths in this order:

1. environment variables
2. a TOML configuration file
3. built-in defaults

| Environment variable | Purpose | Default |
|---|---|---|
| `VITALS_MH_REFERENCE_GGUF` | BF16 reference model used for replay | `~/models/Qwen3.6-27B-GGUF/BF16/...` |
| `VITALS_MH_DETOK_GGUF` | Smaller same-vocabulary model used for CPU tokenization; glob patterns are accepted | Qwen 3.5 0.8B BF16 in the Hugging Face cache |
| `VITALS_MH_ROOT` | Meta-Harness `reference_examples` directory | `~/meta-harness/reference_examples` |
| `VITALS_MH_HANDBOOK_TOOLS_JSON` | Cached OpenAI tool schemas for HANDBOOK.md trials without their own `tools_openai.json` | `runs/handbook/tools_openai.json` |
| `VITALS_MH_CONFIG` | Alternate TOML configuration file | `~/.config/vitals/metaharness.toml` |

The TOML file uses the lower-case keys `reference_gguf`, `detok_gguf`, `root`,
and `handbook_tools_json`.

```toml
reference_gguf = "/models/worker-bf16.gguf"
detok_gguf = "/models/same-vocabulary-small.gguf"
root = "/work/meta-harness/reference_examples"
handbook_tools_json = "/data/handbook/tools_openai.json"
```

The replay drivers also read these environment variables directly.

| Environment variable | Purpose | Default |
|---|---|---|
| `VITALS_GPU_SLOTS` | In-loop replays that may hold a reference model on the GPU at once | `2` |
| `VITALS_LOCK_DIR` | Directory of the GPU slot lock files | `/tmp/vitals-gpu` |
| `HB_ROOT` | HANDBOOK.md benchmark checkout, added to `sys.path` to import candidate modules | `~/benchmarks/handbook` |
| `HANDBOOK_IMAGE` | Docker image that converts logged tool definitions to OpenAI schemas | `handbook_base` |

Run `vitals metaharness config` after changing the configuration. It prints
the resolved paths that the commands will use.

## Declare harness sections

The harness supplies semantic boundaries that a structural parser cannot infer
reliably.

- `prompt_sections(...)` declares the harness-authored scaffold.
- `feedback_sections(observation_text)` declares text that the harness injects
  into an observation during a multi-turn run.

Declarations contain ordered `(name, exact_text)` pairs. Vitals validates them
before associating measurements with section names. Invalid declarations fall
back to a coarser structural partition instead of producing incorrect labels.

See [Section annotation](SECTION_ANNOTATION.md) for signatures, validation
rules, naming conventions, and fallback order.

## Connect an optimization loop

The loop-side hooks ship as a patch against upstream Meta-Harness, with a
step-by-step guide, in [`integrations/meta_harness`](../../../integrations/meta_harness/README.md).
This section describes what each patched loop calls.

### Text classification

Run `text_classification/meta_harness.py` with `--vitals`. After each
benchmark, the loop calls:

1. `vitals metaharness replay-inloop` on a fixed subset of 12 episodes per
   system and dataset
2. `vitals metaharness digest` to produce the proposer-facing report

The next proposer prompt receives the digest paths, raw store path, schema
document, and standalone loader. Both hooks are non-fatal.

For a multi-call design, the per-example map reports the final call that
produced the answer. The digest adds a `calls` column and prefixes earlier
calls with `call<k>/`.

### Terminal-Bench

Run `terminal_bench_2/meta_harness.py` with `--vitals`. Candidate validation
renders the prompt and checks that `prompt_sections()` is present, current, and
fine-grained. It also probes `feedback_sections()` when provided.

After each Harbor benchmark, the loop calls:

1. `vitals metaharness replay-inloop-tb` on a fixed outcome-stratified sample
   of trials, six by default
2. `vitals metaharness digest-tb`

The digest writes `vitals_digest.md` and `vitals_digest.parquet` into the
candidate job directory. It also updates
`logs/<run>/reports/vitals_iteration_NN.md`.

The Terminal-Bench environment vendors `contracts/tb.py` as
`prompt_provenance.py`. Keep those files synchronized.

### HANDBOOK.md

The HANDBOOK.md loop follows the same pattern with:

```bash
vitals metaharness replay-inloop-handbook
vitals metaharness digest-handbook
```

These commands ingest and summarize OpenHands completion logs. A trial's
tool schemas come from its own `agent/tools_openai.json`, else from the cached
list (`--tools-json`, default `VITALS_MH_HANDBOOK_TOOLS_JSON`) when it covers
every logged tool, else from the SDK converter run in the benchmark image.

## Command reference

### Replay commands

These commands load the reference model and normally run on a GPU.

| Command | Purpose |
|---|---|
| `replay-inloop` | Replay the classification systems from the latest benchmark. Pass `--subsample 0` to replay every episode. |
| `replay-inloop-tb` | Replay one candidate's Harbor job. `--turn-head N` and `--turn-tail M` sample turns. |
| `replay-inloop-handbook` | Replay one HANDBOOK.md candidate's completion logs. |

### Reporting and analysis commands

These commands read existing stores and do not load the worker model.

| Command | Purpose |
|---|---|
| `digest` | Build a classification digest with section measures, phase summaries, hottest lines, and optional per-example rows. Use `--norm-dir` to compare with accumulated role norms. |
| `digest-tb` | Build a Terminal-Bench digest with multi-turn pooling, turn buckets, and a per-trial map. |
| `digest-handbook` | Build a HANDBOOK.md digest over harness sentences and handbook passages. |
| `diff` | Compare a candidate with its parent and report reading over retained, removed, and added prompt text. |
| `readmaps` | Build Terminal-Bench read maps and derived matrices, atlases, trajectories, and event views. |
| `config` | Print the resolved Meta-Harness paths. |

The digest commands embed a `Delta vs parent` section when they can identify a
parent. Pass `--parent-hint name=base_system` to name it explicitly or
`--no-delta` to omit the comparison. The raw hunk rows are written to
`vitals_delta.parquet`.

## Store contents

Each candidate and dataset pair produces a store with this shape:

```text
<candidate>--<dataset>/
└── store/
    └── ep-.../
        ├── manifest.json
        ├── messages.jsonl
        ├── regions.jsonl
        ├── tokens-*.parquet
        ├── attn_received.jsonl
        ├── attn_tokens.parquet
        └── attn_sparse.parquet
```

Terminal-Bench and HANDBOOK.md in-loop replay directories also contain
`replay_timing.jsonl`, with token counts, prefix-reuse counts, and elapsed time
for each episode (and, for Terminal-Bench, each turn).

See [Vitals raw-data schema](VITALS_SCHEMA.md) for the file and column
definitions. Any individual store can also be summarized with:

```bash
vitals report --store <store>
```

## Analyze replayed stores

These steps use pandas and run without a model.

```bash
# section measures across correct and incorrect outcomes
vitals sections <replay-root>

# Terminal-Bench read maps over a replay directory
vitals metaharness readmaps build --replay <replay-dir>
vitals metaharness readmaps matrices --replay <replay-dir>
vitals metaharness readmaps atlas --replay <replay-dir>
vitals metaharness readmaps events --replay <replay-dir>
```

### How ingestion reconstructs episodes

For classification, `ingest/tc.py` converts each `val.json` prediction into
one Vitals episode per worker call. Episode IDs end in `-c<k>`, and the
manifest records `call_idx` and `n_calls`. When captured reasoning is
available, replay teacher-forces it at its original position before the final
content.

For Terminal-Bench, `ingest/tb.py` reconstructs the prompt visible at every
turn from the Harbor trajectory. Scaffold sections come from the trial
agent's `prompt_sections()` declaration, then the baseline declaration, then
the structural parser. Feedback declarations label harness text inside
observations. Untagged text remains environment output.

## Replay efficiency

The replay drivers use the core replay with prefix reuse across turns and episodes, capture limited to the
positions that need it, pooling on the GPU, and one write per episode; the optimized path keeps received attention
within a 0.999 correlation of the plain one. The settings are described in
[replay](../../../docs/replay.md#speed) (`VITALS_REPLAY_CHUNK`, `VITALS_REPLAY_HEAD_MEAN`,
`VITALS_REPLAY_DEVICE_POOL`).

## Comparison with Vitals feedback

With `--vitals`, the loop reports section footprint, attention share, read
lift, peak lift, consult ratio, consult contrast, role-normalized values, and
the delta against the parent. The proposer can use those measurements when
forming a hypothesis or deciding where to place an edit. The benchmark score
remains the sole optimization objective.

`digest/delta.py` token-diffs the candidate and parent prompts on paired
examples. Each hunk is classified as:

- **retained**, for text present in both prompts
- **removed**, for parent-only text
- **added**, for candidate-only text

The `rel` field normalizes a hunk's lift by the mean lift in its own prompt.
The parent comes from `base_system` when supplied through `--parent-hint`.
Otherwise, the digest selects the existing design with the closest prompt
overlap.

Run the comparison directly with:

```bash
vitals metaharness diff \
  --vitals-dir <replay-root> \
  --child <candidate> \
  --parent <parent> \
  --dataset LawBench
```

## Add another optimizer

Implement three adapters.

1. **Section contract.** Add `prompt_sections()` and, for runtime-injected
   text, `feedback_sections()`. Validate both during candidate acceptance.
2. **Log ingestion.** Convert stored trajectories into `TraceStore` episodes.
   Use `ingest/tc.py` as a single-turn example and `ingest/tb.py` as a
   multi-turn example.
3. **Digest hook.** After each benchmark, call a `replay-inloop`-style command,
   run the appropriate digest, and expose its path to the proposer.

The existing reporting and analysis code can then operate on the resulting
stores.

## Reference documents

- [Section annotation](SECTION_ANNOTATION.md) describes declaration
  signatures, validation, and fallbacks.
- [Vitals raw-data schema](VITALS_SCHEMA.md) describes replay artifacts and
  their columns.
- [Proposer kit](proposer_kit/README.md) describes the isolated schema and
  loader supplied to the proposer in runs with Vitals feedback.
