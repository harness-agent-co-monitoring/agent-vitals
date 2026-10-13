# Meta-Harness integration

This directory connects agent-vitals to
[Meta-Harness](https://github.com/stanford-iris-lab/meta-harness). After each
benchmark, the loop replays a fixed sample of the worker's episodes through a
local copy of the worker model and gives the proposer a per-section digest of
how the worker read its harness. The benchmark score stays the only objective.

`meta-harness-vitals.patch` adds this to three reference examples:

| Loop | Directory | Change |
|---|---|---|
| Text classification | `reference_examples/text_classification` | `--vitals`, call capture for replay, the `prompt_sections()` contract, an example declaration on `agents/no_memory.py` |
| Terminal-Bench 2 | `reference_examples/terminal_bench_2` | `--vitals`, section validation of candidates, declarations on `agents/baseline_kira.py`, `HARBOR_MODEL` selects the worker |
| HANDBOOK.md | `reference_examples/handbook` (new) | a complete loop over the HANDBOOK.md benchmark, with `--vitals` |

Without `--vitals`, each loop runs as in upstream Meta-Harness. The patch is
tested against upstream commit `8123cca`. The add-on it calls is documented in
[`vitals/addons/metaharness`](../../vitals/addons/metaharness/README.md).

## 1. Apply the patch

```bash
git clone https://github.com/stanford-iris-lab/meta-harness
cd meta-harness
git checkout 8123cca
git apply /path/to/agent-vitals/integrations/meta_harness/meta-harness-vitals.patch
```

## 2. Set up agent-vitals for replay

Replay runs the reference model on a GPU, so install agent-vitals in its own
environment and build the engine (see the main [README](../../README.md)).

```bash
python3.11 -m venv ~/vitals-env
~/vitals-env/bin/pip install -e "/path/to/agent-vitals[gpu]"
(cd /path/to/agent-vitals && scripts/build_engine.sh)

export VITALS_PYTHON=~/vitals-env/bin/python               # the loops call agent-vitals through it
export VITALS_MH_REFERENCE_GGUF=/models/worker-bf16.gguf   # the worker's weights
~/vitals-env/bin/vitals metaharness config                 # prints the resolved paths
```

| Variable | Meaning |
|---|---|
| `VITALS_PYTHON` | Python interpreter with agent-vitals installed. Defaults to the interpreter running the loop. |
| `VITALS_MH_REFERENCE_GGUF` | GGUF of the worker model, used for replay. |
| `VITALS_MH_DETOK_GGUF` | Small model with the same vocabulary, used for CPU tokenization (default: Qwen3.5-0.8B in the Hugging Face cache). |
| `VITALS_LLAMA_LIB` | Path to `libllama`. Not needed when agent-vitals is an editable install with a built engine. |
| `VITALS_GPU_SLOTS` | How many replays may hold the reference model on the GPU at once (default 2). |

With `--vitals`, a loop starts only if `VITALS_PYTHON` can import agent-vitals.
A failed replay or digest is reported and the loop continues without it.

## 3. Use an open-weight worker

Replay teacher-forces the worker's logged outputs through the same weights, so
the worker must be an open-weight model of a
[supported family](../../docs/models.md), and `VITALS_MH_REFERENCE_GGUF` must
hold those weights.

- **Text classification:** set `models:` in `config.yaml`, for example
  `- model: openrouter/qwen/qwen3.6-27b`. The upstream default (gpt-oss-120b)
  is not a supported family.
- **Terminal-Bench 2:** set `HARBOR_MODEL`, for example
  `HARBOR_MODEL=openrouter/qwen/qwen3.6-27b`. The default remains upstream's
  Claude model.
- **HANDBOOK.md:** `HARBOR_MODEL` defaults to `openrouter/qwen/qwen3.6-27b`.
  `HB_OPENROUTER_PROVIDER` pins one OpenRouter provider.

## 4. Declare sections

Read maps are reported per declared section, so every harness the proposer
writes declares its own. Each declaration is an ordered list of
`(name, exact_text)` pairs, and every text must occur verbatim in the rendered
prompt. Name sections by function (`task_framing`, `format_contract`,
`directive:1`, `completion_check`).

- **Text classification.** Implement `prompt_sections(self, input)` on the
  memory system. It returns the pairs for the prompt `predict()` builds for
  that input, covering at least 80% of it. A declaration that fails validation
  is dropped and the episode falls back to structural sections.
  `agents/no_memory.py` shows a three-section example.
- **Terminal-Bench 2 and HANDBOOK.md.** Define module-level
  `prompt_sections(system_prompt)` in the agent file, and
  `feedback_sections(observation_text)` for text the harness injects into
  observations at run time. With `--vitals`, a candidate is rejected unless its
  declaration matches the rendered prompt and is fine-grained.
  `agents/baseline_kira.py` and `agents/baseline_openhands.py` show complete
  declarations. Check a candidate with
  `python meta_harness.py --validate-agent agents.<name>:AgentHarness --vitals`.

The proposer prompt states the contract, so evolved harnesses keep declaring.
The full rules are in
[Section annotation](../../vitals/addons/metaharness/SECTION_ANNOTATION.md).

## 5. Run each loop with `--vitals`

```bash
# text classification: replays 12 validation predictions per system and dataset
cd reference_examples/text_classification
uv run python meta_harness.py --run-name my-run --iterations 10 --vitals [--vitals-subsample 12]

# Terminal-Bench 2: replays an outcome-stratified sample of 6 trials per agent
cd reference_examples/terminal_bench_2
HARBOR_MODEL=openrouter/qwen/qwen3.6-27b uv run python meta_harness.py --iterations 10 --vitals \
  [--vitals-subsample 6] [--vitals-turn-head N --vitals-turn-tail M]

# HANDBOOK.md: needs a HANDBOOK.md checkout (see reference_examples/handbook/README.md)
cd reference_examples/handbook
HB_ROOT=/path/to/handbook uv run python meta_harness.py --run-name my-run --iterations 5 --vitals [--vitals-subsample 6]
```

After the baseline and after every benchmark, the loop runs two agent-vitals
commands: a replay of the sampled episodes (`replay-inloop`,
`replay-inloop-tb`, or `replay-inloop-handbook`), then the matching digest
(`digest`, `digest-tb`, or `digest-handbook`). The digest compares each
candidate with the parent the proposer named in `base_system`.

## 6. What the proposer receives

| | Text classification (under `logs/<run>/`) | Terminal-Bench 2 and HANDBOOK.md |
|---|---|---|
| Digest per candidate | `<dataset>/<system>/<model>/vitals_digest.md` and `.parquet`, beside `val.json` | `jobs/<run>/<job>/vitals_digest.md` and `.parquet` |
| All candidates of one iteration | `reports/vitals_iteration_NN.md` | `logs/<run>/reports/vitals_iteration_NN.md` |
| Raw per-token data | `vitals/<system>--<dataset>/store` | `logs/<run>/vitals/<agent>--tb2/store`, `logs/<run>/vitals/<agent>--handbook/store` |

Each digest reports, for every declared section, its footprint in tokens, its
share of the worker's attention, read lift, peak lift with the hottest line,
consult ratio and consult contrast, and z-scores against the same role across
the candidates replayed so far. A **Delta vs parent** block reports how the worker read the
text the candidate kept, removed, and added relative to its parent, and a
per-example (classification) or per-trial map shows individual episodes.

The prompt also gives the proposer the paths of the
[schema](../../vitals/addons/metaharness/proposer_kit/VITALS_SCHEMA.md) and of
a standalone pandas loader, `vitals_data.py`, for its own analysis of the raw
stores. It adds these rules:

- Read maps are evidence and never the objective. The score stays the only
  objective.
- An observation from a read map counts only when a trace confirms it (a named
  prediction, or a trial and turn).
- Name the parent in `base_system`, and say when a read map shaped a
  hypothesis (Terminal-Bench 2 and HANDBOOK.md record it in an `evidence`
  field).
- Do not tune text to move the measures.

## Maintaining the patch

`reference_examples/terminal_bench_2/prompt_provenance.py` is a copy of
`vitals/addons/metaharness/contracts/tb.py`. Regenerate the patch after
changing that file, or after any change to the add-on's command-line options.
