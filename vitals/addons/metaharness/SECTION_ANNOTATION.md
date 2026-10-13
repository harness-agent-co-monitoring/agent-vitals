# Declare harness sections

This page covers the Meta-Harness contract. Declaring sections in general,
including static prompt files, anchors, and checking a declaration, is
described in [Declaring sections](../../../docs/sections.md).

Vitals reports measurements for named sections of the worker's context. A
section is a contiguous span with one functional purpose, such as a task
description, policy, retrieved example, or output contract.

The harness should declare these boundaries because it knows which text it
authored and why that text exists. A structural parser remains available as a
coarser fallback.

Meta-Harness uses two declarations.

| Declaration | Text it labels | Supported tasks |
|---|---|---|
| `prompt_sections(...)` | The scaffold assembled before the worker runs | Classification and Terminal-Bench |
| `feedback_sections(observation_text)` | Harness-authored text injected into an observation during a run | Terminal-Bench |

Both return ordered `(section_name, exact_text)` pairs. The text must match the
rendered context verbatim.

## Declare the prompt scaffold

### Classification harnesses

Add this optional method to the harness base class:

```python
def prompt_sections(self, input: str) -> list[tuple[str, str]] | None:
    ...
```

The method must describe the same prompt that `predict()` builds for the given
input. Return one pair for each functionally distinct block. This can include
directives, memory profiles, demonstrations, retrieved cases, and formatting
rules.

`inner_loop.py` validates the declaration against the built prompt and stores
it in `val.json`. During ingestion, `ingest/tc.py` converts the stored pairs
into replay span requests.

### Terminal-Bench harnesses

Add this module-level function to
`terminal_bench_2/agents/<agent_name>.py`:

```python
def prompt_sections(system_prompt: str) -> list[tuple[str, str]] | None:
    ...
```

The function labels the authored portion of the first rendered prompt. In the
current harness, that is the text before `Current terminal state:`.

The function must be self-contained because Vitals extracts it from the agent
file with the Python AST. The replay environment may not be able to import the
agent's Harbor dependencies.

- Define required constants inside the function.
- Place any imports inside the function.
- Import only the standard library or `prompt_provenance`.

`agents/baseline_kira.py` is the reference implementation. It divides the
scaffold into `task_framing`, `autonomy_policy`, `completion_check`, and
`task_description`.

## Validation rules

Vitals validates a prompt declaration against the actual prompt, not the
template that was intended to produce it.

| Rule | Classification | Terminal-Bench candidates |
|---|---:|---:|
| Every declared span occurs verbatim | Required | Required |
| Declared spans cover at least 80% of the authored region | Required | Required |
| At least four sections are declared | Not required | Required |
| The declaration executes without an exception | Required | Required |

`contracts.validate_prompt_sections` implements the shared checks.
`meta_harness.py --validate-agent`, which `run_eval.sh` also invokes, rejects a
Terminal-Bench candidate when its declaration is missing, stale, too coarse,
or raises an exception. Stock Terminus scaffolds may use the structural parser
instead because that parser resolves them completely.

Classification declarations are optional. If a declaration is absent or
invalid during ingestion, Vitals uses structural segmentation.

## Choose section names

Use stable names that describe function rather than wording or position.
Create a separate span when a block serves a distinct purpose. Avoid splitting
one coherent block into arbitrary lines.

Classification uses the following vocabulary:

```text
task_framing
format_examples
memory_intro
label_inventory
coverage_example
contrastive_pair
profile:<label>
signals:<label>
instead:<label>
demo:<label>
cases:<label>
query_wrapper
query_case
directive:<n>
format_contract
```

Include the label in per-label section names. This lets downstream analyses
compare reading with the worker's chosen answer.

Terminal-Bench can add names that fit its agent, including:

```text
env_bootstrap
skills_catalogue
memory_context
tool_spec:<tool>
```

## Declare runtime feedback

A multi-turn harness may insert its own text into an observation. Examples
include a completion checklist, retry instruction, truncation warning, or
tool-result header. Without a declaration, that text is indistinguishable from
environment output after ingestion.

Add this optional module-level function to a Terminal-Bench agent:

```python
def feedback_sections(
    observation_text: str,
) -> list[tuple[str, str]] | None:
    ...
```

Return pairs only for harness-authored text inside that observation. Leave
environment output untagged. Return `None` or an empty list when the
observation contains no harness-authored text.

The same self-containment rules as `prompt_sections()` apply. Vitals checks
that every returned span occurs verbatim, but it does not impose a coverage
threshold. Candidate validation also runs the function on synthetic
observations to ensure it does not raise and returns valid spans.

`agents/baseline_kira.py` provides examples named:

```text
confirm_task_echo
confirm_checklist
truncation_warning
file_read_header
image_read_error
file_read_error
```

Declaring harness feedback prevents that text from being attributed to the
environment.

## Fallback order during ingestion

Vitals accepts a section name only after validating it against the logged
text. A failed declaration moves to the next source instead of attaching an
incorrect label.

### Classification

1. Use the declaration stored with the episode.
2. Otherwise, run the structural segmenter in `contracts/tc.py`.

The structural path uses `segment()` and `char_to_span_requests()`.

### Terminal-Bench scaffold

1. Use the trial agent's `prompt_sections()` declaration.
2. If the prompt is byte-identical to the shared Kira scaffold, try the
   `baseline_kira` declaration.
3. Otherwise, use structural spans from `contracts/tb.py`.

### Terminal-Bench feedback

1. Use the trial agent's `feedback_sections()` declaration.
2. Try the `baseline_kira` declaration where applicable.
3. Leave unmatched content as environment output.

In the resulting store, observations use IDs such as `obs:<k>`. Declared
feedback uses IDs such as `fb:<k>:<name>`.

## How replay anchors a span

A span request can carry three location hints:

- `msg_index` identifies the message containing the span.
- `within` provides a larger enclosing string.
- `occurrence` selects a repeated copy of that enclosing string.

`render_turn()` in `vitals/backends/llamacpp.py` first searches the asserted
message. It skips a span when that message has not entered the prompt yet.
This prevents a short observation from matching the same text inside a later
confirmation echo and distinguishes byte-identical feedback across turns.

Span text is stored without surrounding whitespace because chat templates may
trim trailing message whitespace.
