# GEPA `optimize_anything` add-on

This add-on gives GEPA's reflection LM a reading line for every candidate component: how much
attention the agent paid to it, which line it read most, and whether it turned to it when uncertain.
You change your evaluator by one line, and GEPA itself is unchanged.

## How it works

1. Your evaluator calls `vitals.record(messages, response)` after each model call.
2. When the evaluator returns, the recorded calls are teacher-forced through the open-weight
   reference model, with each candidate component declared as a section.
3. The measures are written into the evaluator's side information. GEPA shows them to the
   reflection LM when it reflects on that component.

Nothing is added to `side_info["scores"]`. The readings are evidence for the reflection LM, not
optimization targets.

## Quick start

```python
from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything
from vitals.addons.gepa import GepaVitals

vitals = GepaVitals("/models/Qwen3.6-27B-BF16-00001-of-00002.gguf", "runs/gepa-vitals")

def evaluate(candidate, example):
    messages = [{"role": "system", "content": candidate["instructions"] + "\n\n" + candidate["format"]},
                {"role": "user", "content": example["input"]}]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    vitals.record(messages, response, sections=[{"name": "input", "text": example["input"]}])
    return score(response, example), {"Output": response.choices[0].message.content}

result = optimize_anything(
    {"instructions": "...", "format": "..."},
    evaluator=vitals.wrap(evaluate),
    dataset=train, valset=val,
    objective="...",
    background=MY_BACKGROUND + "\n\n" + vitals.background,   # explains the measures once
    config=OptimizeAnythingConfig(engine="gepa", max_evals=300),
)
```

`vitals.wrap` keeps your evaluator's signature, so `example` and `opt_state` are passed as before.
A plain-string candidate is reported under GEPA's own component name, `current_candidate`.

## What the reflection LM sees

Each component's reflection record gains one field, and the evaluation gains a map of all measured
sections:

```
## Reading (agent-vitals)
read lift 16.63; peak lift 27.0 (hottest line: 'Reply with exactly one JSON object and nothing else. It has'); consult ratio 0.85; consult contrast 0.38; 41 tokens

## Read map (agent-vitals)
task_framing: read lift 2.71; consult ratio 1.26; consult contrast 1.60; 22 tokens
label_inventory: read lift 0.86; peak lift 1.9 (hottest line: '- bronchial asthma  (9x)'); ...
```

`vitals.background` defines read lift, peak lift and the consult measures. It also says that close
reading does not mean a component helps. Every measured evaluation is also appended to
`<out_dir>/evaluations.jsonl` with all section measures, for analysis after the run.

## When the candidate is code

If GEPA optimizes the code that builds the prompt, the component text never appears in the prompt.
Turn off component sections with `components=[]` and declare the sections the code produced:

```python
vitals = GepaVitals(GGUF, "runs/gepa-vitals", components=[])

def evaluate(candidate, example):
    parts = run_harness(candidate["harness"], example)        # [(section_name, text), ...]
    messages = [{"role": "user", "content": "\n\n".join(t for _, t in parts)}]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    vitals.record(messages, response, sections=[{"name": n, "text": t} for n, t in parts])
    ...
```

The read map then covers every section the harness wrote and appears in every reflection record.

## Consult measures

By default (`consult="section"`), the consult ratio and contrast are the section-specific part of
each measure: the measure divided by the same measure on the whole prompt. Some agents look back at
their prompt less overall when uncertain, which pushes every section's raw consult ratio below 1. The
section-specific part removes that shared shift. `consult="raw"` reports the measures as defined in
the main README, and `vitals.background` describes whichever is in use.

## Requirements and limits

- **Same model, or the same tokenizer and template.** The reference should be the model the
  evaluator calls. The read map warns when the reference's top choice matches less than 80% of
  the logged tokens. Greedy generation by the reference model itself matched 99.9% when tested.
- **Record the reasoning.** Replay forces exactly the recorded text. If the provider returns
  reasoning (`reasoning_content` or `reasoning`), `record` picks it up from the response.
- **Record the template options.** If the request set chat template variables, such as
  `enable_thinking`, pass them as `chat_kwargs` (per call or as a default in `GepaVitals`). An
  output that cannot follow the rendered prompt raises an error that says so.
- **Components must appear verbatim.** A component whose text is filled into a template is
  reported as not found. Declare the filled-in text with `record(..., sections=[...])`.
- **Cost.** Replays run one at a time on one GPU engine. A call with an 8k-token prompt and 1.7k
  output tokens replays in about 3.5 s on an H200, against 31 s to generate it.
  Use `sample=0.3` to measure a fraction of evaluations. Repeated evaluations hit a cache.
- **Truncated outputs.** A generation cut off at the token limit ends inside an open block. Re-rendering
  it closes that block with a few tokens the model never produced, so those calls carry a small error.
- **Evaluator only.** `batch_evaluator` is not wrapped yet.
- **Muse Glimmer.** Works the same way with `VITALS_LLAMA_LIB` pointing at its llama.cpp build. Send a
  system message (the template otherwise writes today's date into the prompt) and pass
  `chat_kwargs={"reasoning_strength": ...}` if the agent set one. Providers return reasoning with its
  surrounding whitespace trimmed, which costs about one token per segment of exactness.
- **Recording uses a context variable.** Call `record` from the evaluator's own thread, or run
  worker threads with `contextvars.copy_context().run`.

## API

| name | purpose |
|---|---|
| `GepaVitals(reference_gguf, out_dir, *, n_ctx=65536, sample=1.0, seed=0, components=None, chat_kwargs=None, min_match=0.8, backend=None, consult="section")` | the add-on; loads the reference on the first measured evaluation |
| `GepaVitals.record(messages, response=None, *, content, reasoning, tool_calls, tools, chat_kwargs, sections)` | record one model call inside a wrapped evaluator |
| `GepaVitals.wrap(evaluator)` | evaluator with the same signature that adds the readings |
| `GepaVitals.background` | text to append to `optimize_anything(background=...)` |
| `GepaVitals.measure(components, calls)` / `annotate(info, components, measured)` | the two steps of `wrap`, for custom use |
