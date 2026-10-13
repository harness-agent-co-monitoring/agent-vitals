# Testing a section's effect on output predictions

The sensitivity probe compares the reference model's next-token predictions
before and after replacing one prompt section. It keeps every other prompt
token and the logged output prefix fixed, so the comparison asks whether the
prediction at that point depends on the section's original content.

This requires new inference, but not a new agent run. The probe does not sample
an alternative trajectory or measure a change in task accuracy. Use
[section diagnostics](section_diagnostics.md) to describe attention patterns
and sensitivity tests to investigate their consequences for specific predictions.

## What the test measures

At output position `t`, both conditions force the same logged tokens `y[:t]`.
The probe compares the full vocabulary distributions that predict `y[t]`:

```text
p = next-token distribution with the original prompt and prefix y[:t]
q = next-token distribution with the edited prompt and prefix y[:t]
m = (p + q) / 2
JS(p, q) = (KL(p || m) + KL(q || m)) / 2
```

JS ranges from 0 to `ln(2)`, approximately 0.693 nats. Zero means the two
distributions agree. Larger values indicate greater change, not whether the
change is helpful. The probe applies softmax to the full logits without
temperature scaling or top-k truncation. It does not reproduce the provider's
sampling policy.

At step 0, the retained output prefix is empty. Later steps include all preceding
logged tokens, even if an edited model would not have generated those tokens.
The result is therefore output dependence **conditional on the retained prefix**.

## Prepare the inputs

Use the [installed package and compatible engine](../README.md#install-from-source)
with a reference GGUF model. The selected episode needs `messages.jsonl`,
`regions.jsonl`, `attn_tokens.parquet`, and contiguous generated-token rows in
`tokens-*.parquet`. The rendered prompt must match the stored token IDs exactly.

Check the section declaration against the rendered text before testing an old
store. Matching the full prompt's token IDs does not verify its saved section
boundaries. Older Unicode declarations can have shifted bounds, and updating
Vitals does not repair existing stores. The
[section diagnostics guide](section_diagnostics.md#distinguish-pooled-hotspots-from-changing-focus)
explains the byte-alignment check and how to replay a corrected copy.

Create a JSON file mapping each condition name to replacement text or explicit
token IDs. The following shows the format only. Replace every placeholder with
content for your actual section and tokenizer:

```json
{
  "noop": {"token_ids": [101, 202, 303]},
  "alternative_text": "An equal-token-count replacement for your section."
}
```

For `noop`, supply the original token IDs from the section's half-open range
`[token_start, token_end)`. Every replacement must contain exactly that many
tokens. Text values are tokenized without adding or parsing special tokens.
The probe rejects a length mismatch instead of padding or truncating text.

Equal character counts do not imply equal token counts. The probe splices token
IDs into the selected range, preserving all outside IDs by construction. For a
text-edit experiment, also retokenize the complete edited prompt and verify
that only the intended range changes. The basic CLI does not perform this
additional check for merges across section boundaries.

## Run a comparison

Choose the section name and zero-based generated-token positions from the store:

```bash
vitals sensitivity --gguf /path/to/model.gguf --store runs/traces \
  --episode ep-example --turn 0 --section rules --steps 0,5,12 \
  --replacements replacements.json --out sensitivity.json
```

Replace the example paths, section, and steps with your own values. The section
can be named `rules` or `span:rules` and must identify one saved span in that
turn. Every requested step must exist. Use `--n-ctx` if the prompt and retained
prefix exceed the default 32,768-token capacity, and `--lib` to select a
compatible engine library. `--n-gpu-layers` controls model offloading.

The probe resets KV and recurrent state before the original condition and each
replacement. It writes a separate JSON result, leaving the trace store
unchanged. `--out` overwrites an existing output file, so choose a fresh path
when keeping results from multiple comparisons.

## Choose controls and positions before inference

1. **Include an unchanged-token control.** Its JS should be near numerical zero.
   Inspect every position, not only the mean. Fix a numerical tolerance (for
   example `1e-7` nats) in advance; it is not a significance threshold. The CLI
   neither adds this control nor enforces a threshold automatically.
2. **Score early and late predictions.** Include positions before information
   may enter the retained reasoning, as well as answer or tool-argument
   positions. Small effects at a late answer token cannot rule out earlier
   dependence. A label's first mention can be useful to inspect, but need not
   mark a decision to choose that label.
3. **Keep replacement types separate.** Reordering intact examples tests
   something different from changing their labels. Match token length and
   outside positions, then inspect the resulting text. Those controls do not
   make interventions equally strong or equally natural.
4. **Freeze and record the comparison.** Save the selected episodes, steps,
   original and replacement token IDs, section bounds, and model, library, and
   code hashes before scoring. Repeat a subset from fresh model state. The CLI
   records the reference model path, but does not create this full record of the
   inputs or freeze a study plan for you.

Preserve the original stores and perform any boundary repair on copies.

## Read the result

The output identifies the episode, turn, section bounds, prompt length, scored
steps, and reference model path. Each entry in `variants` contains:

| Field | Meaning |
|---|---|
| `replacement` | Condition name from the input JSON |
| `changed_tokens` | Number of positions whose token ID differs from the original section, not a measure of semantic edit size |
| `mean_js_nats` | Arithmetic mean over the selected positions, not over the entire turn |
| `positions[*].step` | Zero-based position in the logged output |
| `positions[*].token_id` | Original logged token being predicted |
| `positions[*].js_nats` | JS between the original and edited full vocabulary distributions |
| `positions[*].original_logprob` | Natural-log probability of the logged token under the original prompt |
| `positions[*].perturbed_logprob` | Natural-log probability of the same token under the edited prompt |

Inspect the per-position results and the surrounding logged text before
averaging. A large effect during a reference to an example label has a different
interpretation from an effect while emitting the final answer. When comparing
episodes, aggregate positions within episodes first. Additional positions and
replacement conditions do not create independent experimental samples.

Large JS demonstrates sensitivity to that particular edit under the retained
prefix. It does not show that the original section improved accuracy, that the
final answer would change under free generation, or that attention lift ranks
section importance. Small JS does not prove historical irrelevance because the
prefix may already contain information from the original section.

## Keep the three JS fields distinct

| Measurement | Comparison | Units |
|---|---|---|
| `js` in token rows | Next-token distributions at successive output steps | Nats |
| `focus_js_bits` in section diagnostics | Within-section attention distributions across steps of one trajectory | Bits |
| `js_nats` in sensitivity results | Original and edited next-token distributions at the same fixed prefix | Nats |

Only the last comparison requires a section intervention. Running
`vitals report --diagnostics` or capturing attention-focus JS does not also
run the sensitivity test.
