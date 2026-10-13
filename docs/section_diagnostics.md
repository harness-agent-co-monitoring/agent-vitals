# Understanding section diagnostics

The four standard measures remain unchanged. Optional diagnostics separate
effects that those measures combine, so you can check an interpretation
before deciding how to edit a harness.

```bash
vitals report --store runs/traces --diagnostics
vitals report --store runs/traces --diagnostics --json --out diagnostics.json
```

Each JSON row gains a `diagnostics` object. The server exposes the same object
at `/v1/vitals/episodes/EPISODE_ID/report?diagnostics=true`. The default report
keeps its existing fields and values.

## Separate context attention from section allocation

Let `a[t]` be the attention share assigned to a section at output step `t`,
`C[t]` the share assigned to the entire prompt, and `H[t]` predictive entropy.
The entire prompt includes undeclared positions. It excludes tokens generated
during the current turn. Earlier messages already in the prompt count as context.

The original consult ratio is `mean(H * a) / (mean(H) * mean(a))`. The
diagnostics apply the same calculation to `C`, then divide the section's
ratio by that context ratio.

| Field | Interpretation |
|---|---|
| `context_consult_ratio` | Whether context attention covaries positively or negatively with predictive entropy |
| `relative_consult_ratio` | The section's fraction of context attention under entropy weighting, divided by its fraction under ordinary attention-mass weighting |
| `context_consult_contrast` | Context attention at the highest-entropy steps divided by context attention at the lowest-entropy steps |
| `relative_consult_contrast` | The section's original contrast divided by the context contrast |

For example, suppose context attention rises with entropy while a section
always receives 20% of context attention. The section's consult ratio rises,
but its relative consult ratio stays at 1. The additional attention reflects
the change in total context allocation.

Each section measure equals its context factor multiplied by its relative
factor, before display rounding. Compute these factors within the same turn
and on the same steps. Products of separately computed medians do not generally
preserve the identity. Geometric means preserve it when all factors use the
same positive observations and weights.

A consult ratio of 1 means zero covariance, not statistical independence.
A section can receive more attention at both entropy extremes and still have
a ratio of 1. These factors describe associations. They do not establish a
temporal sequence of returning to context and then selecting a section.

## Compare the two lift averages

The existing `lift` and `peak` reports first average raw attention across
output steps and then normalize by the averaged context mass. They therefore
weight steps by how much attention lands on context.

For a section of `m` tokens in a prompt of `L` tokens, the diagnostics report:

| Field | Calculation |
|---|---|
| `context_attention` | `mean(C)` |
| `section_attention` | `mean(a)` |
| `section_share_mass_weighted` | `sum(a) / sum(C)` |
| `section_share_equal_step` | `mean(a / C)` |
| `lift_mass_weighted` | `(L / m) * sum(a) / sum(C)` |
| `lift_equal_step` | `(L / m) * mean(a / C)` |

Equal-step averaging gives a step with little context attention the same
weight as one with substantial context attention. Comparing the two values
shows whether a section receives a larger fraction mainly when context
attention is low. Neither convention measures whether the section helps.

`lift_mass_weighted` recomputes the existing estimator from per-step shares.
Small differences from `lift` can arise because `attn_received.jsonl` stores
each prompt position rounded to six decimal places.

Across turns, inspect section share and context length alongside lift.
For a fixed-length section, the change in log lift equals the change in log
context length plus the change in log section share, using the same averaging
convention at both turns. Lift can rise while the section share stays constant.

## Distinguish pooled hotspots from changing focus

The ordinary peak selects eight content tokens after averaging across a turn.
Per-step peak selects those eight tokens separately at every output step.
To capture the extra information, enable diagnostics during attention replay:

```bash
vitals replay --gguf /path/to/model.gguf --store runs/traces \
  --spans-file sections.json --section-diagnostics
```

For an existing annotated store, you can omit `--spans-file` when its saved
prompt token IDs exactly match the rendered prompt. Replay then reuses the
saved declarations, including observations. A mismatch requires explicit
declarations for the current rendering. Turns with attention but no diagnostic
capture are replayed to fill the missing statistics.

Check saved section boundaries as well as prompt token IDs. Older declarations
can drift after Unicode characters split across tokens, because decoding those
tokens individually inserts replacement characters. New declarations use
lossless UTF-8 token bytes and record `token_alignment: utf8_bytes` in their
metadata. The renderer rejects a tokenization that cannot reconstruct the
rendered prompt exactly instead of assigning approximate spans.

Updating the package does not repair existing stores. If an older partition is
affected, work on a copy, rebuild the declarations from their original exact
text, and force replay with `--spans-file` and `--section-diagnostics`. Matching
the old prompt IDs or repeating an old measurement exactly does not establish
that its section boundaries were correct.

The accumulator reads full attention rows before sparse truncation. It keeps
running vectors and section sums instead of saving a dense step-by-position
matrix. New `section_diagnostics.parquet` rows contain:

| Fields | Meaning |
|---|---|
| `peak_pooled_mass_weighted`, `peak_step_mass_weighted` | Pooled and per-step peaks using context-attention mass weights |
| `peak_pooled_equal_step`, `peak_step_equal_step` | The same comparison using equal weights per output step |
| `peak_gap_mass_weighted`, `peak_gap_equal_step` | `1 - pooled / step` under the corresponding weights |

Within either convention, per-step peak is at least pooled peak. The same
content-token filter, top-eight rule, and 32-token section minimum apply to
both sides of the comparison.

Consider a section of 32 tokens. Uniform attention and attention that cycles
equally through four disjoint eight-token groups have identical pooled peaks.
The moving groups have a larger per-step peak and a gap of 0.75. Uniform
attention has a gap of 0. A fixed eight-token hotspot also has a gap of 0,
so interpret the gap together with the two peaks.

The gap detects changes that alter which eight tokens form the hotspot. It
does not describe visit order, revisit counts, or all changes within the same
eight tokens. A pooled hotspot can arise from a brief burst, so a high pooled
peak alone does not establish persistent attention.

## Measure changes in attention focus with Jensen–Shannon divergence

Two sections can have the same turn-averaged attention even when one receives
broad attention at every step and the other receives narrow attention to
different tokens. The attention-focus JS measure separates these cases without
choosing a top-eight cutoff.

At step `t`, let `a[t]` be the total attention paid to this section and let
`r[t]` be that attention normalized within the section. Weight each distribution
by `omega[t] = a[t] / sum(a)`. The diagnostic computes

```text
focus_js_bits = entropy(sum(omega[t] * r[t]))
                - sum(omega[t] * entropy(r[t]))
```

Entropies here describe attention over section tokens, not uncertainty about
the next generated token. All logarithms use base two. We use whole-attention
section mass for the weights, not context-normalized shares. A step with no
attention to the section has zero weight.

The same `--section-diagnostics` replay captures JS exactly before sparse
truncation. The accumulator stores running sums, not an output-by-context
attention matrix. Reports expose these additional fields:

| Field | Meaning |
|---|---|
| `focus_js_bits` | Variation in within-section attention distributions across attended steps |
| `focus_pooled_entropy_bits` | Breadth of the pooled within-section distribution |
| `focus_step_entropy_bits` | Section-attention-weighted average breadth at each step |
| `focus_weight_entropy_bits` | Entropy of the step weights, an upper bound on JS |
| `focus_attention_mass` | Sum of whole-attention section mass across output steps |
| `focus_n_attended_steps` | Number of steps with positive section attention |

Uniform attention over 32 tokens has pooled and per-step entropies of five
bits, so JS is zero. A fixed eight-token hotspot has both entropies at three
bits and also has JS zero. Cycling equally through four disjoint eight-token
groups gives pooled entropy five, per-step entropy three, and JS two bits.
Interpret JS alongside both entropies to distinguish broad attention from a
fixed hotspot.

Unlike peak lift, JS includes punctuation and every other token in the section,
and has no 32-token minimum. Its upper bound is the smaller of `log2(section
length)` and the entropy of the step weights. Raw values therefore depend on
section size and the number and distribution of attended steps. Compare matched
sections and report those quantities; a larger JS alone is not better reading.
One attended step has JS zero. A section receiving no attention has undefined
JS and `focus_status: no_attention`.

JS detects changes in the allocation of attention, including changes within
the same top-eight tokens. It does not establish an ordered scan, revisits,
loops, or causal influence: permuting output steps leaves the result unchanged.
The measure uses the same head- and layer-averaged attention as the other
section diagnostics. Differences between individual heads or layers can still
be hidden by that earlier averaging.

## Missing data and reproducibility

Diagnostics require complete per-step section shares with a generated-token
bucket. `status` reports `missing_complete_attention` or `invalid_attention`
when these requirements fail. Sparse attention is not used as an exact fallback:
the default global top 64 prompt positions need not include a section's top eight.

Missing or duplicated output steps produce `missing_output_steps` rather than
a diagnostic computed on an incomplete prefix.

Undefined ratios are JSON `null`. The existing minimums still apply: eight
steps for consult ratio and sixteen observations in each entropy tail for
contrast. Ties use the same inclusive quantile thresholds as the original
contrast. An equal-step average is undefined if any step has zero context mass.
A mass-weighted average remains defined when total context mass is positive.

Missing per-step capture appears as `peak_status: not_captured`, not a zero
gap. A short section or one with fewer than eight content tokens is
`ineligible`. Capture whose turn size or section bounds no longer match is
`capture_mismatch`.

The diagnostic parquet file has its own `version` field, currently 2, and
records turn, section bounds, context length, output count and peak settings.
Replaying a turn replaces its diagnostic rows. Replaying attention without
diagnostic capture removes that turn's old summary. Sidecar replay writes
`replay_section_diagnostics.parquet` separately.

Version-1 captures remain readable, with `focus_status: not_captured`. Replay
with diagnostics enabled refreshes these older captures. A mismatch in section
bounds or turn size marks both peak and focus diagnostics as `capture_mismatch`.

## Test conditional output sensitivity

Attention diagnostics describe where attention goes. To test whether a section
edit also affects the next-token distribution, use the separate
[output sensitivity probe](output_sensitivity.md). It performs new inference
with the logged prefix fixed and reports full-vocabulary JS in nats.

Include positions before the final answer as well as within the answer. Once
the retained reasoning already contains a section's information, a late probe
can show little dependence even if the model depended on that section earlier.
This output JS is distinct from attention-focus JS, which compares attention
distributions across steps and is measured in bits. Neither measure establishes
whether the section improves task performance.
