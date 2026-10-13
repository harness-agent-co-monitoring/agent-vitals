# Concepts and terms

One page for the vocabulary used across the CLI, the dashboard and the output files. Each entry names the
code that defines it.

## Stores and sections

**Trace store.** A directory of episodes (`vitals.core.store`). Written by `vitals serve`, `vitals import`,
`vitals.replay.record_turn` or the Meta-Harness add-on. Layout: [store reference](reference/store.md).

**Episode, turn.** An episode is one task's directory of model calls. A turn is one call: the prompt the model saw
and the tokens it generated.

**Region.** A token range of one turn's prompt, listed in `regions.jsonl`: a message (`sys_0`, `user_1`, …), a
section (`span:<name>`), or the generated tokens (`gen`).

**Section.** A named piece of the context and the exact text it covers, stored as a `span:<name>` region. Sections
are *declared* by the harness (a sections JSON, `vitals annotate`, the `vitals.spans` request field) or added
*automatically* where nothing is declared: `system/<heading>`, `tool:<name>`, `user:<k>`, `self:<k>` (the agent's
own earlier turns), `tool_result:<tool>#<k>`. See [sections](sections.md).

**Section key.** The name under which a section is compared across decisions (`vitals.core.sections.section_keys`).
Harness sections keep their name. Conversation sections become their role (and tool) plus recency, for example
`user:latest` or `tool_result:get_order:earlier`. The turn's own reasoning is `reasoning`, chat-template text no
section covers is `template`.

## The read map

The read map says where the model looked: for each turn and section, from the attention the generated tokens pay
to the prompt (`vitals.analysis.readmap`).

**Read share.** A section's share of the attention, averaged over heads and the 16 full-attention layers of
Qwen3.6-27B (all softmax-attention layers in general).

**Read lift.** A section's attention per token relative to uniform reading of the context: share × L / (context
share × section tokens). `1.0` is an average context token. The model's attention to its own generated tokens is
left out of the normalization, so a long reasoning trace does not lower every lift.

**Peak lift, hottest line.** The mean lift of the section's 8 most-read content tokens (sections of 32 tokens or
more), and the prompt line holding most of that attention.

**Consult ratio.** mean(H·a) / (mean(H)·mean(a)) over a turn's generated steps, where H is the next-token entropy and
a the section's attention. Above 1, the section is read more when the model is uncertain. Needs 8 steps or more.

**Consult contrast.** The section's share at the 10% most uncertain steps over its share at the 10% least uncertain
steps (16 steps or more on each side).

**Footing.** The generated steps a measure averages over:

| footing | steps | exists before the action |
|---|---|---|
| turn | every generated token of the turn | no |
| decision | the action onset token only | yes |
| reasoning | every step up to and including the action onset | yes |

**Decision (action onset).** The step where the turn commits to its action: the first `<tool_call>` token after
`</think>`, scored against its strongest stored alternative, else the first message token scored against
`<tool_call>` (`vitals.explain.auto_decision`). The rule needs `</think>` and `<tool_call>` to be single tokens
(Qwen chat templates); other templates fall back to the first non-blank token.

## Monitoring

**Turn record.** What a turn did, from the store alone (`vitals.monitor.turns`): its tool calls and arguments, the
exact tokens of each argument value, their log-probabilities, the sections whose text contains the value, and the
sections read while it was written.

**Check.** A label-free test over turn records (`vitals.monitor.checks`): `uncertain_argument` (a value written with
low confidence), `value_not_in_context` (an id-like value found nowhere in the context), `drift` (section reading
against a reference store).

**False-alarm budget.** The share of ordinary cases a threshold lets through. The argument check warns below the
10th percentile of all argument confidences in the store and alerts below the 2nd; a probe's threshold is the score
10% of decisions without the behavior exceed, its alert threshold the score 2% exceed.

**Flag, severity.** A flag is a check alarm at `warn` or `alert`, or a probe score at or above its threshold.
Severities are `info`, `warn` and `alert`.

**Ledger.** A cache with one file per decision (`vitals.monitor.ledger`): per-head reading of every section at the
decision and over the reasoning, the attention each prompt position gets over the reasoning, the residual at a few
layers, each section's mean residual, and each section's writes into the hidden state. Probes are trained and
split from it without another forward pass.

**Normal.** A section key's typical value over reference decisions: the median and 1.4826 × MAD, with the spread
floored so near-constant sections do not give large z on noise (`vitals.monitor.measures.Normal`). Keys seen in
fewer than 5 decisions get no z.

**Read z.** (log2 read lift − median) / max(1.4826 MAD, 1.0) at the decision or over the reasoning. Two normals are
used: every decision in the store's ledger (no labels; the read map view of an explanation) and the decisions
without a behavior (stored with a probe; beside each probe row).

**Probe.** A logistic regression on the residual at one layer at the decision, trained on labeled decisions in
seconds (`vitals.monitor.probes`). It stores its threshold, its section normal and its read normal.

**Probe split, contribution.** Each attention layer up to the probe's layer adds what its heads read, and those
writes split exactly by section. A section's contribution c_s is the probe direction applied to its writes: how far
what the model read from that section pushes the score toward the behavior.

**Probe lift.** (c_s / section tokens) / (Σ|c| / L): contribution per token over the uniform share, signed.

**Probe z.** Probe lift against the section key's normal over decisions without the behavior: what is unusual at
this decision.

**Remainder share.** The part of the score no section accounts for (embeddings, MLPs, linear-attention layers left
unsplit): |remainder| / (|remainder| + |Σ c|).

**Explanation.** One record per flagged decision with both views: the read map rows (picked by reasoning lift,
read z and peak), and per probe the split rows ranked by probe z with read share and read z beside them
(`vitals.monitor.flags`). With `--confirm`, the top sections are replaced with filler and the probe re-scored.

## Other measurements

**Write share.** A section's share of what a token-mixing layer writes into the hidden state
(`vitals.probes.write_share`).

**Linear reading.** For linear-attention layers (Gated DeltaNet and similar): how much of each section the layer's
memory retains, its share of the memory, how much the current step reads it, and read over memory share
(`vitals.probes.linear_reading`, [linear layers](linear_layers.md)).

**Calibrated reading.** Per-head reading weighted by head weights fit to ablation effects
(`vitals.analysis.calibration`), used by `vitals explain --weights`. Not related to `vitals calibrate`, which runs
the per-model head census.

**Ablation.** Replacing a section with equal-length filler and re-decoding from the replacement onward, with the
earlier prefix reused (`vitals.probes.ablation`). A no-op replacement reproduces the original logits exactly.

**Diagnostic attention.** Flash attention turned off so attention weights can be captured. `vitals gate fa` checks
that it produces the same greedy tokens as the fast path.
