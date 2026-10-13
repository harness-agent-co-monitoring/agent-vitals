# Explanations

Reading says what the model looked at. `vitals explain` says which sections pushed a decision, which way and how
far, by replacing them and measuring the change. It needs the `gpu` extra and a replayed store.

```bash
vitals explain --store runs/traces --episode ep-0001 --turns 5 --gguf model.gguf
```

The decision is the turn's action onset: the first `<tool_call>` token against its strongest stored alternative,
or the first message token against `<tool_call>` ([concepts](concepts.md)). Each explanation (`vitals.explain`)
stacks four layers:

1. **Reading.** The head-mean attention each section gets at the decision and over the reasoning before it.
2. **Calibrated reading** (`--weights`). Per-head reading weighted by head weights fit to ablation effects, which
   ranks sections by how much they are likely to matter.
3. **Ablation.** The top sections (`--ablate`, default 4) are replaced with equal-length filler and the rest is
   re-decoded from the replacement on, which gives each one's exact effect on the logit margin. Each explanation
   also reports their joint effect and a no-op replacement, whose effect must be exactly zero.
4. **Claims.** For each ablated section: the probability of the chosen token with and without it, and whether the
   decision flips. With a behavior map (`--behavior`), also the expected change in how often the model acts
   differently.

Results go to the explanations cache, never into the store. `--json` prints them in full, `--agent` as one compact
JSON object per turn. The dashboard shows them as **Why this action** in the pinned-cell panel.

## Head weights and behavior maps

`--weights` and `--behavior` take files fit on your own decisions with `vitals.analysis.calibration`:

- `fit_head_weights(units)` fits nonnegative per-head weights so that weighted reading predicts each section's share
  of the ablation effect. Each unit is one explained decision: `reading` [F, R] (one feature per row, for example
  the per-head reading of `HeadReading.measure`, for the R ablated sections) and `effect` [R] (their Δm, from
  `explain` with `--ablate` high enough to cover them). `cross_validate` compares the fit with equal weights on
  held-out decisions. Save the weights with `save_weights(path, w, features, meta)`, where `features` lists
  `("span" | "at", layer, head)` in the order of `w` (over the reasoning or at the decision).
- `fit_behavior_map(delta_m, delta_p)` fits an increasing map from a margin change to a change in behavior rate,
  measured by sampling the model with and without sections; save the returned dict as JSON.

Without them, sections are ranked by plain reading and claims stay in logits and probabilities.
