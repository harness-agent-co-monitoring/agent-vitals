# Monitoring

Three layers, from cheapest to most specific. All results go to caches outside the store
([environment variables](reference/env.md)); file formats are in [monitoring outputs](reference/outputs.md).

| Layer | Command | Needs | Says |
|---|---|---|---|
| checks | `vitals monitor` | a replayed store, the GGUF for its tokenizer (no model pass) | which tool calls look wrong |
| behavior probes | `vitals ledger`, `vitals probe` | the model (GPU), labeled decisions to train | whether a behavior is coming, and which sections drive it |
| the pipeline | `vitals pipeline` | the model (GPU) | everything above, kept current, with an explanation per flag |

## Checks

```bash
vitals monitor --store runs/traces --gguf model.gguf [--reference runs/baseline]
```

Every turn gets a record (`vitals.monitor.turns`): the action, the step where it starts, the sections read at that
step, and for every tool argument its value, the exact tokens that wrote it, their log-probabilities, the sections
whose text contains the value, and the sections read while it was written. The checks (`vitals.monitor.checks`):

| Check | Fires when |
|---|---|
| `uncertain_argument` | an argument value's minimum log-probability (over its leaf values for lists and objects) is in the lowest 10% of the store's argument values (warn) or the lowest 2% (alert). `--budget` sets the share. Needs 20 argument values in the store |
| `value_not_in_context` | an id-like string value (letters and digits, no spaces) appears nowhere outside the model's own earlier text |
| `drift` | with `--reference`: a section's reading departs from the reference store (info at z ≥ 3, warn at z ≥ 6, warn for a declared section missing or new). The same statistics as the dashboard's Monitor tab |

Every alarm names its episode, turn, section, tool, argument, evidence (log-probability, threshold, where the value
was found, what was read while writing it) and a one-sentence message. `--json` prints them. Where a value came from
(a user turn, tool results only, the harness text) is evidence, not a severity rule.

## Behavior probes

A behavior probe is a logistic regression on the residual at one layer at a decision (the turn's action onset,
[concepts](concepts.md)). It answers "is this behavior happening" for a behavior you can label on some decisions:
a rule followed, an injection obeyed, a confirmation skipped.

### The ledger

`vitals ledger` records, for every decision of a store, what probes train and split on: per-head reading of every
section at the decision and over the reasoning before it, the attention each prompt position gets over that
reasoning, the residual at a few layers, each section's mean residual, and each section's writes into the hidden
state (a second teacher-forced pass, skipped with `--no-writes`).

```bash
vitals ledger --store runs/traces --gguf model.gguf
```

The default layers sit at 36%, 48%, 61% and 73% of the model's depth (23, 31, 39 and 47 of Qwen3.6-27B's 64),
moved to the nearest softmax-attention layer on hybrid models; `--layers` picks others. The ledger records what it
captured (`_ledger.json`: model file, layers, writes); a later request it does not cover (another model, a layer it
lacks, writes it skipped) recaptures it instead of reusing it.

### Training and scoring

```bash
vitals probe train --store runs/traces --labels labels.csv --name follows_injection --layer 39 --out probe.json
vitals probe score --store runs/traces --probe probe.json
```

`labels.csv` has columns `episode`, `turn`, `label` (0 or 1) and optionally `group` (folds stay within a group;
default: the episode). `--layer` must be one of the ledger's layers. Training takes seconds. The probe is scored
out of fold; its threshold is the score that 10% of decisions without the behavior exceed (`--budget`), its alert
threshold the score 2% exceed. The probe file also stores each section's normal probe lift and normal read lift over
the decisions without the behavior (`--normal-on all` uses every decision).

### The probe split

Every attention layer up to the probe's layer adds what its heads read, and those writes split exactly by section.
Applying the probe direction to each section's writes gives its signed contribution to the score
(`vitals.monitor.measures.split_report`), reported in the read map's own measures:

| Measure | What it says |
|---|---|
| contribution | the section's signed part of the score |
| probe lift | contribution per token over the uniform share, signed: toward the behavior (+) or against it (−) |
| z | probe lift against the section's normal over decisions without the behavior: what is unusual here |
| probe peak | the 8 tokens that contribute most and the prompt line holding them (with `--confirm`) |
| remainder share | the part of the score no section accounts for (embeddings, MLPs, unsplit linear-attention layers) |

## The pipeline

```bash
vitals pipeline --store runs/traces --gguf model.gguf --probes probe.json \
    [--reference runs/baseline] [--confirm] [--webhook https://example.com/hook] [--watch 60]
```

Each pass (`vitals.monitor.pipeline`):

1. replays turns not replayed yet (`--no-replay` for stores captured live with attention);
2. records the ledger of every new decision, at the default layers plus every probe's layer (`--layers`);
3. runs the checks, and drift against `--reference`;
4. scores every probe at every decision;
5. writes one **explanation per flagged decision** with both views (`vitals.monitor.flags`);
6. writes `flags.jsonl` and `explanations/<episode>.jsonl` to the alarm cache, and POSTs each flag at or above
   `--webhook-min` that was not delivered before.

A flag is a check alarm at warn or alert, or a probe score at or above its threshold. A probe whose layer the ledger
lacks is reported and produces no flags.

### What an explanation shows

- **The read map before the action.** For each section, at the decision and over the reasoning before it: the
  share of attention, read lift and read z against this store's decisions (no labels); the read peak and hottest
  line over the reasoning; and, when the episode was replayed, the turn's read lift, peak and consult contrast. The
  rows are the top sections by read lift over the reasoning, by read z and by peak, each marked with what picked it.
- **The probe split**, per probe: score against threshold, the sections ranked by z, and beside each section its
  read share over all attention layers, its read share and read lift over the layers the split covers, and read z
  for this behavior (against the decisions without it). Then the remainder share.
- **Claims**, one sentence each: the alarms, the read map's strongest sections, and per firing probe its most
  unusual section.
- **Confirmation** (`--confirm`): for each probe flag, the top sections by z are replaced with filler and the probe
  re-scored, with the probe peaks and hottest lines.

Explanations are reused while their flags, the probes and the store's read normals are unchanged.

### Webhook

`--webhook URL` POSTs one JSON object per flag: the flag's fields, the explanation's claims, the top three read map
rows and each probe's top three rows ([monitoring outputs](reference/outputs.md)). A failed delivery is logged and
tried again on the next pass, and a flag that escalates from warn to alert is sent again. `--watch N` repeats the
pass every N seconds for a store that is still growing.
