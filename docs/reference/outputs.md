# Monitoring outputs

Everything the monitoring commands compute goes to caches outside the store, one directory per store under a key of
its resolved path ([environment variables](env.md) for the locations). All files are rewritten whole through a
temporary file, so a reader never sees a partial one.

## Alarm cache (`vitals monitor`, `vitals pipeline`)

```text
<alarms>/<store key>/
├── <episode>.jsonl          check alarms of one episode (vitals monitor)
├── _thresholds.json         the argument-confidence thresholds used
├── flags.jsonl              every flag of the last pipeline pass
├── explanations/<episode>.jsonl   one explanation per flagged decision
└── _pipeline_state.json     flags seen and delivered, time of the last pass
```

### Check alarm

| Field | Meaning |
|---|---|
| `episode`, `turn` | where (`turn` is null for drift, which is per episode) |
| `check` | `uncertain_argument`, `value_not_in_context` or `drift` |
| `severity` | `info`, `warn` or `alert` |
| `section` | the section responsible, when there is one |
| `tool`, `arg`, `value` | the tool call and argument (argument checks) |
| `evidence` | `min_lp`, `threshold`, `source`, `found_in`, `read_while_writing` (argument checks); `kind`, `z`, `lift`, `normal` (drift) |
| `message` | one sentence |

### Flag (`flags.jsonl`)

A check alarm at warn or alert, with `kind: "check"`, or a probe flag: `episode`, `turn`, `check: "probe:<name>"`,
`kind: "probe"`, `severity` (`warn` at the probe's threshold, `alert` at its alert threshold), `score`, `message`.
Every flag has an `id` (a hash of episode, turn, check, argument and value, the same across passes) and
`explained` (whether an explanation exists for its decision).

### Explanation (`explanations/<episode>.jsonl`)

One object per flagged decision (`vitals.monitor.flags.explain_decision`):

| Field | Meaning |
|---|---|
| `episode`, `turn`, `position` | the decision (its token position in the turn's prompt plus output) |
| `decision` | `a_text`, `b_text`: the chosen token and the alternative it was scored against |
| `alarms` | the check flags at this turn |
| `read_map` | the read view, below |
| `probes` | one probe view per probe with writes at its layer, below |
| `confirmed` | with `--confirm`, per probe the result of `explain_flag`: `score`, `rows`, `confirmed` (the score without each top section), `peaks`, `claims` |
| `claims` | one-sentence claims |
| `flag_ids`, `key`, `version` | what the explanation was computed for (it is recomputed when they change) |

`read_map`:

| Field | Meaning |
|---|---|
| `rows` | one per picked section: `section`, `key`, `tokens`, `read` and `read_lift` (at the decision), `read_reasoning` and `reasoning_lift` (over the reasoning), `read_z_decision` and `read_z_reasoning` (against this store's decisions), `reason_peak` and `reason_hot_line`, and when the episode was replayed `turn_lift`, `turn_peak`, `turn_hot_line`, `consult_contrast`; `picked_by` lists the measures that put it in the top |
| `lift_by`, `z_by`, `peak_by` | the field each measure used (reasoning when the ledger has it, else the decision or the turn) |
| `best` | the top section by `lift`, `z` and `peak` over all sections |
| `normal` | `"store"` when read z is against the store's decisions, else null |
| `reasoning_share` | the share of attention at the decision on the turn's own reasoning |

Probe view: `probe`, `layer`, `score`, `threshold`, `severity`, `ranked_by` (`z`, or `contribution` without a normal),
`remainder_share`, `read_normal` (whether read z is available), and `rows`, one per section: `section`, `key`,
`tokens`, `contribution`, `contribution_linear` (the linear-attention part, when split), `share`, `lift` (probe lift),
`z` (probe z), `read` (read share over all attention layers), `read_layers` and `read_lift_layers` (over the layers at
or below the probe's layer), `read_z_decision` and `read_z_reasoning` (against the decisions without the behavior).

### Webhook payload

One POST per flag at or above `--webhook-min`, `Content-Type: application/json`:

```json
{"source": "agent-vitals", "store": "/abs/path/to/store", "episode": "...", "turn": 7, "severity": "warn",
 "check": "probe:follows_injection", "message": "...",
 "claims": ["..."],
 "read_map": [{"section": "...", "read": 0.12, "read_lift": 3.1, "reasoning_lift": 4.0, "read_z_decision": 1.2,
               "read_z_reasoning": 2.0, "reason_peak": 11.0, "reason_hot_line": "..."}],
 "probes": [{"probe": "follows_injection", "score": 2.4, "threshold": 1.4,
             "top": [{"section": "...", "lift": 5.2, "z": 2.3, "read": 0.08, "read_layers": 0.07,
                      "read_z_decision": 1.4, "read_z_reasoning": null}]}]}
```

`claims`, `read_map` and `probes` are present when the flag's decision has an explanation (`read_map` and each
probe's `top` hold up to three rows).

## Ledger cache (`vitals ledger`, `vitals pipeline`)

```text
<ledger>/<store key>/
├── _ledger.json             what was captured: version, model {name, size}, layers, writes, linear
├── ledger.parquet           one row per decision and section
├── <episode>/<turn>.npz     one file per decision
└── probe_<name>.jsonl       scores written by vitals probe score
```

`ledger.parquet` columns: `episode`, `turn`, `position`, `section`, `tokens`, `read_at`, `read_reasoning` (head-mean
read share at the decision and over the reasoning) and `reason_peak`.

Each `.npz` (`vitals.monitor.ledger.read_decision`) holds:

| Array | Shape | Meaning |
|---|---|---|
| `sections` | [R] | section names in prompt order (`output` is the turn's own reasoning) |
| `tokens` | [R] | each section's tokens up to the decision |
| `heads` | [H, 2] | (layer, head) of each row below |
| `at`, `span` | [H, R] | per-head read share at the decision, and averaged over the reasoning |
| `span_pos` | [L] | head-mean attention each prompt position gets over the reasoning |
| `residual_layers` | [K] | the captured layers |
| `residual_at` | [K, d] | the residual at the decision |
| `section_residual` | [K, R, d] | each section's mean residual |
| `writes`, `writes_linear` | [K, R, d] | each section's writes into the hidden state (absent with `--no-writes`; linear part with `--linear`) |
| `meta` | JSON | `episode`, `turn`, `step`, `position`, `a`, `b`, `a_text`, `b_text`, `prompt_tokens`, `reason_peaks` |

## Probe file (`vitals probe train`)

JSON: `format`, `name`, `layer`, `mu`, `sd`, `w`, `b` (score = w · (h − mu) / sd + b), `threshold`, and `meta` with
`alert_threshold`, `oof_auroc`, `n`, `positives`, `budget`, `folds`, `C`, `normal` (per section key: median, spread,
count of the probe lift), `normal_on`, `read_normal` (per footing, the same for log2 read lift) and `store`. Load it
with `vitals.monitor.probes.LinearProbe.load`.

## Explanations cache (`vitals explain`)

`<explanations>/<store key>/<episode>.jsonl`, one object per explained turn: `store`, `episode`, `turn`, `step`,
`decision` (position, tokens, margin, probability), `sections` (reading, calibrated reading, ablation effect, the
probability without the section, whether the decision flips), `joint`, `checks` (no-op effect, additivity gap),
`claims` and `method`.
