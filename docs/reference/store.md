# Trace store

A trace store is a directory with one subdirectory per episode (`vitals.core.store`). An episode directory is any
directory holding `manifest.json` or `regions.jsonl`; its name is the episode id (one directory name: no `/`, not
`.` or `..`).

```text
runs/traces/
└── <episode id>/
    ├── manifest.json            episode configuration and metadata
    ├── messages.jsonl           request messages per turn (what replay renders)
    ├── regions.jsonl            token ranges of messages and sections per turn
    ├── tokens-00000.parquet     one row per generated token (and scored prompt token)
    ├── attn_received.jsonl      attention received by each prompt position, per turn
    ├── attn_tokens.parquet      the same per prompt token, with its text and section
    ├── attn_sparse.parquet      top attended positions of each generated token
    ├── reasoning_steps.jsonl    how many leading generated tokens of each turn are reasoning
    ├── readmap.parquet          cached read map (when the episode directory is writable)
    └── ...                      optional channels, below
```

Files are optional except `manifest.json` (or `regions.jsonl`), `messages.jsonl` and the token parts, which replay
needs. Every whole-file rewrite goes through a temporary file renamed over the target.

## `manifest.json`

`EpisodeManifest` (`vitals.core.schema`): schema and library versions, engine and model, sampler and seed, capture
settings, creation time, and `meta`, a free object an ingester or harness fills. `vitals sections` and the dashboard
read `meta.reward` (1.0 correct, 0.0 incorrect), and `meta.task`, `meta.memory` (the harness or design name),
`meta.prediction` and `meta.target` when present. `vitals import` writes `meta.source`, `meta.log` and
`meta.logged_model`. To attach outcomes to your own episodes, write them into `meta` with
`TraceStore.write_manifest`.

## `messages.jsonl`

One row per turn: `turn`, `messages` (OpenAI chat format), and `tools` and `chat_kwargs` when the request had them.
Replay renders the prompt from it with the model's chat template. It may contain sensitive user and tool data.

## `regions.jsonl`

One row per turn: `turn` and `regions`, an ordered list of `{id, kind, token_start, token_end, ...}`. Message
regions are ids such as `sys_0`, `user_1`, `obs_3`, `gen`; sections are `span:<name>` (with `meta.auto` for
automatic sections). The order is the order of the `attn_share` vectors in the token parts.

## `tokens-*.parquet`

One row per generated token (`region_id == "gen"`), in parts written as they fill (`TOKEN_SCHEMA`):

| Column | Meaning |
|---|---|
| `episode_id`, `turn`, `step` | the token's episode, turn, and position among the turn's generated tokens |
| `token_id` | vocabulary id |
| `ent` | entropy of the next-token distribution (nats) |
| `m2` | margin between the two most probable tokens |
| `js` | Jensen-Shannon divergence from the previous step's distribution (nats) |
| `fork` | entropy above a running percentile after warmup |
| `kl_ref` | optional KL divergence from a reference condition |
| `surprisal` | negative log-probability of the emitted token (null when not captured) |
| `top_ids`, `top_lp` | top-K token ids and log-probabilities (null when not captured) |
| `region_id` | `gen`, or a prompt region id for scored prompt tokens |
| `attn_share` | attention shares over the turn's regions in `regions.jsonl` order, then the share on earlier generated tokens |
| `probes` | named scalar probes |
| `t_ms` | decode time of the step |

The parquet metadata carries `schema_version`; `TraceStore.read_tokens` refuses parts of another version.

## Attention files

- **`attn_received.jsonl`**: per turn, `vec`, the mean attention each prompt position received from the turn's
  generated tokens, averaged over heads and the captured attention layers. Its sum is the share of attention on the
  prompt (the rest is on earlier generated tokens).
- **`attn_tokens.parquet`**: one row per prompt token: `turn`, `idx` (position), `token_id`, `token_str`,
  `region_id` (the innermost region, a `span:` section over its message) and `attn` (as in `attn_received`).
- **`attn_sparse.parquet`**: one row per generated token: `turn`, `step`, `pos` and `w` (the top 64 prompt
  positions and weights, on the `idx` axis), `prompt_mass` and `gen_mass` (the full totals), `gpos` and `gw` (the
  top 256 earlier generated steps). Summing `w` inside a section's range recomputes its share after changing
  section bounds, low by at most the omitted tail.

The first four prompt positions hold attention-sink mass unrelated to their text; the read map leaves them out of
harness-level totals.

## Optional channels

| File | Written by | Contents |
|---|---|---|
| `section_diagnostics.parquet` | `vitals replay --section-diagnostics` | exact per-step section peaks and attention-focus JS ([section diagnostics](../section_diagnostics.md)) |
| `linear_reading.parquet`, `linear_steps.parquet` | `vitals replay --linear-reading` | linear-attention measures per section ([linear layers](../linear_layers.md)) |
| `activations.zarr` | `vitals serve --capture-layers` | residual activations, `layer<L>/turn<T>` arrays [tokens, d] |
| `events.jsonl` | `vitals serve` | harness events (turn boundaries, compactions) |
| `replay_*` | `vitals replay --sidecar` | replay channels kept apart from live capture |

Results computed from a store (alarms, ledger, explanations) live in caches outside it
([monitoring outputs](outputs.md)). Meta-Harness stores add digests beside the store
([add-on schema](../../vitals/addons/metaharness/VITALS_SCHEMA.md)).
