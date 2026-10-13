# Replay

`vitals replay` teacher-forces every logged turn of a trace store through a local GGUF copy of the model and
records what the original run did not: attention to every prompt position and the next-token statistics of every
generated token. The agent's trajectory is not changed; replay only reads it.

```bash
vitals replay --store runs/traces --gguf /path/to/model-bf16.gguf
vitals report --store runs/traces            # the read map (--json for rows)
```

The store must already hold each episode's request messages and generated tokens. `vitals serve --store-messages`,
`vitals import`, `vitals.record_turn` and the Meta-Harness ingesters write that format
([trace store](reference/store.md)). Any episode directory with `messages.jsonl` is replayed, whatever its name.

For the closest match to the served model, use the same weights, tokenizer, chat template and precision. A
different quantization can still give useful measurements, but the gap should be measured: the share of forced
tokens whose replayed top-1 token (`top_ids[0]` in `tokens-*.parquet`) equals the logged token is a quick check.

## What replay writes

| Channel | Files | Option |
|---|---|---|
| attention | `attn_received.jsonl`, the per-token section shares in `tokens-*.parquet`, `attn_sparse.parquet`, `attn_tokens.parquet` | on by default; `--channels logits` skips it |
| next-token statistics | entropy, margin, surprisal and top-K log-probabilities in `tokens-*.parquet` | on by default; `--top-k` sets K |
| per-token attention | `attn_tokens.parquet`: each prompt token, its text and section, and the attention it received | on by default; `--no-token-attention` skips it. Peaks, hottest lines, the ledger and explanations need it |
| section diagnostics | `section_diagnostics.parquet` ([section diagnostics](section_diagnostics.md)) | `--section-diagnostics` |
| linear-attention layers | `linear_reading.parquet`, `linear_steps.parquet` ([linear layers](linear_layers.md)) | `--linear-reading` (needs the `gpu` extra) |

Channels already stored are skipped per turn, so an episode that grew since the last replay resumes where it
stopped. `--force` recomputes them. `--sidecar` writes `replay_*` files beside the live-capture channels instead,
for replay-versus-capture checks.

Sections come from, in order: `--spans-file`, the sections stored with the episode, and automatic sections
(`--no-auto-sections` turns those off). See [declaring sections](sections.md).

## Speed

- **Prefix reuse.** The prompt prefix shared with the next turn or episode stays resident on one sequence; each
  turn's new tokens and forced output run on a second sequence. Episodes are replayed in the order that shares the
  most prefix. Any token divergence falls back to a full prefill.
- **Capture only where needed.** Attention is captured for the final prompt chunk (which seeds the first forced
  token) and for the forced tokens.
- **Pooling on the GPU.** Attention is averaged over heads and layers on the device and reduced to section shares,
  received attention and the sparse top-K there, so no attention row is copied to the host
  (`VITALS_REPLAY_HEAD_MEAN=0` and `VITALS_REPLAY_DEVICE_POOL=0` fall back to host copies; both give the same
  measurements).
- **Chunk size.** `--chunk` (default `$VITALS_REPLAY_CHUNK`, else 64) forced tokens per engine call. 256 is 10-20%
  faster on an H200 at the price of a larger attention transient on the device.
- **Writes once per episode.** Every table a turn produces is buffered and written once per episode, through a
  temporary file renamed over the target, so a reader never sees a half-written file.

`--n-ctx` is the context per sequence (the backend holds two).

## From Python

```python
import vitals

vitals.replay_store("model.gguf", "runs/traces")                   # what the CLI runs
rows = vitals.episode_readmap("runs/traces/ep-0001")               # the read map of one episode
```

`replay_episode(backend, store, episode, ...)` replays one episode with a backend you keep loaded
(`LlamaCppBackend(gguf, n_ctx=2 * n, n_seq_max=2, diagnostic_attention=True)`).
