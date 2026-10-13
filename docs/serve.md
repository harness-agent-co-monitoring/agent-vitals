# Live capture

`vitals serve` runs an OpenAI-compatible chat-completions server on a local GGUF model and records every call into
a trace store. Point an agent's OpenAI client at it and every episode becomes a store you can report on, monitor
and replay. It needs the `server` extra.

```bash
vitals serve --gguf /path/to/model.gguf --store runs/traces \
  --store-messages --diagnostic-attention --capture-attention --spans-file sections.json
```

The client's base URL is `http://127.0.0.1:8033/v1`. Calls of one conversation are grouped into an episode by
prefix matching, or by an explicit id: the `X-Vitals-Episode` request header, `metadata.episode_id`, or
`vitals.episode_id` in the request body. The response carries the episode id in the `X-Vitals-Episode` header. An
id must be one directory name (no `/`, not `.` or `..`).

| Option | What it records |
|---|---|
| `--store-messages` | each request's messages, tools and chat-template options (`messages.jsonl`), which replay needs. They may hold sensitive data |
| `--diagnostic-attention` | turns flash attention off so attention can be captured (slower serving) |
| `--capture-attention` | per-token section shares of the attention (needs `--diagnostic-attention`) |
| `--top-logprobs K` | the top-K token ids and log-probabilities and the sampled token's surprisal |
| `--prompt-scores` | entropy, margin and surprisal of the prompt tokens during prefill |
| `--capture-layers 3,15,31` | residual activations at those layers (`activations.zarr`) |
| `--spans-file` | sections attached to every request (a request's `vitals.spans` overrides them) |
| `--no-auto-sections` | no automatic sections for parts no declared section covers |
| `--reuse-kv` | keeps an episode's KV cache across turns whose prompt extends the previous one |

Serving with capture is slower than a production server. A common setup is to serve with `--store-messages` only
and recover attention later with `vitals replay`, which gives the same measurements.

## Requests and replies

Replies follow the OpenAI shape for every supported chat template:

- reasoning goes to `message.reasoning_content` and the answer to `content` (Qwen and GLM think blocks, Muse
  Glimmer's `to=self` channel);
- tool calls are parsed only when the request offers `tools`;
- template variables come from `chat_template_kwargs` (for example `{"enable_thinking": true}`) or
  `vitals.chat_kwargs`, and are stored with the messages so replay renders the same prompt.

The `vitals` request field takes `spans` (sections for this request), `capture` (a capture specification for this
request), `episode_id` and `chat_kwargs`.

## Routes

| Route | Returns |
|---|---|
| `POST /v1/chat/completions` | the completion (streaming supported) |
| `GET /v1/models` | the served model |
| `GET /health` | the model name and the pinned engine release |
| `GET /v1/vitals/episodes/{id}` | the episode's token rows (`offset`, `limit`) |
| `GET /v1/vitals/episodes/{id}/report` | manifest, per-turn summaries, the read map (`diagnostics=true` adds section diagnostics), events and regions |

Client setups for Hermes Agent, Pi and mini-SWE-agent are under `integrations/`.
