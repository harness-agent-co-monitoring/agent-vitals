# Using agent-vitals with your own harness

Any harness that calls an open-weight model through a chat-completions API can
produce read maps. You need:

- the model's weights as a GGUF of a [supported family](models.md), ideally at
  the precision you served;
- for every model call, the request messages, the reply exactly as the model
  produced it (including reasoning and tool calls), and the tools and chat
  template options the request used;
- a list of the sections you want measured.

There are three ways to get there: import an existing JSONL log of the calls
(`vitals import`), serve the model through `vitals serve` while the agent runs,
or write the calls into a store yourself (`record_turn`) and replay them.

## Declare your sections

A section is a name and the exact text that appears in the context.

```json
[
  {"name": "task", "text": "You are a coding agent working in /repo."},
  {"name": "rules", "text": "Never edit files outside /repo. Run the tests before finishing."},
  {"name": "output_format", "text": "End with a list of the files you changed."}
]
```

If your harness is a static prompt file, `vitals annotate prompt.md --by paragraph` writes this list
for you (see [declaring sections](sections.md)). Every text must appear verbatim in the rendered prompt. Name sections after
their function, so measurements from different harness versions stay
comparable. Without a declaration, automatic sections cover the system prompt
by heading, each tool schema, each user turn, the agent's earlier turns and each
tool result; declared sections replace them where they apply. Section boundaries
can be changed later by replaying the same logs with a new declaration.

## Import a log

Agents that log their OpenAI-style calls need no changes. Each line of the log is
`{"request": {...}, "response": {...}}` (the request body and the API response)
or `{"messages": [...], "tools": [...], "response": ...}`:

```bash
vitals import --log calls.jsonl --store runs/imported --gguf /path/to/model.gguf
vitals replay --store runs/imported --gguf /path/to/model.gguf --spans-file sections.json
```

Calls are grouped into episodes by a session id when the line has one
(`session_id`, `conversation_id`, `episode`, `metadata.session_id` or
`metadata.episode`), otherwise by prefix chaining: a call whose messages continue
an earlier call's messages and reply joins its episode. Re-importing a log, or a
log that grew, adds only the calls not stored yet. A reply with reasoning that the
default template cannot place is retried with thinking on, then stored without
the reasoning (listed in the import notes). The import only tokenizes: the model
weights stay off the GPU.

Episode outcomes (`meta.reward` and other fields `vitals sections` and the
dashboard read) are not part of a chat log; write them into each episode's
`manifest.json` ([trace store](reference/store.md)).

## Live capture with `vitals serve`

The server is an OpenAI-compatible endpoint that writes a trace store as the
agent runs.

```bash
pip install -e ".[server]"
vitals serve --gguf /path/to/model.gguf --store runs/traces \
  --diagnostic-attention --capture-attention --store-messages \
  --spans-file sections.json
```

Point the agent's client at `http://127.0.0.1:8033/v1`. Sections from
`--spans-file` apply to every request; a request can also carry its own in a
`vitals` field. Calls of one task form an episode: pass the same
`X-Vitals-Episode` header (or `metadata.episode_id`) on each call, or let the
server group calls whose message history continues the previous one.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8033/v1", api_key="local")
reply = client.chat.completions.create(
    model="local",
    messages=messages,
    tools=tools,
    extra_headers={"X-Vitals-Episode": "ep-task-17"},
    extra_body={"vitals": {"spans": sections}, "chat_template_kwargs": {"enable_thinking": True}},
)
```

Read the result with `vitals report --store runs/traces` or
`GET /v1/vitals/episodes/ep-task-17/report`. Diagnostic attention turns flash
attention off, so this mode serves more slowly than a production stack.

## Offline replay from your own logs

Keep your serving stack and log each call. Afterwards, write the calls into a
trace store and replay them through the reference model.

```python
from pathlib import Path

from vitals import LlamaCppBackend, TraceStore, episode_readmap, record_turn, replay_episode

backend = LlamaCppBackend("/path/to/model.gguf", n_ctx=32768, n_seq_max=2, diagnostic_attention=True)
store = TraceStore("runs/traces")

# one entry per model call of the task, in order
for turn, call in enumerate(logged_calls):
    reply = {"role": "assistant", "content": call["content"]}
    if call.get("reasoning"):
        reply["reasoning_content"] = call["reasoning"]
    if call.get("tool_calls"):
        reply["tool_calls"] = call["tool_calls"]
    record_turn(store, backend, "ep-task-17", turn, call["messages"], reply,
                tools=call.get("tools"), chat_kwargs=call.get("chat_kwargs"))

replay_episode(backend, "runs/traces", "ep-task-17", spans=sections)
for row in episode_readmap(Path("runs/traces") / "ep-task-17"):
    print(row["turn"], row["section"], row["lift"], row["peak"], row["consult_ratio"])
```

- `messages` is the full request history the model saw at that call, and
  `reply` is what it returned. `record_turn` renders the request with the
  model's chat template and stores the reply's tokens as the turn's output. It
  raises if the reply is not a continuation of the rendered prompt, which
  usually means the tools or template options (for example
  `{"enable_thinking": true}`) were not recorded.
- Once a store is written you can also replay it from the command line:
  `vitals replay --gguf /path/to/model.gguf --store runs/traces --spans-file sections.json`,
  then `vitals report --store runs/traces`.
- `tokens-*.parquet` stores the reference model's top token at each replayed
  step (`top_ids`). The share of steps where it equals the logged `token_id`
  shows how closely the replay follows the served model. Sampling at a
  nonzero temperature lowers this share even when the weights match.
