# Connect Hermes Agent to `vitals serve`

[Hermes Agent](https://github.com/NousResearch/hermes-agent) can use
`vitals serve` through its OpenAI-compatible interface. No Hermes source
changes are required.

The integration supports native tool calls:

- request-side `tools`
- response-side `tool_calls`
- messages with `role: "tool"`
- streaming responses over server-sent events

The server renders tools through the model's Jinja chat template. It also
parses Hermes-style `<tool_call>` blocks into OpenAI-compatible tool calls and
sets `finish_reason` to `tool_calls`.

## Before you begin

Start a Vitals server with the model you want Hermes to use.

```bash
vitals serve \
  --gguf /path/to/model.gguf \
  --store runs/hermes \
  --diagnostic-attention \
  --capture-attention \
  --store-messages
```

The model must have a chat template that supports the tools you send.

## Configure Hermes

Use these endpoint settings in the relevant Hermes provider configuration.

```yaml
base_url: http://127.0.0.1:8033/v1
api_key: vitals
model: Qwen3.5-27B-BF16-00001-of-00002
```

The API key can be any non-empty string. The server serves one model and
ignores the requested model name; responses report the GGUF file stem, which
is also returned by:

```bash
curl http://127.0.0.1:8033/v1/models
```

Hermes has additional controls for providers, model switching, and reasoning
effort. Those settings are independent of Vitals.

## Run a task

The following command shows the expected Hermes CLI arguments. Adjust the
installation paths for your machine.

```bash
~/.hermes/hermes-agent/venv/bin/python \
  ~/.hermes/hermes-agent/cli.py \
  --query "<task>" \
  --model vitals-local \
  --provider custom \
  --base_url http://127.0.0.1:8033/v1 \
  --api_key vitals \
  --max_turns 15 \
  --quiet
```

## What Vitals records

Vitals joins related requests into episodes. It uses an explicit
`X-Vitals-Episode` header when present and otherwise fingerprints append-only
message histories.

For Hermes runs, the store includes:

- tool results as observation regions, including detected error status
- tool-call turns as generated reasoning regions
- per-token next-token statistics
- optional attention and activation channels

## Operational notes

- Hermes tools run relative to the Hermes workspace, not necessarily the
  directory from which you launch the CLI. Use absolute paths in tasks that
  depend on local files.
- Tool results arrive as JSON objects such as
  `{"output": "...", "exit_code": 0}`. Vitals uses the exit code when marking
  error observations.
- Hermes session files under `~/.hermes/sessions/` can help diagnose request
  histories and episode stitching.

This integration was last checked on 2026-07-09 with a 10-task tool-use run.
