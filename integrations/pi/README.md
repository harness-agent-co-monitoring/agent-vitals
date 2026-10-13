# Connect Pi to `vitals serve`

[Pi](https://github.com/earendil-works/pi) is a small TypeScript coding agent
with read, shell, edit, and write tools. It can use `vitals serve` as a custom
OpenAI-compatible provider.

## Before you begin

Install Pi and start a Vitals server.

```bash
npm install --prefix ~/pi-agent @earendil-works/pi-coding-agent

vitals serve \
  --gguf /path/to/model.gguf \
  --store runs/pi \
  --diagnostic-attention \
  --capture-attention \
  --store-messages
```

## Configure Pi

Add a `vitals` provider to `~/.pi/agent/models.json`.

```json
{
  "providers": {
    "vitals": {
      "baseUrl": "http://127.0.0.1:8033/v1",
      "api": "openai-completions",
      "apiKey": "vitals",
      "models": [
        {
          "id": "vitals-local",
          "contextWindow": 32768,
          "maxTokens": 2048,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Adjust `contextWindow` and `maxTokens` to match the server configuration and
the model.

## Run a task

```bash
~/pi-agent/node_modules/.bin/pi \
  --provider vitals \
  --model vitals-local \
  -p \
  --no-session \
  "<task>"
```

## How the integration behaves

Pi sends native tool calls. Vitals supports the XML-parameter tool-call format
used by the Qwen 3.5 template as well as the JSON convention.

Pi sends append-only request histories, so Vitals can normally join calls into
one episode without an explicit header. Command failures appear in tool
results as `Command exited with code N`, which Vitals recognizes as an error
observation.

This integration was last checked on 2026-07-09 with a 10-task tool-use run.
