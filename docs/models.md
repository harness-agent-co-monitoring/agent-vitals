# Supported models

Loading a GGUF is not enough for attention capture. Each architecture also
needs a verified graph-node map in `vitals/engine/nodes.py`, and its chat
template must render the logged messages back to the served tokens.

| Family | Token-mixing layers | Captured for read maps | llama.cpp build |
|---|---|---|---|
| Qwen3.5 / Qwen3.6 (0.8B to 27B) | every fourth layer softmax attention, the rest Gated DeltaNet | the softmax layers | pinned `b9910` |
| Muse Glimmer 30B | 13 global attention layers, 39 sliding-window layers (2,048 tokens) | the 13 global layers | master |
| GLM-5.3-Flash | 11 compressed attention layers with sparse key selection, 34 Kimi Delta Attention layers | the 11 attention layers | master from 2026-09-30 |

The linear-attention layers of the hybrid models are measured separately with
`vitals replay --linear-reading` (see [linear layers](linear_layers.md)).

For the closest match to the served model, replay with the same weights,
tokenizer, chat template and numerical precision. Builds newer than `b9910` are
selected with `VITALS_LLAMA_LIB`. The binding detects their context-parameter
layout and passes model parameters opaquely, so one binding covers both.

## Muse Glimmer 30B

Muse Glimmer 30B (Meta, Apache 2.0) is dense: 52 layers, three of every four
with a 2,048-token sliding window and every fourth attending to the whole
context. Its chat template uses Harmony-style channels: reasoning in a
`to=self` block, each tool call in its own block, and the answer `to=user`.

- **Engine.** Use a llama.cpp master build. The binding disables CUDA graphs
  for it, because capture past about 5,000 positions otherwise faults
  (`VITALS_ALLOW_CUDA_GRAPHS=1` re-enables them).
- **Always send a system message.** Without one, the template writes the
  current date into a system block, so a replay on another day sees a
  different prompt.
- **Reasoning strength** is a template option. Pass
  `{"reasoning_strength": "low"}` as `chat_kwargs` when the agent used one. The
  default is `high`.
- **Record verbatim.** For locally generated output, parse with
  `vitals.server.toolcalls.parse_assistant_output(text, strip=False)` so re-rendering reproduces the
  tokens.
- **Validation.** Capture leaves greedy output unchanged, flash attention on
  and off agree on greedy tokens, a copied prefix past the window reproduces
  the full decode, and replaying the model's own greedy output forces identical
  tokens. Replaying OpenRouter outputs (`meta/muse-glimmer-30b`) agrees with
  97% of tokens on average.

## GLM-5.3-Flash

GLM-5.3-Flash (Z.ai, MIT) has 320B parameters with 18B active per token, 288
experts, and 45 layers plus one multi-token-prediction layer that the graph
skips. Every fourth layer (3, 7, ..., 43) is compressed attention (MLA) with a
top-k index (DSA): each query attends to the 2,048 keys the index selects. The
other 34 layers are Kimi Delta Attention. In the attention layers the softmax
runs over the whole cache with unselected keys set to zero, so a section the
index never selects reads as zero.

- **Engine.** Use llama.cpp master from 2026-09-30 or later (the `glm5-next`
  architecture).
- **GGUF.** Uploads that use the pre-merge architecture name `glm5next` are
  rejected by mainline llama.cpp. For Unsloth's upload, replace shard 1 with
  the file of the same name in the repository's `Shard_Rewrite/` folder, which
  holds only metadata. The UD-IQ3_XXS quant (113 GB) fits on one H200.
- **Template.** It uses jinja `{% break %}` and transformers-style
  `tojson(ensure_ascii=False)`. Both are supported, the second only for
  templates that call it. The template writes `Reasoning Effort: Max` unless
  the request passes `{"reasoning_effort": "low"|"high"}`, so record the effort
  the agent used as `chat_kwargs`. Generation starts inside `<think>`. Tool
  calls use `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`,
  which `vitals.server.toolcalls.parse_tool_calls` reads.
- **Validation.** Only the 11 attention layers produce a softmax, and each row
  attends exactly 2,048 keys on a long prompt. Capture leaves greedy output
  unchanged. A copied prefix reproduces the full decode, including the
  linear-attention state, the latent cache and the index cache.
- **Fidelity at 3 bits.** Replaying the model's own greedy output agrees with
  95% of tokens, against 99.9% for Qwen3.6-27B in bf16. Batched decoding and
  flash attention off (needed for capture) take different numerical paths at
  this precision, and the disagreements sit at uncertain steps. A larger quant
  should narrow the gap.
