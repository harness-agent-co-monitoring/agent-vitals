# Linear-attention layers and write shares

Hybrid models do most of their token mixing in linear-attention layers: 48 of
64 layers in Qwen3.6-27B (Gated DeltaNet) and 34 of 45 in GLM-5.3-Flash (Kimi
Delta Attention). These layers keep no token cache, only a fixed-size memory
that fades and is overwritten, so attention lift does not apply to them.

## Linear reading

Add `--linear-reading` to a replay to measure these layers per section.

```bash
vitals replay --gguf /path/to/model.gguf --store runs/traces --linear-reading --channels logits
```

| Measure | Question | Definition (energies summed over heads and linear layers) |
|---|---|---|
| `retained` | Is the section still in memory when the turn starts generating? | sqrt(energy now / energy right after its last token); 1 means all kept |
| `mem_share` | How much of the memory is that section? | per layer <S_r, S> / \|S\|^2, averaged over layers; sums to 1 |
| `read` | How much of the layers' output comes from it at the generated tokens? | per layer <o_r, o> / \|o\|^2, averaged over layers and steps; sums to 1 |
| `retrieval` | Does the agent pull out what is left? | read / mem_share, when mem_share >= 0.001 |

`retrieval` plays the role of lift with the right baseline. An attention layer
keeps every token, so a section's share of memory is its share of the length.
A linear layer keeps whatever survived. For an attention layer the two
measures coincide.

- **Exact.** A section's part of each memory comes from running the layer's
  own update with the values of every other section set to zero. Fading and
  erasing depend only on the token, so the parts add back to the engine's
  output (within about 1e-5 relative error). Pass
  `linear_opts={"check": True}` to `replay_store` to verify it on your data.
- **Sections** partition the tokens. A declared span takes precedence over the
  message it sits in, and the forced output is `output`. Without
  `--spans-file`, the store's saved declarations are reused when the prompt
  still matches. Each section takes one memory slot per linear layer
  (`--linear-max-slots`, default 64). Beyond that, the least recently written
  message regions fold into `region:older`, then spans into `older sections`.
  `--linear-group-sep /` groups spans by their name prefix.
- **Cost.** About 2 ms per replayed token on Qwen3.6-27B with the exactness
  check. Section memories take `2 x slots x layers x heads x d_k x d_v x 4`
  bytes (19 GB for Qwen3.6-27B at 64 slots). For GLM-5.3-Flash on one H200,
  use about 32 slots.
- **Outputs.** `linear_reading.parquet` holds the per-turn measures and
  `linear_steps.parquet` each forced token's per-section share of the linear
  layers' output. `vitals report` adds `lin_retained`, `lin_mem_share`,
  `lin_read` and `lin_retrieval` to each declared section.

## Write shares across every layer type

`vitals.probes.write_share.WriteShareProbe` measures one turn: for every
token-mixing layer, the share of what the layer writes into the hidden state
that comes from each section. It covers attention (including sliding windows),
compressed MLA attention with sparse key selection, and linear attention. Each
layer's vector before its output projection W splits exactly into section
parts, so the share is <W x_s, W x> / |W x|^2, which sums to 1.

```python
from vitals.probes.write_share import WriteShareProbe
from vitals.probes.linear_reading import section_labels

probe = WriteShareProbe(backend, gguf_path)  # backend with diagnostic (flash-attention-off) capture
rows = probe.measure(prompt_tokens, output_tokens,
                     section_labels(ctx.regions, len(prompt_tokens), len(output_tokens)))
# per section: write_all, write_attention, write_linear, weight (raw attention), per_layer, check errors
```

The parts add up to the engine's own tensor before W: within 1e-5 for linear
layers, and within kernel numerics (0.2 to 0.7%) for recomputed attention.

Both probes need PyTorch (`pip install 'agent-vitals[gpu]'`).
