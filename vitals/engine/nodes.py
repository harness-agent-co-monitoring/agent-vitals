"""Architecture-aware graph-node naming: which cb_eval tensors mean what.

llama.cpp names graph nodes systematically (`l_out-N`, `kq_soft_max-N` —
verified in the pinned checkout's graph builder). Hybrid architectures have
attention matrices only on their full-attention layers; that is a property of
the model, not the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# arch spellings differ per source (HF config vs GGUF metadata, pre-merge names)
_QWEN35 = ("qwen3_5", "qwen35")
_MUSE = ("muse-glimmer", "muse_glimmer")
_GLM5 = ("glm5-next", "glm5next", "glm5_next")


def _glm5_prefix(meta) -> str:
    """GGUF metadata key prefix: mainline `glm5-next`, or the pre-merge `glm5next` upload."""
    return "glm5-next" if meta("glm5-next.kda.head_dim") is not None else "glm5next"


@dataclass(frozen=True)
class ArchNodeMap:
    arch: str
    n_layers: int
    residual_fmt: str = "l_out-{i}"
    attn_matrix_fmt: str = "kq_soft_max-{i}"  # materializes only with flash-attention off
    full_attn_layers: tuple[int, ...] = field(default_factory=tuple)

    def residual_node(self, layer: int) -> str:
        return self.residual_fmt.format(i=layer)

    def attn_node(self, layer: int) -> str:
        if layer not in self.full_attn_layers:
            raise ValueError(
                f"{self.arch} layer {layer} is a linear-attention layer: no attention matrix exists. "
                f"Full-attention layers: {list(self.full_attn_layers)}"
            )
        return self.attn_matrix_fmt.format(i=layer)


@dataclass(frozen=True)
class LinearMixerSpec:
    """Where a hybrid model's linear-attention layers expose their inputs.

    Gated DeltaNet (Qwen3.5/3.6) and Kimi Delta Attention (GLM-5.3-Flash) keep, per value head, a
    memory S (key x value) updated per token as

        S <- (I - b k k^T) D S + b k v^T,        o = scale * S^T q

    with q, k L2-normalized, b the write strength and D the fade: one factor per head (GDN, ``decay
    == "head"``) or one per key channel (KDA, ``decay == "channel"``). Verified against llama.cpp's
    CUDA kernel (ggml-cuda/gated_delta_net.cu) and its graph builders. Value head h reads key head
    h % n_k_heads. ``g`` is the log fade (D = exp(g)).

    Node names are format strings over the layer index. GDN exposes q|k|v as one fused node
    (``qkv_node``, split by width); KDA exposes them separately. ``out_node`` is the engine's own
    raw output of the update, used only for the exactness check.
    """

    kind: str                       # "gdn" | "kda"
    layers: tuple[int, ...]
    n_k_heads: int
    n_v_heads: int
    d_k: int
    d_v: int
    decay: str                      # "head" | "channel"
    g_node: str
    beta_node: str
    out_node: str
    qkv_node: str | None = None
    q_node: str | None = None
    k_node: str | None = None
    v_node: str | None = None
    norm_eps: float = 1e-6

    def input_nodes(self, layer: int) -> list[str]:
        names = [self.qkv_node] if self.qkv_node else [self.q_node, self.k_node, self.v_node]
        return [n.format(i=layer) for n in (*names, self.g_node, self.beta_node)]

    def node_names(self, check: bool = False) -> set[str]:
        out = {n for i in self.layers for n in self.input_nodes(i)}
        if check:
            out |= {self.out_node.format(i=i) for i in self.layers}
        return out


def linear_spec(engine, node_map: ArchNodeMap) -> LinearMixerSpec | None:
    """The linear-layer spec of a loaded model, or None when it has no linear layers (dense models,
    and Muse Glimmer, whose sliding-window layers are softmax attention). Dimensions come from the
    GGUF metadata, so every size of a verified family works."""
    arch = node_map.arch
    meta = engine.meta
    linear = tuple(i for i in range(node_map.n_layers) if i not in node_map.full_attn_layers)
    if arch in _QWEN35:
        p = "qwen35"
        d = int(meta(f"{p}.ssm.state_size"))
        return LinearMixerSpec(
            kind="gdn", layers=linear, n_k_heads=int(meta(f"{p}.ssm.group_count")),
            n_v_heads=int(meta(f"{p}.ssm.time_step_rank")), d_k=d, d_v=d, decay="head",
            qkv_node="conv_output_silu-{i}", g_node="gate-{i}", beta_node="beta_sigmoid-{i}",
            out_node="attn_output-{i}")
    if arch in _GLM5:
        p = _glm5_prefix(meta)
        d = int(meta(f"{p}.kda.head_dim"))
        h = int(meta(f"{p}.attention.head_count"))
        return LinearMixerSpec(
            kind="kda", layers=linear, n_k_heads=h, n_v_heads=h, d_k=d, d_v=d, decay="channel",
            q_node="kda_q_conv-{i}", k_node="kda_k_conv-{i}", v_node="kda_v_conv-{i}",
            g_node="kda_g1-{i}", beta_node="kda_beta-{i}", out_node="kda_scan_out-{i}")
    return None


@dataclass(frozen=True)
class WriteLayer:
    """How one token-mixing layer's write into the hidden state is split by section.

    kinds:
      "attention"  softmax attention recomputed from the captured post-RoPE queries and keys (exact up to
                   the engine's kernel numerics); a sliding window is a mask (``window``).
      "mla"        compressed attention (GLM-5.3-Flash's DSA layers): the engine's own softmax rows
                   (``rows_node``, flash attention off; non-selected keys are zero) over the cached latent
                   (``v_node``), then the per-head value up-projection ``vb_weight``.
      "linear"     a delta-rule layer, split by vitals.engine.linear_memory, then the per-head RMSNorm
                   (``norm_weight``) and output gate.

    In every case the parts pass the (optional) output gate and then ``out_weight``, which is linear, so
    the section parts of the write add up to the layer's write exactly. ``check_node`` is the engine's own
    tensor right before ``out_weight`` (``check_pre_gate``: before the gate) for the sum check."""

    layer: int
    kind: str
    out_weight: str
    n_head: int
    d_v: int
    q_node: str | None = None
    k_node: str | None = None
    v_node: str | None = None
    rows_node: str | None = None
    n_kv: int = 0
    d_k: int = 0
    window: int | None = None
    vb_weight: str | None = None
    gate_node: str | None = None
    gate_act: str = "none"          # "none" (the node is already the activation), "sigmoid", "silu"
    norm_weight: str | None = None
    norm_eps: float = 1e-6
    check_node: str | None = None
    check_pre_gate: bool = False

    def nodes(self, phase: str, check: bool = False) -> set[str]:
        """Capture nodes for "prompt" positions (keys/values/linear inputs only) or "wanted" positions."""

        def fmt(*names: str | None) -> set[str]:
            return {n.format(i=self.layer) for n in names if n}

        keep = fmt(self.k_node, self.v_node)
        if phase == "wanted":
            keep |= fmt(self.q_node, self.rows_node, self.gate_node)
            if check:
                keep |= fmt(self.check_node)
        return keep


def write_plan(engine, node_map: ArchNodeMap) -> list[WriteLayer]:
    """One WriteLayer per token-mixing layer of a verified family (dimensions from the GGUF metadata)."""
    arch, meta = node_map.arch, engine.meta

    def num(key: str, default: float | None = None) -> float | None:
        v = meta(key)
        return float(v) if v is not None else default

    plan: list[WriteLayer] = []
    if arch in _QWEN35:
        p = "qwen35"
        H, nkv = int(num(f"{p}.attention.head_count")), int(num(f"{p}.attention.head_count_kv"))
        d, eps = int(num(f"{p}.attention.key_length")), num(f"{p}.attention.layer_norm_rms_epsilon", 1e-6)
        nv, dv = int(num(f"{p}.ssm.time_step_rank")), int(num(f"{p}.ssm.state_size"))
        for i in range(node_map.n_layers):
            if i in node_map.full_attn_layers:
                plan.append(WriteLayer(i, "attention", f"blk.{i}.attn_output.weight", H, d, q_node="Qcur-{i}",
                                       k_node="Kcur-{i}", v_node="Vcur-{i}", n_kv=nkv, d_k=d,
                                       gate_node="gate_sigmoid-{i}", check_node="attn_gated-{i}"))
            else:
                plan.append(WriteLayer(i, "linear", f"blk.{i}.ssm_out.weight", nv, dv, gate_node="z-{i}",
                                       gate_act="silu", norm_weight=f"blk.{i}.ssm_norm.weight", norm_eps=eps,
                                       check_node="final_output-{i}"))
        return plan
    if arch in _MUSE:
        p = "muse-glimmer"
        H, nkv = int(num(f"{p}.attention.head_count")), int(num(f"{p}.attention.head_count_kv"))
        d, win = int(num(f"{p}.attention.key_length")), int(num(f"{p}.attention.sliding_window", 0)) or None
        for i in range(node_map.n_layers):
            glob = i in node_map.full_attn_layers
            plan.append(WriteLayer(i, "attention", f"blk.{i}.attn_output.weight", H, d,
                                   q_node="Qcur_normed-{i}" if glob else "Qcur_rope-{i}",
                                   k_node="Kcur_normed-{i}" if glob else "Kcur_rope-{i}", v_node="Vcur-{i}",
                                   n_kv=nkv, d_k=d, window=None if glob else win,
                                   gate_node="attn_gate_sig-{i}", check_node="attn_gated-{i}"))
        return plan
    if arch in _GLM5:
        p = _glm5_prefix(meta)
        H = int(num(f"{p}.attention.head_count"))
        dv_mla = int(num(f"{p}.attention.value_length_mla"))
        dk = int(num(f"{p}.kda.head_dim"))
        eps = num(f"{p}.attention.layer_norm_rms_epsilon", 1e-5)
        for i in range(node_map.n_layers):
            if i in node_map.full_attn_layers:
                plan.append(WriteLayer(i, "mla", f"blk.{i}.attn_output.weight", H, dv_mla, rows_node="kq_soft_max-{i}",
                                       v_node="kv_cmpr-{i}", vb_weight=f"blk.{i}.attn_v_b.weight",
                                       check_node="kqv_out-{i}"))
            else:
                plan.append(WriteLayer(i, "linear", f"blk.{i}.attn_output.weight", H, dk, gate_node="kda_g2-{i}",
                                       gate_act="sigmoid", norm_weight=f"blk.{i}.ssm_norm.weight", norm_eps=eps,
                                       check_node="kda_normed-{i}", check_pre_gate=True))
        return plan
    raise KeyError(f"no write plan for arch {arch!r}")


def _every_4th(n_layers: int) -> tuple[int, ...]:
    """Full attention on layers 3, 7, 11, ... — the [L,L,L,G] hybrid pattern.
    Verified for qwen3_5 (0.8B/24L, 27B/64L configs) and muse-glimmer (52L:
    `set_swa_pattern(4, dense_first=false)` marks il % 4 == 3 as global)."""
    return tuple(i for i in range(n_layers) if i % 4 == 3)


REGISTRY: dict[str, ArchNodeMap] = {
    "qwen3_5:27b": ArchNodeMap(arch="qwen3_5", n_layers=64, full_attn_layers=_every_4th(64)),
    "qwen3_5:9b": ArchNodeMap(arch="qwen3_5", n_layers=32, full_attn_layers=_every_4th(32)),
    "qwen3_5:0.8b": ArchNodeMap(arch="qwen3_5", n_layers=24, full_attn_layers=_every_4th(24)),
}


def graph_layers(engine) -> int:
    """Layers the forward graph runs. Qwen3.5-family files can carry trailing multi-token-prediction layers
    (``<arch>.nextn_predict_layers``, e.g. one in Qwen3.8-27B, whose ``block_count`` is 65). The pinned
    llama.cpp build already leaves them out of ``n_layer``; subtract them only if a build counts them."""
    n = int(engine.n_layer)
    if engine.arch in _QWEN35:
        nextn = int(engine.meta(f"{engine.arch}.nextn_predict_layers") or 0)
        if nextn and n == int(engine.meta(f"{engine.arch}.block_count") or 0):
            n -= nextn
    return n


def for_model(arch: str, n_layers: int) -> ArchNodeMap:
    """Build a node map from model metadata (arch + layer count). Layer counts and
    the hybrid pattern must be re-verified for each new checkpoint."""
    if arch in _QWEN35:
        return ArchNodeMap(arch=arch, n_layers=n_layers, full_attn_layers=_every_4th(n_layers))
    if arch in _MUSE:
        # Same [L,L,L,G] layout and graph-node names as qwen3_5 (verified against
        # llama.cpp's src/models/muse-glimmer.cpp cb() calls). Sliding layers use
        # windowed KV, not linear attention, so an attention matrix DOES exist on
        # every layer — but capture targets the global layers, as for qwen.
        return ArchNodeMap(arch=arch, n_layers=n_layers, full_attn_layers=_every_4th(n_layers))
    if arch in _GLM5:
        # GLM-5.3-Flash: 45 layers (+1 trailing MTP layer the graph skips). Layers with
        # attention.head_count_kv > 0 are the MLA layers with a DSA top-k indexer; the rest are KDA
        # (linear, recurrent). In the released checkpoint that is il % 4 == 3 below the MTP layer
        # (3, 7, ..., 43). The DSA layers' softmax over the full cache (non-selected keys masked
        # out) is `kq_soft_max-{il}` with flash attention off (llama.cpp src/models/glm5-next.cpp).
        return ArchNodeMap(arch=arch, n_layers=n_layers,
                           full_attn_layers=tuple(i for i in _every_4th(n_layers) if i < 45))
    raise KeyError(
        f"no node map for arch {arch!r}; add one to vitals/engine/nodes.py after verifying "
        "its graph-node naming with examples/eval-callback"
    )
