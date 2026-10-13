"""cb_eval plumbing: CaptureSpec -> node-name filter -> per-step numpy harvest.

Verified properties:
- capture is observation-only and bitwise non-perturbing;
- capture-off engines never register the callback (zero-overhead path);
- attention tensors only exist in diagnostic (fa-off) mode, on full-attention
  layers, and their KV axis is padded to the cache allocation granularity with
  masked columns exactly 0 — `harvest` slices them to the used width.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vitals.core.schema import AttentionMode, CaptureSpec, CaptureUnsupported
from vitals.engine.nodes import ArchNodeMap


@dataclass
class CapturePlan:
    """Resolved node names + copy modes for a (spec, architecture) pair."""

    resid_layers: list[int] = field(default_factory=list)
    attn_layers: list[int] = field(default_factory=list)
    names: set[str] = field(default_factory=set)
    modes: dict[str, str] = field(default_factory=dict)  # name -> "full" | "last_row"


def plan_capture(spec: CaptureSpec, node_map: ArchNodeMap) -> CapturePlan:
    """Translate a CaptureSpec into graph-node names. Raises CaptureUnsupported
    for attention requests on linear-attention layers (model fact, any engine)."""
    plan = CapturePlan()
    for layer in spec.activations.layers:
        if not 0 <= layer < node_map.n_layers:
            raise CaptureUnsupported(
                f"activation layer {layer} out of range for {node_map.arch} ({node_map.n_layers} layers)"
            )
        plan.resid_layers.append(layer)
        plan.names.add(node_map.residual_node(layer))
    if spec.attention.mode is not AttentionMode.NONE:
        # REGION_POOLED needs only the current token's attention row (pooled into
        # attn_share) -> cheap last-row copies. PER_HEAD keeps full matrices.
        attn_mode = "last_row" if spec.attention.mode is AttentionMode.REGION_POOLED else "full"
        layers = spec.attention.layers or list(node_map.full_attn_layers)
        for layer in layers:
            if layer not in node_map.full_attn_layers:
                raise CaptureUnsupported(
                    f"attention layer {layer} is a linear-attention layer on {node_map.arch}; "
                    f"full-attention layers: {list(node_map.full_attn_layers)}"
                )
            plan.attn_layers.append(layer)
            plan.names.add(node_map.attn_node(layer))
            plan.modes[node_map.attn_node(layer)] = attn_mode
    return plan


class CaptureSession:
    """Per-episode harvest of an engine's `captured` dict into StepOutput fields."""

    def __init__(self, plan: CapturePlan, node_map: ArchNodeMap):
        self.plan = plan
        self.node_map = node_map

    def harvest(self, engine) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Collect (resid, attn) captured by the engine's last decode call.
        Multi-ubatch/chunk decodes arrive concatenated on the token axis.
        Attention is sliced to the used KV width."""
        resid: dict[int, np.ndarray] = {}
        attn: dict[int, np.ndarray] = {}
        cap = engine.collect_captured()
        for layer in self.plan.resid_layers:
            a = cap.get(self.node_map.residual_node(layer))
            if a is not None:
                resid[layer] = a
        # engine.n_past was already advanced past the decoded token(s), so it equals
        # the number of KV cells in use — the correct un-padded width.
        n_kv = engine.n_past
        for layer in self.plan.attn_layers:
            a = cap.get(self.node_map.attn_node(layer))
            if a is not None:
                attn[layer] = a[..., :n_kv]
        return resid, attn


def pool_attention(
    attn: dict[int, np.ndarray], regions, n_kv: int
) -> list[float] | None:
    """Pool the current token's attention over region token spans.

    Input arrays are `harvest` outputs: (heads, T_q, n_kv) for prefill graphs or
    (heads, n_kv) for single-token steps — only the LAST query row (the token
    whose logits we are about to read) is pooled. Averaged over heads and layers.

    Returns one share per region (in region order) plus a final bucket for
    generated-so-far tokens (positions past the last region span). Shares sum
    to ~1. None when spans are unresolved or no attention was captured.
    """
    if not attn or not regions:
        return None
    spans = [(r.token_start, r.token_end) for r in regions]
    if any(s is None or e is None for s, e in spans):
        return None
    rows = []
    for a in attn.values():
        a = np.asarray(a, dtype=np.float32)
        row = a[:, -1, :] if a.ndim == 3 else a  # (heads, n_kv)
        rows.append(row.mean(axis=0))
    vec = np.stack(rows).mean(axis=0)[:n_kv]  # (n_kv,) averaged over layers+heads
    shares = [float(vec[s:e].sum()) for s, e in spans]
    gen_start = max(e for _, e in spans)
    shares.append(float(vec[gen_start:n_kv].sum()) if gen_start < n_kv else 0.0)
    return shares
