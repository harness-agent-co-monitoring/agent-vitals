"""Section perturbations at a fixed logged output prefix.

Each condition rebuilds KV and recurrent state from the perturbed prompt and
teacher-forces the same output token IDs. The result measures conditional
output dependence, not historical relevance or whether an influence helps.
"""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import numpy as np


def _log_softmax(logits) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("expected a finite full-vocabulary logits vector")
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def distribution_js(original_logits, perturbed_logits) -> float:
    """Jensen–Shannon divergence of full output distributions, in nats."""
    lp, lq = _log_softmax(original_logits), _log_softmax(perturbed_logits)
    if lp.shape != lq.shape:
        raise ValueError("output distributions must use the same vocabulary")
    lm = np.logaddexp(lp, lq) - np.log(2.0)
    value = .5 * (np.sum(np.exp(lp) * (lp - lm)) + np.sum(np.exp(lq) * (lq - lm)))
    return max(0.0, float(value))


def _score_prefixes(engine, prompt: list[int], output: list[int], steps: list[int], chunk: int) -> dict:
    # Always rebuild: cached states computed from the original section cannot
    # be reused after replacing it, including the hybrid model's recurrent state.
    engine.reset_context(capture_names=set(), capture_modes={})
    engine.decode(prompt)
    scored, cursor = {}, 0
    batch = min(chunk, int(getattr(engine, "_ctx_opts", {}).get("n_batch", chunk)))
    for step in steps:
        while cursor < step:
            end = min(step, cursor + batch)
            engine.decode(output[cursor:end])
            cursor = end
        scored[step] = np.array(engine.logits(-1), dtype=float, copy=True)
    return scored


def section_sensitivity(backend, prompt_tokens, output_tokens, *, start: int, end: int,
                        replacements: dict[str, list[int]], steps: list[int], chunk: int = 64) -> dict:
    """Compare equal-length token replacements at selected output positions.

    At step t, score the distribution predicting logged token y[t], after
    forcing y[:t]. Replacement token IDs preserve all outside positions.
    Include a no-op replacement and several substantive replacement types
    when interpreting a small experiment. This function changes backend state.
    """
    prompt, output = list(prompt_tokens), list(output_tokens)
    if not prompt or not output or not 0 <= start < end <= len(prompt):
        raise ValueError("need a nonempty prompt/output and a valid section range")
    if not steps or any(not isinstance(t, Integral) or not 0 <= t < len(output) for t in steps):
        raise ValueError("steps must name existing, zero-based generated-token positions")
    steps = sorted(set(int(t) for t in steps))
    if chunk < 1 or not replacements:
        raise ValueError("need a positive chunk size and at least one replacement")
    engine = backend.engine
    for name, replacement in replacements.items():
        if not name or len(replacement) != end - start:
            raise ValueError(f"replacement {name!r} must contain exactly {end - start} token IDs")
    for sequence in [prompt, output, *replacements.values()]:
        if any(not isinstance(t, Integral) or not 0 <= t < engine.n_vocab for t in sequence):
            raise ValueError("token IDs must be integers in the reference model's vocabulary")
    capacity = getattr(engine, "n_ctx_per_seq", 0)
    if capacity and len(prompt) + max(steps) > capacity:
        raise ValueError("prompt and retained output prefix exceed the engine context capacity")
    if hasattr(backend, "_replay_kept"):
        backend._replay_kept = None
    if hasattr(backend, "_kv_tokens"):
        backend._kv_tokens = []
        backend._kv_plan_sig = None
    original = _score_prefixes(engine, prompt, output, steps, chunk)
    variants = []
    for name, replacement in replacements.items():
        replacement = list(replacement)
        perturbed_prompt = prompt[:start] + replacement + prompt[end:]
        perturbed = _score_prefixes(engine, perturbed_prompt, output, steps, chunk)
        values = []
        for step in steps:
            lp = _log_softmax(original[step])
            lq = _log_softmax(perturbed[step])
            values.append({"step": step, "token_id": int(output[step]),
                           "js_nats": distribution_js(original[step], perturbed[step]),
                           "original_logprob": float(lp[output[step]]),
                           "perturbed_logprob": float(lq[output[step]])})
        variants.append({"replacement": name,
                         "changed_tokens": sum(a != b for a, b in zip(prompt[start:end], replacement, strict=True)),
                         "mean_js_nats": float(np.mean([v["js_nats"] for v in values])), "positions": values})
    return {"interpretation": "output dependence conditional on the retained logged prefix",
            "section_start": start, "section_end": end, "prompt_tokens": len(prompt),
            "steps": steps, "variants": variants}


def sensitivity_from_store(backend, store_dir: str | Path, episode: str, *, turn: int, section: str,
                           replacements: dict, steps: list[int]) -> dict:
    """Verify the prompt against stored token IDs before applying a perturbation.

    Replacement values are text or {"token_ids": [...]}. Text is tokenized
    without special tokens and must match the section's token count exactly.
    There is no padding or truncation to hide a length mismatch.
    """
    import pandas as pd

    from vitals.core.store import read_regions

    ep = Path(store_dir) / episode
    records = [json.loads(line) for line in (ep / "messages.jsonl").read_text().splitlines() if line.strip()]
    message = next((r for r in reversed(records) if r["turn"] == turn), None)
    if message is None:
        raise ValueError(f"no logged messages for turn {turn}")
    _, prompt = backend.render_turn(message["messages"], tools=message.get("tools"),
                                    chat_kwargs=message.get("chat_kwargs"))
    path = ep / "attn_tokens.parquet"
    if not path.exists():
        raise ValueError("sensitivity needs attn_tokens.parquet to verify the original prompt token IDs")
    at = pd.read_parquet(path, columns=["turn", "idx", "token_id"])
    at = at[at.turn == turn].sort_values("idx")
    if at.idx.tolist() != list(range(len(prompt))) or at.token_id.tolist() != prompt:
        raise ValueError("rendered prompt differs from stored tokens; check reference weights, tokenizer and template")
    rid = section if section.startswith("span:") else f"span:{section}"
    spans = [r for r in read_regions(ep).get(turn, []) if r.get("id") == rid]
    if len(spans) != 1:
        raise ValueError(f"expected one declared section {rid!r} at turn {turn}")
    start, end = spans[0].get("token_start"), spans[0].get("token_end")
    if start is None or end is None:
        raise ValueError("section token bounds are missing")
    parts = sorted(ep.glob("tokens-*.parquet"))
    if not parts:
        raise ValueError("no logged generated tokens")
    tk = pd.concat([pd.read_parquet(p) for p in parts])
    g = tk[(tk.turn == turn) & (tk.region_id == "gen")].sort_values("step")
    if g.step.tolist() != list(range(len(g))):
        raise ValueError("logged output steps must be contiguous and unique")
    variants = {}
    for name, value in replacements.items():
        if isinstance(value, str):
            variants[name] = backend.engine.tokenize(value, add_special=False, parse_special=False)
        elif isinstance(value, dict) and isinstance(value.get("token_ids"), list):
            variants[name] = value["token_ids"]
        else:
            raise ValueError(f"replacement {name!r} must be text or an object containing token_ids")
    result = section_sensitivity(backend, prompt, g.token_id.tolist(), start=int(start), end=int(end),
                                 replacements=variants, steps=steps)
    return {"episode": episode, "turn": turn, "section": rid[5:], **result}
