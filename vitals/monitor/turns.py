"""Turn records: what an agent turn did, and with what evidence, from a stored episode.

Per turn:
- **action**: the tool calls (name and parsed arguments) or a message, and the onset step where the action starts
  (the first tool-call marker after the reasoning, else the first message token);
- **read at onset**: the sections the model attended most at that step;
- **per argument**: its value, the exact value tokens (located inside the call for the XML-parameter, GLM, JSON and
  channel formats of ``vitals.server.toolcalls``), their confidence (minimum and mean log-probability, mean and
  maximum entropy, top-1 minus top-2 margin at the first value token; for a list or object value, over the tokens of
  its leaf values only, since punctuation and keys are formatting), the sections whose text contains the value
  (innermost section per occurrence), and the sections attended while the value was written.

Per-token attention comes from ``attn_sparse.parquet`` (top attended prompt positions per generated token, written
by replay), else from the per-token section shares when they align with the stored regions. Token text comes from
the model's GGUF tokenizer metadata (``vitals.monitor.vocab``), so no model is loaded.
"""

from __future__ import annotations

import bisect
import json
import re
from pathlib import Path

import numpy as np

from vitals.core.messages import coerce, schema_types
from vitals.core.sections import section_labels
from vitals.core.store import read_regions

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FN = re.compile(r"<function=([\w.\-]+)>")
_PARAM = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", re.S)
_GLM_NAME = re.compile(r"\s*([\w.\-]+)\s*<arg_key>")
_GLM_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
_ATEM_INVOKE = re.compile(r'<atem:invoke name="([\w.\-]+)">(.*?)</atem:invoke>', re.S)
_ATEM_PARAM = re.compile(r'<atem:parameter name="([\w.\-]+)">(.*?)</atem:parameter>', re.S)
_MARKERS = ("<tool_call>", "<atem:function_calls>")
MIN_LEAF = 3


def locate_calls(text: str, tools: list[dict] | None = None) -> list[dict]:
    """Tool calls in generated ``text`` with each argument's value and character span [start, end) in ``text``."""
    types = schema_types(tools)
    calls = []

    def add(name, start, end, items):
        ptypes = types.get(name, {})
        args = [{"name": k, "value": coerce(v, ptypes.get(k, "string")) if isinstance(v, str) else v, "span": sp}
                for k, v, sp in items]
        calls.append({"name": name, "span": (start, end), "args": args})

    for b in _BLOCK.finditer(text):
        inner, off = b.group(1), b.start(1)
        if m := _FN.search(inner):
            add(m.group(1), b.start(), b.end(), [(p.group(1), p.group(2), (off + p.start(2), off + p.end(2)))
                                                  for p in _PARAM.finditer(inner)])
        elif m := _GLM_NAME.match(inner):
            add(m.group(1), b.start(), b.end(), [(a.group(1).strip(), a.group(2), (off + a.start(2), off + a.end(2)))
                                                  for a in _GLM_ARG.finditer(inner)])
        else:
            try:
                obj = json.loads(inner)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            args = obj.get("arguments", {})
            items = []
            if isinstance(args, dict):
                spans = _object_value_spans(inner, inner.find('"arguments"'))
                items = [(k, v, (off + spans[k][0], off + spans[k][1]) if k in spans else None) for k, v in args.items()]
            add(str(obj.get("name")), b.start(), b.end(), items)
    for inv in _ATEM_INVOKE.finditer(text):
        off = inv.start(2)
        add(inv.group(1), inv.start(), inv.end(), [(p.group(1), p.group(2), (off + p.start(2), off + p.end(2)))
                                                    for p in _ATEM_PARAM.finditer(inv.group(2))])
    return sorted(calls, key=lambda c: c["span"][0])


def _object_value_spans(text: str, key_at: int) -> dict[str, tuple[int, int]]:
    """Character spans of the values of the JSON object that follows ``text[key_at:]`` (a ``"key":`` before it),
    keyed by their top-level keys only: a key of a nested object never takes a top-level argument's span."""
    if key_at < 0:
        return {}
    dec = json.JSONDecoder()
    start = text.find("{", key_at)
    if start < 0:
        return {}
    out, i = {}, start + 1
    ws = re.compile(r"[\s,]*")
    try:
        while True:
            i = ws.match(text, i).end()
            if i >= len(text) or text[i] == "}":
                return out
            key, i = dec.raw_decode(text, i)
            i = re.compile(r"\s*:\s*").match(text, i).end()
            _, end = dec.raw_decode(text, i)
            out[str(key)] = (i, end)
            i = end
    except (ValueError, AttributeError):          # malformed after this point: keep what was found
        return out


def leaves(value) -> list[str]:
    """The scalar values inside a (possibly nested) argument value, as text of at least ``MIN_LEAF`` characters."""
    if isinstance(value, dict):
        return [x for v in value.values() for x in leaves(v)]
    if isinstance(value, list):
        return [x for v in value for x in leaves(v)]
    if isinstance(value, bool) or value is None:
        return []
    s = str(value).strip()
    return [s] if len(s) >= MIN_LEAF else []


def _name(region_id: str) -> str:
    return region_id[5:] if region_id.startswith("span:") else f"region:{region_id}"


class _Prompt:
    """One turn's prompt: text with token offsets, the innermost section per token, and section names."""

    def __init__(self, ids, strs, regions, vocab):
        if strs is not None and all(s is not None for s in strs):
            self.offs, pos = [], 0
            for s in strs:
                self.offs.append((pos, pos + len(s)))
                pos += len(s)
            self.text = "".join(strs)
        else:
            self.text, self.offs = vocab.text_and_offsets(ids)
        regs = [type("R", (), {"id": r.get("id"), "token_start": r.get("token_start"), "token_end": r.get("token_end")})
                for r in regions]
        self.labels = section_labels(regs, len(ids), 0)
        self.starts = [a for a, _ in self.offs]

    def sections_containing(self, value) -> list[str]:
        found = []
        for leaf in leaves(value):
            i = self.text.find(leaf)
            while i >= 0:
                tok = max(0, bisect.bisect_right(self.starts, i) - 1)
                name = self.labels[tok]
                if name not in found:
                    found.append(name)
                i = self.text.find(leaf, i + 1)
        return found

    def label_of(self, position: int) -> str | None:
        return self.labels[position] if 0 <= position < len(self.labels) else None


def _read_by_step(ep: Path, turn: int, prompt: _Prompt, gen, regions) -> dict[int, dict[str, float]]:
    """step -> {section: share of the step's prompt attention}."""
    sp = ep / "attn_sparse.parquet"
    out: dict[int, dict[str, float]] = {}
    if sp.exists():
        import pandas as pd

        s = pd.read_parquet(sp, columns=["turn", "step", "pos", "w", "prompt_mass"], filters=[("turn", "==", turn)])
        for r in s.itertuples():
            acc: dict[str, float] = {}
            for p, w in zip(r.pos, r.w, strict=False):
                lab = prompt.label_of(int(p))
                if lab is not None:
                    acc[lab] = acc.get(lab, 0.0) + float(w)
            mass = float(r.prompt_mass) if r.prompt_mass else sum(acc.values()) or 1.0
            out[int(r.step)] = {k: v / mass for k, v in acc.items()}
        return out
    ids = [str(r.get("id")) for r in regions]
    for step, share in zip(gen.step, gen.attn_share, strict=True):
        if share is not None and len(share) == len(ids):
            out[int(step)] = {_name(i): float(v) for i, v in zip(ids, share, strict=True)}
    return out


def _top(shares: dict[str, float], k: int, named_first: bool = True) -> list[tuple[str, float]]:
    named = {n: v for n, v in shares.items() if not n.startswith("region:")}
    pool = named if named_first and named else shares
    return [(n, round(v, 4)) for n, v in sorted(pool.items(), key=lambda kv: -kv[1])[:k]]


def _finite(fn, values) -> float | None:
    """``fn`` over the finite values, None when there are none (a turn stored without log-probabilities)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(fn(v)) if len(v) else None


def _neg(v: float | None) -> float | None:
    return None if v is None else -v


def turn_records(ep_dir: str | Path, vocab, *, turns: list[int] | None = None, top: int = 5) -> list[dict]:
    """Turn records of a stored episode (see the module docstring)."""
    import pandas as pd

    from vitals.core.store import read_token_rows

    ep = Path(ep_dir)
    tk = read_token_rows(ep)
    if tk is None:
        return []
    regions_by_turn = read_regions(ep)
    msgs = {}
    if (ep / "messages.jsonl").exists():
        for line in (ep / "messages.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                msgs[int(r["turn"])] = r
    at = pd.read_parquet(ep / "attn_tokens.parquet") if (ep / "attn_tokens.parquet").exists() else None
    out = []
    for t in sorted(int(x) for x in tk.turn.unique()):
        if turns is not None and t not in turns:
            continue
        gen = tk[(tk.turn == t) & (tk.region_id == "gen")].sort_values("step").reset_index(drop=True)
        if gen.empty:
            continue
        ids = gen.token_id.astype(int).tolist()
        text, offs = vocab.text_and_offsets(ids)
        tools = (msgs.get(t) or {}).get("tools")
        calls = locate_calls(text, tools)
        regions = regions_by_turn.get(t, [])
        prompt = None
        if at is not None:
            a = at[at.turn == t].sort_values("idx")
            if len(a):
                strs = a.token_str.tolist() if "token_str" in a else None
                prompt = _Prompt(a.token_id.astype(int).tolist(), strs, regions, vocab)
        read = _read_by_step(ep, t, prompt, gen, regions) if prompt is not None else {}
        sur = gen.surprisal.to_numpy(dtype=float) if "surprisal" in gen else np.full(len(gen), np.nan)
        ent = gen.ent.to_numpy(dtype=float) if "ent" in gen else np.full(len(gen), np.nan)
        think = text.find("</think>")
        body = think + len("</think>") if think >= 0 else 0
        marks = [i for i in (text.find(m, body) for m in _MARKERS) if i >= 0]
        onset_char = min(marks) if marks else next((i for i in range(body, len(text)) if not text[i].isspace()), None)
        onset = next((k for k, (s, e) in enumerate(offs) if e > onset_char), None) if onset_char is not None else None
        rec = {"turn": t, "action": "tool_call" if calls else "message", "onset_step": onset,
               "read_at_onset": _top(read.get(int(gen.step[onset]), {}), top) if onset is not None else [],
               "calls": []}
        for c in calls:
            args = []
            for a in c["args"]:
                steps = [k for k, (s, e) in enumerate(offs) if a["span"] and e > a["span"][0] and s < a["span"][1] and e > s]
                row = {"name": a["name"], "value": a["value"], "steps": steps,
                       "found_in": prompt.sections_containing(a["value"]) if prompt is not None else None}
                scored = steps
                if steps and isinstance(a["value"], (list, dict)):
                    # a structured value: score the tokens of its leaf values, not its punctuation and keys
                    lo, hi = a["span"]
                    chars = set()
                    for leaf in leaves(a["value"]):
                        i = text.find(leaf, lo, hi)
                        while i >= 0:
                            chars.update(range(i, i + len(leaf)))
                            i = text.find(leaf, i + 1, hi)
                    scored = [k for k in steps if any(c in chars for c in range(*offs[k]))] or steps
                    row["leaf_steps"] = scored
                if steps:
                    row.update(min_lp=_neg(_finite(np.nanmax, sur[scored])), mean_lp=_neg(_finite(np.nanmean, sur[scored])),
                               mean_ent=_finite(np.nanmean, ent[scored]), max_ent=_finite(np.nanmax, ent[scored]))
                    tl = gen.top_lp[scored[0]] if "top_lp" in gen else None
                    row["first_margin"] = float(tl[0] - tl[1]) if tl is not None and len(tl) > 1 else None
                    acc: dict[str, float] = {}
                    for k in steps:
                        for n, v in read.get(int(gen.step[k]), {}).items():
                            acc[n] = acc.get(n, 0.0) + v / len(steps)
                    row["read_while_writing"] = _top(acc, 3)
                args.append(row)
            rec["calls"].append({"name": c["name"], "args": args})
        out.append(rec)
    return out
