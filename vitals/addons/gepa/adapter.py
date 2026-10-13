"""Per-component reading diagnostics as GEPA ``optimize_anything`` side information.

``GepaVitals.wrap(evaluator)`` returns an evaluator with the same signature.
Inside it, the user's evaluator calls ``GepaVitals.record(messages, response)``
after each model call. When the evaluator returns, the recorded calls are
teacher-forced through the open-weight reference model, and every candidate
component that appears verbatim in the prompt becomes a declared section.

Each component then receives its reading under
``side_info["<component>_specific_info"]["Reading (agent-vitals)"]``. GEPA merges
that dict into the reflection record when it reflects on that component. A
compact map of all measured sections goes under ``side_info["Read map
(agent-vitals)"]``. Nothing is added to ``side_info["scores"]``: the readings
are diagnostics for the reflection LM, not optimization targets.

Replays share one GPU engine and run one at a time under a lock. GEPA's
evaluations run in a thread pool, so the model calls still overlap.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vitals.replay import record_turn

FIELD = "Reading (agent-vitals)"
MAP_FIELD = "Read map (agent-vitals)"
STR_CANDIDATE_KEY = "current_candidate"   # GEPA's component name for a plain-string candidate

_BACKGROUND_TEMPLATE = """Reading diagnostics (agent-vitals). Each evaluation also reports how the agent read each named \
component of the candidate. The logged model call is replayed through an open-weight reference model, which records \
where each generated token's attention went and how uncertain the next-token prediction was.
- read lift: the attention a component received per token, relative to an even spread over the whole prompt. \
1 means as much as its length predicts, 10 means ten times that, 0.2 means a fifth.
- peak lift and hottest line: the same measure for the component's 8 most-read content tokens, and the line holding \
them. A peak far above the read lift means the agent reads one part of a long component.
{consult}
These are measurements of where the agent looked, not scores. Close reading does not mean a component helps: a \
component can be read closely and still mislead. Use them together with the outputs and scores, for example to find \
text the agent does not read, to see which part of a long component it uses, or to check where an edit lands."""

CONSULT_TEXT = {
    "section": "- consult ratio and consult contrast: whether the agent turns toward this component (above 1) or away "
               "from it (below 1) when it is uncertain, relative to how it spreads attention over the rest of the prompt "
               "at the same steps. They are separated from the agent's overall shift toward or away from the prompt. "
               "The contrast compares the hardest and easiest tenth of the generated tokens.",
    "raw": "- consult ratio and consult contrast: whether the component is read more when the agent is uncertain "
           "(above 1) or when it is confident (below 1). The contrast compares the hardest and easiest tenth of the "
           "generated tokens.",
}
BACKGROUND = _BACKGROUND_TEMPLATE.format(consult=CONSULT_TEXT["section"])   # the default GepaVitals.background


@dataclass
class Call:
    """One recorded model call: the request as sent and what the model emitted."""

    messages: list[dict]
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tools: list[dict] | None = None
    chat_kwargs: dict | None = None
    sections: list[dict] = field(default_factory=list)


_CALLS: contextvars.ContextVar[list[Call] | None] = contextvars.ContextVar("vitals_gepa_calls", default=None)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def message_fields(response: Any) -> tuple[str, str, list[dict]]:
    """(content, reasoning, tool_calls) of an OpenAI-style chat completion, object or dict.

    Reasoning is read from ``reasoning_content`` (vLLM, SGLang, llama.cpp
    server) or ``reasoning`` (OpenRouter). Tool calls are normalized to the
    OpenAI dict form with JSON-string arguments.
    """
    choices = _get(response, "choices") or []
    if not choices:
        raise ValueError("the response has no choices")
    msg = _get(choices[0], "message") or {}
    content = _get(msg, "content") or ""
    reasoning = _get(msg, "reasoning_content") or _get(msg, "reasoning") or ""
    calls = []
    for tc in _get(msg, "tool_calls") or []:
        fn = _get(tc, "function") or {}
        args = _get(fn, "arguments") or ""
        calls.append({"id": _get(tc, "id") or f"call_{len(calls)}", "type": "function",
                      "function": {"name": _get(fn, "name") or "",
                                   "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
    return str(content), str(reasoning), calls


def _geomean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and v > 0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def aggregate(rows: list[dict], consult: str = "raw") -> dict[str, dict]:
    """Per section over the turns of one evaluation: mean read lift, the largest peak and its
    line, geometric means of the consult measures.
    ``consult="section"`` reports the section-specific factor of each consult measure (the
    measure divided by the same measure on the whole prompt), read from ``row["diagnostics"]``."""
    if consult not in CONSULT_TEXT:
        raise ValueError(f"consult must be one of {sorted(CONSULT_TEXT)}")

    def cv(r: dict, key: str):
        return (r.get("diagnostics") or {}).get(f"relative_{key}") if consult == "section" else r.get(key)

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["section"], []).append(r)
    out = {}
    for name, rs in by.items():
        lifts = [r["lift"] for r in rs if r.get("lift") is not None]
        peaks = [(r["peak"], r.get("hot_line")) for r in rs if r.get("peak") is not None]
        top = max(peaks, key=lambda p: p[0]) if peaks else (None, None)
        out[name] = {
            "tokens": max(int(r.get("tokens") or 0) for r in rs),
            "turns": len(rs),
            "read_lift": sum(lifts) / len(lifts) if lifts else None,
            "peak_lift": top[0],
            "hot_line": top[1],
            "consult_ratio": _geomean([cv(r, "consult_ratio") for r in rs]),
            "consult_contrast": _geomean([cv(r, "consult_contrast") for r in rs]),
        }
    return out


def describe(m: dict) -> str:
    """One line of reading measures for the reflection LM."""
    parts = []
    if m.get("read_lift") is not None:
        parts.append(f"read lift {m['read_lift']:.2f}")
    if m.get("peak_lift") is not None:
        line = (m.get("hot_line") or "").strip().replace("\n", " ")
        line = line if len(line) <= 80 else line[:77] + "..."
        parts.append(f"peak lift {m['peak_lift']:.1f}" + (f" (hottest line: {line!r})" if line else ""))
    if m.get("consult_ratio") is not None:
        parts.append(f"consult ratio {m['consult_ratio']:.2f}")
    if m.get("consult_contrast") is not None:
        parts.append(f"consult contrast {m['consult_contrast']:.2f}")
    parts.append(f"{m.get('tokens', 0)} tokens" + (f", {m['turns']} model calls" if m.get("turns", 1) > 1 else ""))
    return "; ".join(parts)


NOT_FOUND = ("not found verbatim in the rendered prompt, so its reading was not measured "
             "(if the component is a template, declare the filled-in text with record(..., sections=[...]))")


class GepaVitals:
    """Adds agent-vitals reading diagnostics to a GEPA ``optimize_anything`` evaluator.

    Args:
        reference_gguf: GGUF of the open-weight reference model. It should be the model the
            evaluator calls, or share its tokenizer and chat template.
        out_dir: Directory for the replay store and ``evaluations.jsonl`` (one record per
            measured evaluation with all section measures).
        n_ctx: Context of the replay engine. Replay uses two sequences, so each prompt plus
            output must fit in ``n_ctx // 2`` tokens.
        sample: Fraction of evaluations to measure (1.0 = all).
        components: Optional subset of candidate components to declare as sections.
        chat_kwargs: Default chat template variables for every recorded call
            (e.g. ``{"enable_thinking": False}``). ``record(..., chat_kwargs=...)`` overrides it.
        min_match: Below this top-1 agreement between the reference model and the logged
            output, the read map carries a warning.
        backend: An existing ``LlamaCppBackend`` built with ``diagnostic_attention=True`` and
            ``n_seq_max=2`` (otherwise one is loaded on the first measured evaluation).
        consult: ``"section"`` (default) reports the section-specific part of the consult measures,
            which removes the agent's overall shift toward or away from the prompt under
            uncertainty. ``"raw"`` reports the measures as defined in the main README.
    """

    def __init__(self, reference_gguf: str | Path | None, out_dir: str | Path, *, n_ctx: int = 65536,
                 sample: float = 1.0, seed: int = 0, components: list[str] | None = None,
                 chat_kwargs: dict | None = None, min_match: float = 0.8, backend=None,
                 consult: str = "section"):
        if not 0.0 <= sample <= 1.0:
            raise ValueError("sample must be between 0 and 1")
        if consult not in CONSULT_TEXT:
            raise ValueError(f"consult must be one of {sorted(CONSULT_TEXT)}")
        self.consult = consult
        self.reference_gguf = None if reference_gguf is None else str(reference_gguf)
        self.out = Path(out_dir)
        self.store_dir = self.out / "store"
        self.n_ctx = n_ctx
        self.sample = sample
        self.components = None if components is None else list(components)
        self.chat_kwargs = chat_kwargs
        self.min_match = min_match
        self._backend = backend
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._n = 0
        # episodes of this instance never collide with earlier runs in the same out_dir
        self._tag = time.strftime("%Y%m%d-%H%M%S") + f"-{random.getrandbits(16):04x}"
        self.background = _BACKGROUND_TEMPLATE.format(consult=CONSULT_TEXT[consult])

    # -- recording -------------------------------------------------------------------------

    def record(self, messages: list[dict], response: Any = None, *, content: str | None = None,
               reasoning: str | None = None, tool_calls: list[dict] | None = None,
               tools: list[dict] | None = None, chat_kwargs: dict | None = None,
               sections: list[dict] | None = None) -> None:
        """Record one model call made inside a wrapped evaluator.

        Pass the messages exactly as sent and either the provider ``response`` (an
        OpenAI-style chat completion, object or dict) or ``content`` / ``reasoning`` /
        ``tool_calls``. Explicit fields override the response. Return the reasoning when the
        provider exposes it: replay forces exactly the recorded text, so a missing reasoning
        block changes what is measured. ``sections`` adds named spans ``{"name", "text"}``
        beyond the candidate components, for example the task input.
        """
        calls = _CALLS.get()
        if calls is None:
            raise RuntimeError("GepaVitals.record() was called outside an evaluator wrapped with GepaVitals.wrap()")
        c, r, t = message_fields(response) if response is not None else ("", "", [])
        calls.append(Call(
            messages=json.loads(json.dumps(messages, ensure_ascii=False)),
            content=c if content is None else str(content),
            reasoning=r if reasoning is None else str(reasoning),
            tool_calls=t if tool_calls is None else list(tool_calls),
            tools=tools,
            chat_kwargs=self.chat_kwargs if chat_kwargs is None else chat_kwargs,
            sections=[{"name": str(s["name"]), "text": str(s["text"])} for s in (sections or [])],
        ))

    # -- evaluator wrapper -----------------------------------------------------------------

    def wrap(self, evaluator: Callable[..., Any]) -> Callable[..., tuple[float, dict]]:
        """An evaluator with the same signature that adds reading diagnostics to its side info."""

        @functools.wraps(evaluator)   # GEPA reads the signature through __wrapped__
        def wrapped(candidate, *args, **kwargs):
            calls: list[Call] = []
            token = _CALLS.set(calls)
            try:
                result = evaluator(candidate, *args, **kwargs)
            finally:
                _CALLS.reset(token)
            if isinstance(result, tuple) and len(result) == 2:
                score, info = result
            else:
                score, info = result, {}
            info = dict(info) if isinstance(info, dict) else ({} if info is None else {"Info": info})
            if calls and self._take_sample():
                comps = {STR_CANDIDATE_KEY: candidate} if isinstance(candidate, str) else dict(candidate)
                if self.components is not None:
                    comps = {k: v for k, v in comps.items() if k in self.components}
                try:
                    self.annotate(info, comps, self.measure(comps, calls))
                except Exception as exc:  # never let a diagnostic failure break optimization
                    info[MAP_FIELD] = f"unavailable ({type(exc).__name__}: {exc})"
            return score, info

        return wrapped

    def _take_sample(self) -> bool:
        if self.sample >= 1.0:
            return True
        with self._lock:
            return self._rng.random() < self.sample

    # -- measurement -----------------------------------------------------------------------

    def _ensure_backend(self):
        if self._backend is None:
            if self.reference_gguf is None:
                raise ValueError("no reference model: pass reference_gguf or backend")
            from vitals.backends.llamacpp import LlamaCppBackend

            self._backend = LlamaCppBackend(self.reference_gguf, n_ctx=self.n_ctx, n_seq_max=2,
                                            diagnostic_attention=True)
        return self._backend

    def measure(self, components: dict[str, str], calls: list[Call]) -> dict:
        """Replay ``calls`` with the components (and any recorded extra sections) declared,
        and return {"episode", "sections": {name: measures}, "declared", "missing", "match"}."""
        # Match on the text without surrounding whitespace: chat templates commonly trim message
        # content, so a section that ends a message with "\r\n" or spaces would otherwise not be found.
        spans: list[dict] = [{"name": n, "text": t.strip()} for n, t in components.items()
                             if isinstance(t, str) and t.strip()]
        names = {s["name"] for s in spans}
        for c in calls:
            for s in c.sections:
                if s["name"] not in names and s["text"].strip():
                    spans.append({"name": s["name"], "text": s["text"].strip()})
                    names.add(s["name"])
        key = hashlib.sha256(json.dumps([spans, [asdict(c) for c in calls]], sort_keys=True,
                                        ensure_ascii=False).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        from vitals.analysis.readmap import episode_readmap
        from vitals.core.store import TraceStore
        from vitals.replay import replay_episode

        with self._lock:
            be = self._ensure_backend()
            eid = f"ep-{self._tag}-{self._n:06d}"
            self._n += 1
            store = TraceStore(self.store_dir)
            n_forced = 0
            for turn, c in enumerate(calls):
                assistant: dict[str, Any] = {"role": "assistant", "content": c.content}
                if c.tool_calls:
                    assistant["tool_calls"] = c.tool_calls
                if c.reasoning:
                    assistant["reasoning_content"] = c.reasoning
                n_forced += record_turn(store, be, eid, turn, c.messages, assistant,
                                        tools=c.tools, chat_kwargs=c.chat_kwargs)
            if not n_forced:
                raise ValueError("the recorded calls contain no output to replay")
            t0 = time.time()
            replay_episode(be, self.store_dir, eid, spans=spans, channels=("attention", "logits"),
                           token_attn=True, kv_reuse=True)
            secs = time.time() - t0
            rows = episode_readmap(self.store_dir / eid, diagnostics=self.consult == "section")
            match = _top1_match(self.store_dir / eid)
            sections = aggregate(rows, consult=self.consult)
            out = {"episode": eid, "sections": sections, "declared": [s["name"] for s in spans],
                   "missing": [s["name"] for s in spans if s["name"] not in sections],
                   "match": match, "calls": len(calls), "replay_secs": round(secs, 2)}
            self.out.mkdir(parents=True, exist_ok=True)
            with (self.out / "evaluations.jsonl").open("a") as f:
                f.write(json.dumps({**out, "components": sorted(components)}, ensure_ascii=False) + "\n")
            self._cache[key] = out
        return out

    def annotate(self, info: dict, components: dict[str, str], measured: dict) -> dict:
        """Write the measures into GEPA side info (in place) and return it."""
        sections = measured["sections"]
        for comp in components:
            if comp not in measured["declared"]:
                continue
            text = describe(sections[comp]) if comp in sections else NOT_FOUND
            key = f"{comp}_specific_info"
            specific = dict(info[key]) if isinstance(info.get(key), dict) else {}
            specific[FIELD] = text
            info[key] = specific
        # The map lets the reflection LM compare sections. With one declared section it would only
        # repeat that section's reading line, so it then carries just a warning, if any.
        lines = []
        if len(measured["declared"]) > 1:
            lines = [f"{name}: {describe(sections[name])}" for name in measured["declared"] if name in sections]
            lines += [f"{name}: {NOT_FOUND}" for name in measured["missing"]]
        match = measured.get("match")
        if match is not None and match < self.min_match:
            lines.append(f"warning: the reference model's top choice matches only {match:.0%} of the logged output "
                         "tokens, so these readings may not reflect the model that produced it")
        if lines:
            info[MAP_FIELD] = "\n".join(lines)
        return info

    def close(self) -> None:
        if self._backend is not None and hasattr(self._backend, "close"):
            self._backend.close()
        self._backend = None


def _top1_match(ep: Path) -> float | None:
    """Share of forced tokens that were also the reference model's top choice."""
    import pandas as pd

    parts = sorted(Path(ep).glob("tokens-*.parquet"))
    if not parts:
        return None
    tk = pd.concat([pd.read_parquet(p) for p in parts])
    tk = tk[tk.region_id == "gen"]
    if "top_ids" not in tk or not len(tk):
        return None
    hits = [bool(len(t)) and int(t[0]) == int(tok) for t, tok in zip(tk.top_ids, tk.token_id, strict=True)
            if t is not None]
    return sum(hits) / len(hits) if hits else None
