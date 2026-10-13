"""Importing generic OpenAI-style chat logs into a trace store, so any agent's past calls can be replayed and
monitored without changing the agent.

One JSON object per line, in either shape:

- ``{"request": {...chat completion request...}, "response": {...chat completion...}}`` (SDK and proxy logs;
  ``request`` may be wrapped in ``body``);
- ``{"messages": [...], "tools": [...], "response" | "assistant" | "output": ...}`` where the reply is a chat
  completion, an assistant message or a plain string.

Calls are grouped into episodes by an id field when one is present (``episode``, ``session_id``,
``conversation_id``, ``metadata.session_id``, ``metadata.episode``), otherwise by prefix chaining: a call continues
an episode when its messages start with that episode's previous request plus the reply. Each call becomes one turn
(``record_turn``): the request messages and the reply as forced tokens, ready for ``vitals replay``. Re-importing a
log skips the calls already stored (an index next to the episodes keeps their hashes).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from vitals.core.io import atomic_write_json
from vitals.core.messages import content_text

INDEX = "_import_index.json"


def _assistant(reply) -> dict | None:
    """The reply as an assistant message with ``content`` and, when present, ``tool_calls`` and
    ``reasoning_content``."""
    if reply is None:
        return None
    if isinstance(reply, str):
        return {"role": "assistant", "content": reply}
    if "choices" in reply:
        ch = reply.get("choices") or []
        if not ch:
            return None
        reply = ch[0].get("message") or {}
    msg = {"role": "assistant", "content": reply.get("content") or ""}
    if reply.get("tool_calls"):
        msg["tool_calls"] = reply["tool_calls"]
    reasoning = reply.get("reasoning_content") or reply.get("reasoning")
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def normalize(rec: dict) -> dict | None:
    """One log line as ``{messages, tools, assistant, chat_kwargs, hint, model}``, or None when it holds no
    complete call."""
    if "request" in rec:
        req = rec["request"]
        req = req.get("body", req) if isinstance(req, dict) else {}
        reply = rec.get("response")
    else:
        req = rec
        reply = rec.get("response", rec.get("assistant", rec.get("output")))
    messages = req.get("messages")
    assistant = _assistant(reply)
    if not messages or assistant is None:
        return None
    meta = rec.get("metadata") or req.get("metadata") or {}
    hint = rec.get("episode") or rec.get("session_id") or rec.get("conversation_id") or meta.get("session_id") \
        or meta.get("episode")
    kw = req.get("chat_template_kwargs") or (req.get("extra_body") or {}).get("chat_template_kwargs")
    model = req.get("model") or (reply.get("model") if isinstance(reply, dict) else None)
    return {"messages": messages, "tools": req.get("tools") or None, "assistant": assistant, "chat_kwargs": kw or None,
            "hint": str(hint) if hint is not None else None, "model": model}


def _key(m: dict) -> str:
    """A message's identity for prefix matching: role, text and tool calls (ids and reasoning ignored)."""
    calls = [(c.get("function") or {}).get("name", "") + (c.get("function") or {}).get("arguments", "")
             for c in m.get("tool_calls") or []]
    return hashlib.sha1(json.dumps([m.get("role"), content_text(m).strip(), calls]).encode()).hexdigest()[:16]


def call_hash(c: dict) -> str:
    """A call's identity for re-imports: its messages and reply, plus its session id when the log gives one (so
    identical calls of two sessions are both imported)."""
    key = [[_key(m) for m in c["messages"]], _key(c["assistant"])]
    if c.get("hint"):
        key.append(c["hint"])
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80] or "episode"


def group(calls: list[dict], index: dict, prefix: str) -> list[tuple[str, dict]]:
    """Assign each new call an episode id, continuing the episodes in ``index`` (updated in place)."""
    eps = index.setdefault("episodes", {})              # id -> {"turns": n, "tail": [message keys]}
    out = []
    for c in calls:
        keys = [_key(m) for m in c["messages"]]
        if c["hint"]:
            eid = _safe(c["hint"])
        else:
            best, n_best = None, -1
            for eid_, st in eps.items():
                tail = st["tail"]
                if st.get("hinted") or len(tail) > len(keys) or keys[:len(tail)] != tail:
                    continue
                if len(tail) > n_best:
                    best, n_best = eid_, len(tail)
            eid = best or f"{prefix}-{len(eps):04d}"
        st = eps.setdefault(eid, {"turns": 0, "tail": [], "hinted": bool(c["hint"])})
        st["tail"] = keys + [_key(c["assistant"])]
        out.append((eid, c))
    return out


def import_log(log: str | Path, store_dir: str | Path, backend, *, prefix: str = "log") -> dict:
    """Import every new call of ``log`` into ``store_dir``. Returns counts and the episodes touched."""
    from vitals.core.schema import EpisodeManifest
    from vitals.core.store import TraceStore
    from vitals.replay.record import record_turn

    store_dir = Path(store_dir)
    store = TraceStore(store_dir)
    ipath = store_dir / INDEX
    index = json.loads(ipath.read_text()) if ipath.exists() else {"calls": {}, "episodes": {}}
    calls, bad = [], 0
    for line in Path(log).read_text().splitlines():
        if not line.strip():
            continue
        try:
            c = normalize(json.loads(line))
        except (json.JSONDecodeError, AttributeError, TypeError):
            c = None
        if c is None:
            bad += 1
            continue
        h = call_hash(c)
        if h in index["calls"] or (c.get("hint") and call_hash({**c, "hint": None}) in index["calls"]):
            continue                                   # imported before (older indexes hash without the session)
        c["hash"] = h
        calls.append(c)
    touched, errors, notes, n_tok = {}, [], [], 0
    # group on a copy: the index records a call (and advances its episode) only once the call is in the store
    work = json.loads(json.dumps(index))
    for eid, c in group(calls, work, prefix):
        st = index["episodes"].get(eid) or {"turns": 0, "tail": [], "hinted": bool(c["hint"])}
        turn = st["turns"]
        # Logs rarely record chat template options. A reply with reasoning that the default generation prompt
        # cannot continue is retried with thinking on, then recorded without its reasoning (noted in `notes`).
        tries = [(c["assistant"], c["chat_kwargs"])]
        if c["assistant"].get("reasoning_content"):
            if "enable_thinking" not in (c["chat_kwargs"] or {}):
                tries.append((c["assistant"], {**(c["chat_kwargs"] or {}), "enable_thinking": True}))
            tries.append(({k: v for k, v in c["assistant"].items() if k != "reasoning_content"}, c["chat_kwargs"]))
        err = None
        for i, (asst, kw) in enumerate(tries):
            try:
                n_tok += record_turn(store, backend, eid, turn, c["messages"], asst, tools=c["tools"], chat_kwargs=kw)
                if i:
                    notes.append({"episode": eid, "turn": turn, "note": "recorded with enable_thinking=True" if asst is
                                  c["assistant"] else "recorded without the reply's reasoning (template could not place it)"})
                err = None
                break
            except ValueError as e:                    # the reply is not a continuation of the rendered prompt
                err = str(e)[:300]
        if err is not None:
            errors.append({"episode": eid, "turn": turn, "error": err})
            continue
        if not (store_dir / eid / "manifest.json").exists():
            store.write_manifest(EpisodeManifest(
                episode_id=eid, created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                meta={"source": "import", "log": str(Path(log).resolve()), "logged_model": c["model"]}))
        index["calls"][c["hash"]] = [eid, turn]
        st["turns"] = turn + 1
        st["tail"] = [_key(m) for m in c["messages"]] + [_key(c["assistant"])]
        index["episodes"][eid] = st
        touched[eid] = touched.get(eid, 0) + 1
        atomic_write_json(ipath, index)            # after every turn: an interrupted import resumes, never repeats
    atomic_write_json(ipath, index)
    return {"imported": sum(touched.values()), "forced_tokens": n_tok, "skipped_invalid": bad, "errors": errors,
            "notes": notes, "episodes": touched}
