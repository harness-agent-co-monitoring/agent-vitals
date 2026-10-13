"""Default sections for a chat request whose harness declares none, with names that stay stable across turns
and episodes.

- **System prompt:** one section per Markdown heading block, ``system/<heading path>`` (text before the first
  heading is ``system/preamble``). A long prompt without headings is split by paragraph, a short one is
  ``system``.
- **Tool schemas:** ``tool:<name>``, the line the template renders for each tool inside ``<tools>``, else the
  tool's description text.
- **Conversation:** ``user:<k>`` and ``self:<k>`` (earlier assistant messages) numbered from 1, and
  ``tool_result:<tool>#<k>``, the k-th result of that tool, named through the tool call it answers. Tool
  output that a flat-history harness sends as a user message is ``tool_result:output#<k>``.

Every message section is pinned to its message (``msg_index``), so short texts such as a user's "yes" resolve
exactly. Declared sections take precedence: a message that already holds a declared section, or a tool list
with declared ``tool:`` sections, gets no automatic ones. Each automatic section carries ``"auto": True``.
"""

from __future__ import annotations

import json
from typing import Any

from vitals.core.annotate import sections_from_markdown
from vitals.core.messages import content_text
from vitals.core.schema import RegionKind
from vitals.core.segmentation import segment_messages

PARAGRAPH_SPLIT_CHARS = 1200


def _text(message: dict) -> str:
    return content_text(message).strip()


def system_spans(text: str, msg_index: int) -> list[dict]:
    secs = sections_from_markdown(text, by="heading")
    if len(secs) <= 1 and len(text) > PARAGRAPH_SPLIT_CHARS:
        secs = sections_from_markdown(text, by="paragraph")
    if len(secs) <= 1:
        return [{"name": "system", "text": text.strip(), "msg_index": msg_index, "auto": True}] if text.strip() else []
    return [{"name": f"system/{s['name']}", "text": s["text"], "msg_index": msg_index, "auto": True} for s in secs]


def tool_schema_spans(prompt_text: str, tools: list[dict] | None) -> list[dict]:
    """One section per tool schema as rendered in ``prompt_text``."""
    if not tools:
        return []
    out = []
    i, j = prompt_text.find("<tools>"), prompt_text.find("</tools>")
    if 0 <= i < j:
        for line in prompt_text[i + len("<tools>"):j].splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                name = (d.get("function") or {}).get("name") or d.get("name")
            except json.JSONDecodeError:
                name = None
            if name:
                out.append({"name": f"tool:{name}", "text": line, "auto": True})
    if out:
        return out
    for t in tools:
        fn = t.get("function") or t
        desc, name = str(fn.get("description") or "").strip(), fn.get("name")
        if name and len(desc) >= 16 and desc in prompt_text:
            out.append({"name": f"tool:{name}", "text": desc, "auto": True})
    return out


def _covered(messages: list[dict], declared: list[dict]) -> tuple[set[int], bool]:
    """Messages that hold a declared section, and whether tool schemas are declared."""
    msgs, tools = set(), False
    for s in declared:
        if str(s.get("name", "")).startswith("tool:"):
            tools = True
        if s.get("msg_index") is not None:
            msgs.add(int(s["msg_index"]))
            continue
        text = str(s.get("text") or "")
        for i, m in enumerate(messages):
            if text and text in content_text(m, tool_calls=True):
                msgs.add(i)
                break
    return msgs, tools


def auto_spans(messages: list[dict], tools: list[dict] | None = None, prompt_text: str = "",
               declared: list[dict] | None = None) -> list[dict]:
    """Automatic sections for ``messages`` (and ``tools`` as rendered in ``prompt_text``), leaving out the
    parts ``declared`` sections already cover."""
    covered, tools_declared = _covered(messages, declared or [])
    kinds = [r.kind for r in segment_messages(messages)]
    names: dict[str, str] = {}                          # tool_call_id -> tool name
    pending: list[str] = []                             # tool names of the latest calls without ids
    counts: dict[str, int] = {}
    out: list[dict] = []

    def number(key: str) -> int:
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    for i, m in enumerate(messages):
        role = str(m.get("role", "user"))
        if role == "assistant":
            pending = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or tc
                name = str(fn.get("name") or "tool")
                if tc.get("id"):
                    names[str(tc["id"])] = name
                else:
                    pending.append(name)
        text = _text(m)
        if not text or i in covered:
            if role in ("tool", "function") and not m.get("tool_call_id") and pending:
                pending.pop(0)
            continue
        kind = kinds[i] if i < len(kinds) else None
        if role in ("system", "developer"):
            out += system_spans(text, i)
        elif role in ("tool", "function"):
            tool = names.get(str(m.get("tool_call_id") or "")) or (pending.pop(0) if pending else m.get("name") or "tool")
            out.append({"name": f"tool_result:{tool}#{number('tr:' + tool)}", "text": text, "msg_index": i, "auto": True})
        elif role == "assistant":
            out.append({"name": f"self:{number('self')}", "text": text, "msg_index": i, "auto": True})
        elif kind is RegionKind.OBSERVATION:
            out.append({"name": f"tool_result:output#{number('tr:output')}", "text": text, "msg_index": i, "auto": True})
        else:
            out.append({"name": f"user:{number('user')}", "text": text, "msg_index": i, "auto": True})
    if not tools_declared:
        out += tool_schema_spans(prompt_text, tools)
    return out


def with_auto_spans(messages: list[dict], tools: list[dict] | None, prompt_text: str,
                    spans: list[dict] | None) -> list[dict[str, Any]]:
    """Declared spans followed by the automatic ones they leave uncovered."""
    declared = list(spans or [])
    return declared + auto_spans(messages, tools, prompt_text, declared)
