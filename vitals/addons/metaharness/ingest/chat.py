"""Chat-log ingestion helpers: the reference model's chat template and tool spans."""

from __future__ import annotations

import json

from vitals.engine.template import ChatTemplate

MIN_SPAN = 24


def tool_spans(tpl: ChatTemplate, system: str, tools: list[dict]) -> list[tuple[str, str]]:
    """One span per tool: the JSON line the template renders inside <tools>."""
    p = tpl.render([{"role": "system", "content": system}], tools=tools, add_generation_prompt=True)
    i, j = p.find("<tools>"), p.find("</tools>")
    if i < 0 or j < 0:
        return []
    out = []
    for line in p[i + len("<tools>"):j].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            name = (d.get("function") or {}).get("name") or d.get("name")
        except json.JSONDecodeError:
            name = None
        out.append((f"tool:{name or len(out)}", line))
    return out
