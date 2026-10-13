"""OpenAI-style chat messages and tool schemas: the text a message carries and the types its tools declare."""

from __future__ import annotations

import json


def content_text(message: dict, *, tool_calls: bool = False) -> str:
    """The text of a message: its ``content`` string, or the joined ``text`` of its content parts. With
    ``tool_calls`` the JSON of each call's ``function`` follows (assistant tool-call turns often have no content)."""
    content = message.get("content") or ""
    if isinstance(content, list):  # multimodal-style content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    text = str(content)
    if tool_calls and message.get("tool_calls"):
        text += "".join(json.dumps(tc.get("function", {})) for tc in message["tool_calls"])
    return text


def schema_types(tools: list[dict] | None) -> dict[str, dict[str, str]]:
    """{tool_name: {param_name: json_type}} from OpenAI tool definitions."""
    out: dict[str, dict[str, str]] = {}
    for t in tools or []:
        fn = t.get("function", {})
        props = (fn.get("parameters") or {}).get("properties") or {}
        out[fn.get("name", "")] = {k: (v or {}).get("type", "string") for k, v in props.items()}
    return out


def coerce(value: str, json_type: str):
    """A tool argument written as text, parsed as JSON when its schema type is not ``string``; text that does not
    parse is returned unchanged."""
    if json_type == "string" or not json_type:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
