"""Tool-call extraction from generated text.

Two wire formats inside <tool_call>...</tool_call> blocks:

1. JSON (Hermes convention, used by many models):
       <tool_call>
       {"name": "bash", "arguments": {"cmd": "ls"}}
       </tool_call>
2. XML-parameter (Qwen3.5 / Qwen3-Coder templates instruct exactly this) and GLM's
   `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`:
       <tool_call>
       <function=write>
       <parameter=path>
       hello.txt
       </parameter>
       </function>
       </tool_call>

Plus the channel format (Muse Glimmer / Harmony convention): the completion is
a sequence of `<|start|>assistant to=<recipient><|message|>...` segments ended
by <|eom|> (more follows) or <|eot|> (turn over). `to=self` segments carry
reasoning, tool recipients carry <atem:function_calls> blocks:

       <atem:function_calls>
       <atem:invoke name="bash">
       <atem:parameter name="command">df -h</atem:parameter>
       </atem:invoke>
       </atem:function_calls>

Think-block models (Qwen3.x, GLM) put reasoning before ``</think>``; their
generation cue usually opens the block, so the completion starts inside it.
``split_reasoning`` separates it.

All parse into OpenAI-shaped tool_calls. XML/ATEM parameter values are strings
on the wire; when the request's tool schemas are provided, values are coerced
to the declared JSON types. Malformed blocks stay in content rather than
crashing the episode (the harness's retry loop is the right handler).
"""

from __future__ import annotations

import json
import re

from vitals.core.messages import coerce, schema_types

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FN_RE = re.compile(r"^<function=([\w.\-]+)>\s*(.*?)\s*</function>$", re.S)
_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", re.S)
_GLM_FN_RE = re.compile(r"^([\w.\-]+)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)$", re.S)
_GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


_schema_types, _coerce = schema_types, coerce     # older names


def parse_tool_calls(text: str, tools: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Returns (content_without_valid_calls, openai_tool_calls)."""
    calls: list[dict] = []
    types = _schema_types(tools)

    def _parse_inner(inner: str) -> tuple[str, dict] | None:
        m = _FN_RE.match(inner)
        if m:  # XML-parameter convention
            name = m.group(1)
            ptypes = types.get(name, {})
            args = {
                pname: _coerce(pval, ptypes.get(pname, "string"))
                for pname, pval in _PARAM_RE.findall(m.group(2))
            }
            return name, args
        m = _GLM_FN_RE.match(inner)
        if m:  # GLM convention (a JSON object never starts with a bare name)
            name = m.group(1)
            ptypes = types.get(name, {})
            return name, {k.strip(): _coerce(v, ptypes.get(k.strip(), "string")) for k, v in _GLM_ARG_RE.findall(m.group(2))}
        try:  # JSON convention
            obj = json.loads(inner)
            return obj["name"], obj.get("arguments", {})
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _sub(m: re.Match) -> str:
        parsed = _parse_inner(m.group(1))
        if parsed is None:
            return m.group(0)  # malformed: keep verbatim in content
        name, args = parsed
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
        )
        return ""

    content = _TOOL_CALL_RE.sub(_sub, text).strip()
    return content, calls


_CHANNEL_RE = re.compile(
    r"<\|start\|>(\w+)(?:\s+to=([^\s<]+))?<\|message\|>(.*?)(?:<\|eot\|>|<\|eom\|>|$)",
    re.S,
)
_ATEM_BLOCK_RE = re.compile(r"<atem:function_calls>\s*(.*?)\s*</atem:function_calls>", re.S)
_ATEM_INVOKE_RE = re.compile(r'<atem:invoke name="([\w.\-]+)">\s*(.*?)\s*</atem:invoke>', re.S)
_ATEM_PARAM_RE = re.compile(r'<atem:parameter name="([\w.\-]+)">(.*?)</atem:parameter>', re.S)
_CHANNEL_MARKERS = ("<|message|>", "<atem:function_calls>")
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def split_reasoning(text: str, *, opened: bool = False) -> tuple[str | None, str]:
    """(reasoning, rest) of a think-block completion (Qwen3.x, GLM). The reasoning runs to the first
    ``</think>``. ``opened``: the prompt's generation cue already opened the block, so the completion
    starts inside it and one cut off before ``</think>`` is all reasoning. Otherwise only a completion
    that itself starts with ``<think>`` has reasoning, so an answer that merely mentions the tag is
    left alone. No reasoning -> (None, text)."""
    if not opened:
        body = text.lstrip()
        if not body.startswith(_THINK_OPEN):
            return None, text
        text = body[len(_THINK_OPEN):]
    head, sep, tail = text.partition(_THINK_CLOSE)
    if not sep:
        return text, ""
    if opened and _THINK_OPEN in head:  # the model re-opened the block it was already in
        head = head.split(_THINK_OPEN, 1)[1]
    return head, tail


def parse_assistant_output(
    text: str, tools: list[dict] | None = None, *, strip: bool = True,
    think_opened: bool = False, parse_calls: bool = True,
) -> tuple[str, str | None, list[dict]]:
    """Returns (content, reasoning_content, openai_tool_calls) for any of the
    wire formats. Channel-format completions begin mid-segment (generation
    starts after `<|start|>assistant`), so a virtual header is prepended.
    Think-block completions lose their reasoning to ``reasoning_content``
    (``think_opened``: the prompt opened the block, see split_reasoning).
    ``parse_calls=False`` leaves tool-call markup in the content (a request
    that offered no tools).

    ``strip=False`` keeps the reasoning and the channel segment bodies verbatim
    (no whitespace trimming, segments concatenated as they were generated), so
    re-rendering the message with the chat template reproduces the generated
    tokens exactly, as teacher-forced replay needs."""
    if not any(m in text for m in _CHANNEL_MARKERS):
        reasoning, rest = split_reasoning(text, opened=think_opened)
        content, calls = parse_tool_calls(rest, tools) if parse_calls else (rest.strip() if strip else rest, [])
        if strip and reasoning is not None:
            reasoning = reasoning.strip() or None
        return content, reasoning, calls

    calls: list[dict] = []
    types = _schema_types(tools)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    def _parse_atem(body: str) -> bool:
        """Append calls from a tool-recipient segment; False if nothing parsed."""
        found = False
        for block in _ATEM_BLOCK_RE.findall(body):
            for name, inner in _ATEM_INVOKE_RE.findall(block):
                ptypes = types.get(name) or types.get(name.split(".")[-1], {})
                args = {
                    pname: _coerce(pval, ptypes.get(pname, "string"))
                    for pname, pval in _ATEM_PARAM_RE.findall(inner)
                }
                calls.append(
                    {
                        "id": f"call_{len(calls)}",
                        "type": "function",
                        "function": {
                            "name": name.split(".")[-1],
                            "arguments": json.dumps(args),
                        },
                    }
                )
                found = True
        return found

    full = text if text.lstrip().startswith("<|start|>") else "<|start|>assistant" + text
    matched_any = False
    for role, recipient, body in _CHANNEL_RE.findall(full):
        matched_any = True
        if role != "assistant":
            content_parts.append(body)  # hallucinated non-assistant channel: keep visible
        elif recipient == "self":
            reasoning_parts.append(body)
        elif recipient in ("", "user"):
            content_parts.append(body)
        elif not parse_calls or not _parse_atem(body):
            content_parts.append(body)  # no tools offered, or a malformed call: keep it visible
    if not matched_any:  # ATEM block without channel wrapper
        if not _parse_atem(text):
            content_parts.append(text)
        else:
            content_parts.append(_ATEM_BLOCK_RE.sub("", text))

    if not strip:
        return "".join(content_parts), "".join(reasoning_parts) or None, calls
    content = "\n".join(p.strip() for p in content_parts if p.strip()).strip()
    reasoning = "\n".join(p.strip() for p in reasoning_parts if p.strip()).strip() or None
    return content, reasoning, calls
