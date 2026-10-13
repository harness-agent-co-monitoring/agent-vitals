"""Automatic sections: names, tool-result naming through tool-call ids, precedence of declared sections, tool
schema lines; and, on the 0.8B CI model, that every automatic section resolves to tokens that decode to its
text."""

from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path

import pytest

from vitals.core.auto_sections import auto_spans, system_spans, tool_schema_spans

SYSTEM = ("You help airline customers.\n\n# Policy\n\n## Book flight\n\nAsk for the user id first.\n\n"
          "## Cancel flight\n\nConfirm before cancelling.\n")
TOOLS = [{"type": "function", "function": {"name": "get_reservation", "description": "Look up a reservation by id.",
                                           "parameters": {"type": "object", "properties": {"rid": {"type": "string"}}}}},
         {"type": "function", "function": {"name": "cancel", "description": "Cancel a reservation by id.",
                                           "parameters": {"type": "object", "properties": {"rid": {"type": "string"}}}}}]


def _chat():
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Cancel ABC123 please."},
        {"role": "assistant", "content": "Let me look it up.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_reservation", "arguments": '{"rid": "ABC123"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"rid": "ABC123", "status": "booked"}'},
        {"role": "assistant", "content": "Found it. Shall I cancel ABC123?"},
        {"role": "user", "content": "yes"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "get_reservation", "arguments": '{"rid": "ABC123"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": '{"rid": "ABC123", "status": "booked", "refundable": true}'},
    ]


def test_names_and_anchors():
    spans = auto_spans(_chat())
    names = [s["name"] for s in spans]
    # a single level-1 heading is the document title (annotate's rule): its children are not prefixed with it
    assert names == ["system/preamble", "system/policy", "system/book_flight", "system/cancel_flight", "user:1", "self:1",
                     "tool_result:get_reservation#1", "self:2", "user:2", "tool_result:get_reservation#2"]
    assert all(s["auto"] for s in spans)
    by = {s["name"]: s for s in spans}
    assert by["user:2"]["text"] == "yes" and by["user:2"]["msg_index"] == 5
    assert by["tool_result:get_reservation#2"]["msg_index"] == 7


def test_declared_sections_take_precedence():
    declared = [{"name": "booking_rules", "text": "Ask for the user id first."},
                {"name": "tool:get_reservation", "text": "x"}]
    spans = auto_spans(_chat(), TOOLS, "<tools>\n" + "\n".join(json.dumps(t) for t in TOOLS) + "\n</tools>", declared)
    names = {s["name"] for s in spans}
    assert not any(n.startswith("system") for n in names)            # the system message holds a declared section
    assert not any(n.startswith("tool:") for n in names)             # tool schemas are declared
    assert "tool_result:get_reservation#1" in names


def test_tool_schema_lines_and_fallback():
    prompt = "x <tools>\n" + "\n".join(json.dumps(t) for t in TOOLS) + "\n</tools> y"
    assert [s["name"] for s in tool_schema_spans(prompt, TOOLS)] == ["tool:get_reservation", "tool:cancel"]
    plain = "Tools: Look up a reservation by id. and Cancel a reservation by id."
    assert [s["text"] for s in tool_schema_spans(plain, TOOLS)] == ["Look up a reservation by id.", "Cancel a reservation by id."]


def test_flat_history_tool_output_and_short_system():
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "Running tests."}, {"role": "user", "content": "<returncode>1</returncode>\nFAILED test_x"}]
    names = [s["name"] for s in auto_spans(msgs)]
    assert names == ["system", "user:1", "self:1", "tool_result:output#1"]
    assert [s["name"] for s in system_spans("para one.\n\n" * 300, 0)][:2] != ["system"]


VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS, reason="pinned engine build or 0.8B GGUF not available")
def test_render_resolves_every_auto_section():
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(GGUFS[0], n_ctx=4096)
    try:
        msgs = _chat()
        ctx, toks = be.render_turn(msgs, tools=TOOLS, auto_sections=True)
        plain, toks2 = be.render_turn(msgs, tools=TOOLS)
        regs = {str(r.id)[5:]: r for r in ctx.regions if str(r.id).startswith("span:")}
        expect = {s["name"]: s["text"] for s in auto_spans(msgs, TOOLS, ctx.prompt_text)}
        assert toks == toks2 and not any(str(r.id).startswith("span:") for r in plain.regions)
        assert set(regs) == set(expect)
        for name, text in expect.items():
            r = regs[name]
            assert r.meta.get("auto") is True
            assert text in be.engine.detokenize(toks[r.token_start:r.token_end]), name
    finally:
        be.close()
