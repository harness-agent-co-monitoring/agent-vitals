"""Turn records: argument values located exactly in every tool-call format, vocabulary decoding, leaves."""

from vitals.monitor.turns import leaves, locate_calls
from vitals.monitor.vocab import Vocab

TOOLS = [{"type": "function", "function": {"name": "book", "parameters": {"properties": {
    "rid": {"type": "string"}, "flights": {"type": "array"}, "n": {"type": "integer"}}}}}]


def _values(text, calls):
    return [{a["name"]: text[a["span"][0]:a["span"][1]] for a in c["args"]} for c in calls]


def test_xml_parameter_format():
    text = ('<think>plan</think>\n\nOk.\n<tool_call>\n<function=book>\n<parameter=rid>\nABC123\n</parameter>\n'
            '<parameter=flights>\n[{"flight_number": "HAT005"}]\n</parameter>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call>')
    calls = locate_calls(text, TOOLS)
    assert [c["name"] for c in calls] == ["book"]
    assert _values(text, calls) == [{"rid": "ABC123", "flights": '[{"flight_number": "HAT005"}]', "n": "2"}]
    by = {a["name"]: a["value"] for a in calls[0]["args"]}
    assert by["flights"] == [{"flight_number": "HAT005"}] and by["n"] == 2


def test_json_format_and_two_calls():
    text = ('<tool_call>\n{"name": "book", "arguments": {"rid": "ABC123", "n": 2}}\n</tool_call>\n'
            '<tool_call>\n{"name": "book", "arguments": {"rid": "XYZ999"}}\n</tool_call>')
    calls = locate_calls(text, TOOLS)
    assert _values(text, calls) == [{"rid": '"ABC123"', "n": "2"}, {"rid": '"XYZ999"'}]


def test_glm_and_channel_formats():
    glm = "<tool_call>book\n<arg_key>rid</arg_key>\n<arg_value>ABC123</arg_value>\n</tool_call>"
    assert _values(glm, locate_calls(glm, TOOLS)) == [{"rid": "ABC123"}]
    atem = ('<atem:function_calls>\n<atem:invoke name="book">\n<atem:parameter name="rid">ABC123</atem:parameter>\n'
            '</atem:invoke>\n</atem:function_calls>')
    assert _values(atem, locate_calls(atem, TOOLS)) == [{"rid": "ABC123"}]


def test_malformed_call_is_skipped():
    assert locate_calls("<tool_call>\n{not json\n</tool_call>") == []


def test_leaves():
    assert leaves([{"flight_number": "HAT005", "date": "2024-05-20"}, True, 7, "no"]) == ["HAT005", "2024-05-20"]


def test_gpt2_vocab_and_offsets():
    v = Vocab.from_tokens(["Hello", "Ġwor", "ld", "<tool_call>", "Ã©"], "gpt2", [1, 1, 1, 3, 1])
    text, offs = v.text_and_offsets([0, 1, 2, 3, 4])
    assert text == "Hello world<tool_call>é"
    assert [text[a:b] for a, b in offs] == ["Hello", " wor", "ld", "<tool_call>", "é"]
    sp = Vocab.from_tokens(["▁Hi", "<0x0A>", "x"], "llama", [1, 6, 1])
    assert sp.text_and_offsets([0, 1, 2])[0] == " Hi\nx"
