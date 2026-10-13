"""Unit tests: tool-call and reasoning extraction into OpenAI messages (no GPU)."""

import json

from vitals.server.toolcalls import parse_assistant_output, parse_tool_calls, split_reasoning


def test_single_call_extracted():
    text = 'Let me check.\n<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls -la"}}\n</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == "Let me check."
    assert len(calls) == 1
    assert calls[0]["type"] == "function" and calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"cmd": "ls -la"}


def test_multiple_calls_ordered_ids():
    text = (
        '<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>\n'
        '<tool_call>{"name": "read", "arguments": {"path": "b.py"}}</tool_call>'
    )
    content, calls = parse_tool_calls(text)
    assert content == ""
    assert [c["id"] for c in calls] == ["call_0", "call_1"]


def test_malformed_json_stays_in_content():
    text = '<tool_call>\n{"name": "bash", "arguments": {broken\n</tool_call> trailing'
    content, calls = parse_tool_calls(text)
    assert calls == []
    assert "<tool_call>" in content and "trailing" in content


def test_string_arguments_pass_through():
    text = '<tool_call>{"name": "bash", "arguments": "{\\"cmd\\": \\"ls\\"}"}</tool_call>'
    _, calls = parse_tool_calls(text)
    assert calls[0]["function"]["arguments"] == '{"cmd": "ls"}'


def test_no_calls_is_identity():
    content, calls = parse_tool_calls("Just an answer with <think>tags</think>.")
    assert calls == [] and "tags" in content


TOOLS = [{"type": "function", "function": {
    "name": "write", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"},
        "line": {"type": "integer"}, "force": {"type": "boolean"}}}}}]


def test_xml_parameter_convention():
    text = ("I'll create the file.\n<tool_call>\n<function=write>\n"
            "<parameter=path>\nhello.txt\n</parameter>\n"
            "<parameter=content>\nhello pi\nsecond line\n</parameter>\n"
            "</function>\n</tool_call>")
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "I'll create the file."
    assert calls[0]["function"]["name"] == "write"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"path": "hello.txt", "content": "hello pi\nsecond line"}


def test_xml_schema_type_coercion():
    text = ("<tool_call>\n<function=write>\n"
            "<parameter=path>\n123\n</parameter>\n"        # string schema: stays string
            "<parameter=line>\n42\n</parameter>\n"          # integer schema: coerced
            "<parameter=force>\ntrue\n</parameter>\n"       # boolean schema: coerced
            "</function>\n</tool_call>")
    _, calls = parse_tool_calls(text, TOOLS)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"path": "123", "line": 42, "force": True}


def test_xml_malformed_stays_in_content():
    text = "<tool_call>\n<function=write>\n<parameter=path>\nx\n</parameter>\n</tool_call>"
    content, calls = parse_tool_calls(text, TOOLS)  # missing </function>
    assert calls == [] and "<function=write>" in content


# --- channel format (Muse Glimmer / Harmony convention) ---

BASH_TOOL = [{"type": "function", "function": {
    "name": "bash", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}, "required": ["command"]}}}]

# a verbatim Muse Glimmer completion
MUSE_COMPLETION = (
    " to=self<|message|>Probably use df -h or df. Let's run command.<|eom|>"
    "<|start|>assistant to=bash<|message|><atem:function_calls>\n"
    '<atem:invoke name="bash">\n'
    "<atem:parameter name=\"command\">df -h</atem:parameter>\n"
    "</atem:invoke>\n</atem:function_calls><|eot|>")


def test_channel_reasoning_and_call():
    content, reasoning, calls = parse_assistant_output(MUSE_COMPLETION, BASH_TOOL)
    assert reasoning == "Probably use df -h or df. Let's run command."
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "df -h"}


def test_channel_user_facing_answer():
    text = " to=user<|message|>There is 3.1T free on /.<|eot|>"
    content, reasoning, calls = parse_assistant_output(text, BASH_TOOL)
    assert content == "There is 3.1T free on /." and reasoning is None and calls == []


def test_channel_no_recipient_is_content():
    text = "<|message|>Done - the fix is committed.<|eot|>"
    content, reasoning, calls = parse_assistant_output(text, BASH_TOOL)
    assert content == "Done - the fix is committed." and calls == []


def test_channel_namespaced_invoke_and_coercion():
    text = (" to=fs<|message|><atem:function_calls>\n"
            '<atem:invoke name="fs.write">\n'
            "<atem:parameter name=\"path\">a.txt</atem:parameter>\n"
            "<atem:parameter name=\"line\">7</atem:parameter>\n"
            "</atem:invoke>\n</atem:function_calls><|eot|>")
    tools = [{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"},
                                         "line": {"type": "integer"}}}}}]
    _, _, calls = parse_assistant_output(text, tools)
    assert calls[0]["function"]["name"] == "write"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt", "line": 7}


def test_channel_malformed_atem_stays_in_content():
    text = (" to=bash<|message|><atem:function_calls>\n"
            '<atem:invoke name="bash">\n'
            "<atem:parameter name=\"command\">ls</atem:parameter>\n<|eot|>")
    content, _, calls = parse_assistant_output(text, BASH_TOOL)  # missing closers
    assert calls == [] and "<atem:invoke" in content


def test_channel_multiline_param_preserved():
    text = (" to=bash<|message|><atem:function_calls>\n"
            '<atem:invoke name="bash">\n'
            "<atem:parameter name=\"command\">cat <<'X' > f.py\nline1\n line2\nX</atem:parameter>\n"
            "</atem:invoke>\n</atem:function_calls><|eot|>")
    _, _, calls = parse_assistant_output(text, BASH_TOOL)
    assert json.loads(calls[0]["function"]["arguments"])["command"] == \
        "cat <<'X' > f.py\nline1\n line2\nX"


def test_legacy_format_passthrough():
    text = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
    content, reasoning, calls = parse_assistant_output(text, BASH_TOOL)
    assert reasoning is None and len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"


def test_think_block_reasoning_is_split():
    qwen = "Count the colors.\n</think>\n\nRed, yellow, blue."          # cue opened <think>
    assert split_reasoning(qwen, opened=True) == ("Count the colors.\n", "\n\nRed, yellow, blue.")
    assert parse_assistant_output(qwen, think_opened=True, parse_calls=False) == \
        ("Red, yellow, blue.", "Count the colors.", [])
    # cut off by max_tokens inside the block: all reasoning
    assert parse_assistant_output("still thinking", think_opened=True) == ("", "still thinking", [])
    # the model opened the block itself
    assert parse_assistant_output("<think>R</think>A") == ("A", "R", [])
    # an answer that only mentions the tag is left alone
    assert parse_assistant_output("Close it with </think> later.") == ("Close it with </think> later.", None, [])
    # verbatim mode keeps the reasoning's whitespace
    assert parse_assistant_output(qwen, think_opened=True, strip=False, parse_calls=False)[1] == "Count the colors.\n"


def test_glm_reasoning_and_tool_call():
    text = "Need the listing.</think>Listing it.<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    content, reasoning, calls = parse_assistant_output(text, BASH_TOOL, think_opened=True)
    assert reasoning == "Need the listing." and content == "Listing it."
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}
    # a request that offered no tools keeps the markup in the content
    content, _, calls = parse_assistant_output(text, think_opened=True, parse_calls=False)
    assert calls == [] and content.startswith("Listing it.<tool_call>")

