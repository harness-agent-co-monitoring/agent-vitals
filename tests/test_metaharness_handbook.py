"""HANDBOOK completion-log ingest: pure helpers."""
from vitals.addons.metaharness.ingest.handbook import doc_sections, flatten_message, flatten_text, turn_action


def test_flatten_text_blocks_and_messages():
    assert flatten_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert flatten_text("plain") == "plain"
    m = flatten_message({"role": "assistant", "content": [{"type": "text", "text": "x"}],
                         "tool_calls": [{"id": "c1", "index": 0, "function": {"name": "f", "arguments": "{}"}}]})
    assert m == {"role": "assistant", "content": "x",
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}


def test_doc_sections_numbered_body_after_toc():
    toc = "Contents\\n1. Overview\\n2. Scope\\n3. Roles\\n4. Rules\\n5. Templates\\n"
    body = "".join(f"{i}. {h}\\n" + ("x" * 400) + "\\n" for i, h in
                   enumerate(["Overview", "Scope", "Roles", "Rules", "Templates"], 1))
    secs = doc_sections('{"pages":["' + toc + body + '"]}')
    assert [h for h, _ in secs] == ["1. Overview", "2. Scope", "3. Roles", "4. Rules", "5. Templates"]
    assert all(len(c) >= 300 for _, c in secs)


def test_doc_sections_rejects_numbered_list_items_and_short_docs():
    lst = "".join(f"{i}. Do not contact the supplier by phone: {'y' * 400}\\n" for i in range(1, 8))
    assert doc_sections(lst) == []
    assert doc_sections("1. A\\n2. B\\n") == []


def test_doc_sections_html_headings():
    html = "<!DOCTYPE html><html><body>" + "".join(
        f"<h2>Section {i}</h2><p>{'z' * 400}</p>" for i in range(1, 7)) + "</body></html>"
    secs = doc_sections(html)
    assert [h for h, _ in secs] == [f"Section {i}" for i in range(1, 7)]


def test_turn_action():
    a = turn_action({"content": "", "reasoning_content": "abc",
                     "tool_calls": [{"function": {"name": "think"}}, {"function": {"name": "finish"}}]})
    assert a == {"action": "think", "calls": ["think", "finish"], "says": False, "reasoning_chars": 3}
    assert turn_action({"content": "done"})["action"] == "message"


def test_system_spans_and_subsections_and_paragraphs():
    from vitals.addons.metaharness.ingest.handbook import paragraph_chunks, system_spans
    sp = system_spans("Today's date is, April, 7, 2026, noon. You are an office assistant with tools. "
                      "The workspace directory /workdir contains files. Do NOT ask the user for more information. "
                      "Begin by listing /workdir and reading relevant files.")
    assert [n for n, _ in sp] == ["sys:date", "sys:role", "sys:workspace", "sys:no-ask", "sys:begin"]
    assert sp[3][1].startswith("Do NOT ask")
    body = "1. Leave Policy\\n" + "a" * 300 + "\\n1.1 Eligibility\\n" + "b" * 300 + "\\n1.2 Deadlines\\n" + "c" * 300 + "\\n"
    rest = "".join(f"{i}. Other {i}\\n" + "z" * 400 + "\\n" for i in range(2, 7))
    secs = doc_sections('{"pages":["' + body + rest + '"]}')
    names = [h for h, _ in secs]
    assert names[:3] == ["1. Leave Policy", "1.1 Eligibility", "1.2 Deadlines"]
    assert len(names) == 3 + 5
    paras = paragraph_chunks("x" * 1600 + "\\n" + "y" * 1600 + "\\n" + "z" * 100)
    assert [n for n, _ in paras] == ["¶0", "¶1"] and paras[1][1].endswith("z" * 100)
