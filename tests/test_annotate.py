import json

from vitals.cli import main
from vitals.core.annotate import check_sections, sections_from_markdown, sections_from_markers

DOC = """# Support Agent

You help customers with orders.

## Rules

Never share another customer's data.

Confirm before refunds.

### Refunds

Refund within 30 days.

```bash
# not a heading
echo done
```

## Output

Reply in one short paragraph.
"""


def test_headings_partition_the_document_verbatim():
    secs = sections_from_markdown(DOC)
    assert [s["name"] for s in secs] == ["support_agent", "rules", "rules/refunds", "output"]
    assert all(s["text"] in DOC for s in secs)
    assert "# not a heading" in secs[2]["text"]           # fenced code stays inside its block


def test_paragraphs_are_named_by_heading_path_and_first_words():
    secs = sections_from_markdown(DOC, by="paragraph")
    names = [s["name"] for s in secs]
    assert "rules/never_share_another_customer" in names
    assert "rules/confirm_before_refunds" in names
    assert any(n.startswith("rules/refunds/") and "echo done" in s["text"] for n, s in zip(names, secs, strict=True))
    assert all(s["text"] in DOC for s in secs)


def test_repeated_names_get_a_suffix():
    secs = sections_from_markdown("## A\n\nsame start here\n\nsame start here too\n", by="paragraph")
    assert [s["name"] for s in secs] == ["a/same_start_here", "a/same_start_here_too"]
    secs = sections_from_markdown("## A\n\nx one\n\n## A\n\ny two\n")
    assert [s["name"] for s in secs] == ["a", "a#2"]


def test_markers_are_stripped_and_sections_match_the_clean_text():
    raw = ("<!-- section: role -->\nYou are a support agent.\n<!-- /section -->\n"
           "Unmarked text.\n<!-- section: rules -->\nNo refunds after 30 days.\n")
    clean, secs = sections_from_markers(raw)
    assert "<!--" not in clean
    assert secs == [{"name": "role", "text": "You are a support agent."},
                    {"name": "rules", "text": "No refunds after 30 days."}]
    assert all(s["text"] in clean for s in secs)


def test_check_reports_missing_ambiguous_later_and_coverage():
    prompt = "A rule. B rule. A rule."
    rep = check_sections([{"name": "a", "text": "A rule."}, {"name": "b", "text": "B rule."},
                          {"name": "c", "text": "C rule."}, {"name": "a2", "text": "A rule.", "occurrence": 2},
                          {"name": "m", "text": "B rule.", "msg_index": 5}], prompt, ["A rule. B rule. A rule."])
    status = {r["name"]: r["status"] for r in rep["sections"]}
    assert status == {"a": "ambiguous", "b": "ok", "c": "missing", "a2": "ok", "m": "later"}
    assert rep["missing"] == ["c"]
    assert abs(rep["coverage"] - len("A rule. B rule. A rule.".replace(" ", "")) / len(prompt)) < 0.2


def test_cli_annotates_and_checks(tmp_path, capsys):
    doc = tmp_path / "prompt.md"
    doc.write_text(DOC)
    out = tmp_path / "sections.json"
    assert main(["annotate", str(doc), "--out", str(out)]) == 0
    assert [s["name"] for s in json.loads(out.read_text())][:2] == ["support_agent", "rules"]
    assert main(["annotate", "--check", str(out), str(doc)]) == 0
    assert "coverage" in capsys.readouterr().out
    marked = tmp_path / "marked.md"
    marked.write_text("<!-- section: x -->\nHello there.\n")
    clean = tmp_path / "clean.md"
    assert main(["annotate", str(marked), "--by", "marker", "--clean-out", str(clean)]) == 0
    assert clean.read_text() == "Hello there.\n"


def test_template_placeholders_split_sections():
    tpl = "## Task\n\nSolve this issue: {{ task }}\n\nWork in {workdir} and reply with a patch only.\n"
    secs = sections_from_markdown(tpl, template=True)
    assert [s["text"] for s in secs] == ["## Task\n\nSolve this issue:", "and reply with a patch only."]
    assert [s["name"] for s in secs] == ["task", "task#2"]
    rendered = tpl.replace("{{ task }}", "fix the parser").replace("{workdir}", "/repo")
    assert check_sections(secs, rendered)["missing"] == []
