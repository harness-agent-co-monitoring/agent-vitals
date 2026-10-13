"""Section declarations for static harness text, and a check that declarations resolve.

A static harness (a system prompt file, an AGENTS.md, a policy document) has no code that could declare its
sections. These helpers derive declarations from the text itself: from its Markdown structure, or from inline
markers the author adds. ``check_sections`` reports whether a declaration resolves against the prompt the model
sees, using the same anchors as replay.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_MARKER = re.compile(r"<!--\s*(/?)section(?::\s*(.*?))?\s*-->[ \t]*\n?")
_PLACEHOLDER = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}|\$\{\w+\}|\{\w+\}", re.S)
MIN_FRAGMENT = 8


def _slug(text: str, n_words: int | None = None) -> str:
    words = re.findall(r"[^\W_]+", text.lower())
    return "_".join(words[:n_words] if n_words else words) or "section"


def _unique(name: str, seen: dict[str, int]) -> str:
    seen[name] = seen.get(name, 0) + 1
    return name if seen[name] == 1 else f"{name}#{seen[name]}"


def _blocks(text: str) -> list[tuple[int, int, list[str], int]]:
    """(start, end, heading path, body start) per heading block; text before the first heading has path []."""
    heads, pos, fenced = [], 0, False
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(line.rstrip("\n"))):
            heads.append((pos, len(m.group(1)), m.group(2).strip(), pos + len(line)))
        pos += len(line)
    # a single leading level-1 heading is the document title: its own block keeps it, its children do not
    title = heads[0][2] if heads and heads[0][1] == 1 and sum(h[1] == 1 for h in heads) == 1 else None
    out, stack = [], []
    if not heads or heads[0][0] > 0:
        out.append((0, heads[0][0] if heads else len(text), [], 0))
    for i, (start, level, head, body) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        if title is not None and i == 0:
            out.append((start, end, [head], body))
            continue
        stack = [t for t in stack if t[0] < level] + [(level, head)]
        out.append((start, end, [t for _, t in stack], body))
    return out


def _paragraphs(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Blank-line separated blocks of text[start:end]; fenced code stays in one block."""
    out, pos, para, fenced = [], start, None, False
    for line in text[start:end].splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        if line.strip() or fenced:
            para = (para[0] if para else pos, pos + len(line))
        elif para:
            out.append(para)
            para = None
        pos += len(line)
    if para:
        out.append(para)
    return out


def _span(text: str, start: int, end: int) -> str:
    return text[start:end].strip()


def _fill_holes(sections: list[dict]) -> list[dict]:
    """Split each section at template placeholders, keeping the fixed fragments that render verbatim."""
    out, seen = [], {}
    for sec in sections:
        for part in _PLACEHOLDER.split(sec["text"]):
            if len(part := part.strip()) >= MIN_FRAGMENT:
                out.append({"name": _unique(sec["name"], seen), "text": part})
    return out


def sections_from_markdown(text: str, *, by: str = "heading", template: bool = False) -> list[dict]:
    """Sections of a Markdown document, in document order, each a verbatim substring of ``text``.

    ``by="heading"`` gives one section per heading block (the heading line and the text up to the next
    heading), named by its heading path, e.g. ``book_flight/payment``. ``by="paragraph"`` gives one section per
    paragraph, named by heading path and first words. Text before the first heading is ``preamble``. Headings
    inside fenced code are ignored. With ``template=True`` the file is a prompt template: sections are split at
    placeholders (``{{ x }}``, ``{% ... %}``, ``{x}``, ``${x}``) so that only text rendered verbatim is declared.
    """
    if by not in ("heading", "paragraph"):
        raise ValueError(f"by must be 'heading' or 'paragraph', not {by!r}")
    sections, seen = [], {}
    for start, end, path, body in _blocks(text):
        base = "/".join(_slug(t) for t in path) or "preamble"
        if by == "heading":
            if body_text := _span(text, start, end):
                sections.append({"name": _unique(base, seen), "text": body_text})
            continue
        for s, e in _paragraphs(text, body, end):
            para = _span(text, s, e)
            sections.append({"name": _unique(f"{base}/{_slug(para, 4)}", seen), "text": para})
    return _fill_holes(sections) if template else sections


def sections_from_markers(text: str, *, template: bool = False) -> tuple[str, list[dict]]:
    """Strip inline section markers and return (clean text, sections).

    ``<!-- section: name -->`` opens a section, which runs to ``<!-- /section -->``, the next opening marker,
    or the end of the text. The markers are removed from the returned text, which is what the model should be
    served, and every section text is a verbatim substring of it.
    """
    clean, sections, seen = [], [], {}
    open_name, open_at, pos, length = None, 0, 0, 0

    def close(at: int) -> None:
        body = "".join(clean)[open_at:at].strip()
        if open_name is not None and body:
            sections.append({"name": _unique(open_name, seen), "text": body})

    for m in _MARKER.finditer(text):
        clean.append(text[pos:m.start()])
        length += m.start() - pos
        pos = m.end()
        close(length)
        if m.group(1):
            open_name = None
        else:
            open_name = (m.group(2) or "").strip() or "section"
            open_at = length
    clean.append(text[pos:])
    length += len(text) - pos
    close(length)
    return "".join(clean), _fill_holes(sections) if template else sections


def check_sections(sections: list[dict], prompt: str, messages: list[str] | None = None) -> dict:
    """Resolve each declared section the way replay does and report the result.

    ``prompt`` is the text the model sees. ``messages`` holds each message's text and is needed for sections
    anchored with ``msg_index``. Each section is ``ok``, ``missing`` (not found), ``later`` (anchored to a message
    that is not in this prompt yet), or ``ambiguous`` (found, but its text occurs more than once without an
    anchor, so replay takes the first copy). ``coverage`` is the share of prompt characters inside resolved
    sections.
    """
    covered = bytearray(len(prompt))
    rows = []
    for sec in sections:
        name, text = str(sec.get("name", "")), str(sec.get("text", ""))
        status, pos = "ok", -1
        if sec.get("msg_index") is not None and messages is not None:
            i = int(sec["msg_index"])
            if i < 0 or i >= len(messages):
                status = "later"
            elif text in messages[i]:
                at = prompt.find(messages[i])
                pos = prompt.find(text, at) if at >= 0 else prompt.find(text)
        else:
            anchor = str(sec["within"]) if sec.get("within") else text
            scan = 0
            for _ in range(max(1, int(sec.get("occurrence", 1)))):
                pos = prompt.find(anchor, scan)
                if pos < 0:
                    break
                scan = pos + 1
            if pos >= 0 and sec.get("within"):
                pos = prompt.find(text, pos, pos + len(anchor))
            elif pos >= 0 and "occurrence" not in sec and prompt.count(text) > 1:
                status = "ambiguous"
        if status != "later" and pos < 0:
            status = "missing"
        if pos >= 0:
            covered[pos:pos + len(text)] = b"\x01" * len(text)
        rows.append({"name": name, "status": status})
    return {"sections": rows,
            "coverage": sum(covered) / len(prompt) if prompt else 0.0,
            "missing": [r["name"] for r in rows if r["status"] == "missing"]}
