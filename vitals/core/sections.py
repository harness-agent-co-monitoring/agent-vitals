"""Section names: how positions get them, how they group, and the keys that match a section across turns.

A stored turn names its sections through regions (``regions.jsonl``): message regions (``region:<id>``) and
declared or automatic sections (``span:<name>``). ``section_labels`` turns those into one name per position;
``spans_from_labels`` turns the names back into token runs.

Names follow a small grammar used by every view:

- harness sections are free names, grouped by the part before the first ``:`` or ``/``
  (``policy:Book flight/payment`` is in group ``policy``, under parent ``policy:Book flight``);
- conversation sections start with a role: ``user:k``, ``self:k`` (the agent's own earlier turns),
  ``tool_result:<tool>#k``, and the older ``obs[...]``, ``fb...``, ``gen...`` forms;
- ``output`` is the turn's own generated text so far, ``region:<id>`` a message region no section covers.

``section_keys`` maps names to keys that stay the same across decisions (``user:latest``,
``tool_result:get_order:earlier``), so a section's typical value can be learned over many decisions.
"""

from __future__ import annotations

import re

OUTPUT = "output"            # the turn's forced output in per-position labels
REASONING = "reasoning"      # key of OUTPUT: the turn's own reasoning so far
TEMPLATE = "template"        # key of message regions no section covers (chat template text)
OTHER = "region:other"       # positions no region covers

CONVERSATION = re.compile(r"^(obs|user|self|fb|gen|tool_result)([:_\[]|$)")
INSTANCE = re.compile(r"#[^:/#]*$")     # tool_result:get_order#2 -> instance suffix "#2"


def _get(r, key: str):
    return r.get(key) if isinstance(r, dict) else getattr(r, key, None)


def section_labels(regions, n_prompt: int, n_out: int, *, group_sep: str | None = None) -> list[str]:
    """Section name of every prompt and forced-output position, a partition: a declared span (``span:<name>``
    region) takes precedence over the message region it sits in (``region:<id>``), nested spans go to the
    innermost, uncovered positions are ``region:other``, the forced output is ``output``. ``regions`` are Region
    objects or their dict form. ``group_sep`` maps span names to their part before the first separator."""
    lab = [""] * n_prompt
    spans = []
    for r in regions:
        s, e = _get(r, "token_start"), _get(r, "token_end")
        if s is None or e is None:
            continue
        rid = str(_get(r, "id"))
        if rid.startswith("span:"):
            spans.append((e - s, s, e, rid[5:]))
        else:
            for i in range(max(0, s), min(e, n_prompt)):
                lab[i] = f"region:{rid}"
    for _, s, e, name in sorted(spans, reverse=True):       # longest first: the innermost paints last
        if group_sep and group_sep in name:
            name = name.split(group_sep, 1)[0]
        for i in range(max(0, s), min(e, n_prompt)):
            lab[i] = name
    return [x or OTHER for x in lab] + [OUTPUT] * n_out


def spans_from_labels(labels: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Section name -> its contiguous runs [start, end) in a per-position label list."""
    spans: dict[str, list[tuple[int, int]]] = {}
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            spans.setdefault(labels[start], []).append((start, i))
            start = i
    return spans


def is_conversation(section: str) -> bool:
    """Sections cut from the conversation (observations, user turns, the agent's own turns), as opposed to
    the harness's sections."""
    return bool(CONVERSATION.match(section))


def is_user(section: str) -> bool:
    """A user turn (automatic ``user:k`` sections or the user message regions)."""
    return section.startswith("user:") or section.startswith("region:user")


def is_tool_result(section: str) -> bool:
    """A tool result or observation."""
    return section.startswith(("tool_result:", "obs:", "region:obs"))


def is_own(section: str) -> bool:
    """The model's own earlier text (messages and rendered tool calls), which repeats values from elsewhere."""
    return section.startswith(("self:", "region:gen"))


def group_of(section: str) -> str:
    """``policy:Book flight/payment`` -> ``policy``, ``obs[1-3]`` -> ``obs``; other names without a prefix
    form the top level."""
    m = re.match(r"^([^:/#]+)[:/]", section) or CONVERSATION.match(section)
    return m.group(1) if m else ""


def parent_of(section: str) -> str:
    """The section's enclosing heading: ``policy:Book flight/payment`` -> ``policy:Book flight``,
    ``LawBench:demo:allergy`` -> ``LawBench:demo``."""
    if "/" in section:
        return section.rsplit("/", 1)[0]
    return section.rsplit(":", 1)[0] if section.count(":") >= 2 else group_of(section)


def section_info(name: str) -> dict:
    """Name, group, parent and whether the section is conversation, as the dashboard lists sections."""
    return {"name": name, "group": group_of(name), "parent": parent_of(name), "conversation": is_conversation(name)}


def section_keys(sections: list[str]) -> list[str]:
    """Stable keys, in prompt order (the order sections first appear). ``output`` (the turn's own reasoning so
    far) becomes ``reasoning``, template regions ``template``; a conversation section becomes its base name (role,
    or role and tool for tool results) plus ``:latest`` for the last of that base and ``:earlier`` otherwise."""
    def base(n: str) -> str | None:
        if not CONVERSATION.match(n):
            return None
        n = re.sub(r"#[^:/#]*$", "", n)                  # tool_result:get_order#2 -> tool_result:get_order
        n = re.sub(r"[:_]\d+$", "", n)                   # user:3 -> user
        return re.sub(r"\[[^\]]*\]$", "", n)             # obs[4] -> obs

    bases = [base(n) for n in sections]
    last = {b: i for i, b in enumerate(bases) if b is not None}
    out = []
    for i, (n, b) in enumerate(zip(sections, bases, strict=True)):
        if n == OUTPUT:
            out.append(REASONING)
        elif n.startswith("region:"):
            out.append(TEMPLATE)
        elif b is None:
            out.append(n)
        else:
            out.append(f"{b}:{'latest' if last[b] == i else 'earlier'}")
    return out
