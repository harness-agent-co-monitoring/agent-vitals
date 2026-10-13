"""Prompt-section sources for TB2 agent rollouts (the terminal-bench
counterpart of the text-classification `prompt_sections()` contract).

Canonical home of the declaration contracts (prompt_sections,
feedback_sections: AST loaders + validators) and the TB structural
fallback. Stdlib-only by design so an optimizer environment can import it
without the vitals engine; the meta-harness repo keeps a vendored copy at
reference_examples/terminal_bench_2/prompt_provenance.py that must stay in
sync with this file.

A TB2 worker prompt is: an AUTHORED scaffold header (task framing, response
format, field spec, command structure, notes, the injected task description)
followed by the DYNAMIC terminal transcript, which the environment — not the
harness — generates. Only the authored header is attributable to the harness,
so sections label that header and coverage is measured over it (never over
the growing transcript, unlike classification where the harness authored the
whole prompt).

Used by the deferred replay: for each logged worker call it resolves the
authored section spans (proposer-declared if the harness supplies them, else
structural) so section-pooled attention/reading can be attributed back to the
scaffold component that wrote each span.

`structural_sections` recognizes the stock terminus-2 scaffold (json-plain /
xml-plain). A proposer that adds its own scaffold blocks (environment
bootstrap, a skills catalogue, rewritten instructions) declares them by
defining a module-level `prompt_sections(system_prompt)` in its agent file;
`sections_for` prefers that when it validates, else falls back here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Ordered terminus-2 scaffold markers. task_framing is the head before the
# first marker; the authored region ends at the dynamic terminal transcript.
_MARKERS = [
    ("format_contract", re.compile(r"Format your response as (?:JSON|XML)", re.I)),
    ("field_spec", re.compile(r"^Required fields:", re.M)),
    ("command_structure", re.compile(r"^Command object structure:", re.M)),
    ("important_notes", re.compile(r"^(?:IMPORTANT:|Important notes:)", re.M)),
    ("task_description", re.compile(r"^Task Description:", re.M)),
]
_TRANSCRIPT = re.compile(r"^Current terminal state:", re.M)


def authored_region(prompt: str) -> str:
    """The harness-authored prefix, i.e. everything before the dynamic
    terminal transcript."""
    m = _TRANSCRIPT.search(prompt)
    return prompt[: m.start()] if m else prompt


def structural_sections(prompt: str) -> list[tuple[str, str]]:
    """Label the authored terminus scaffold as [(name, exact_text)]."""
    authored = authored_region(prompt)
    hits = []
    for name, rx in _MARKERS:
        m = rx.search(authored)
        if m:
            hits.append((m.start(), name))
    hits.sort()
    spans: list[tuple[str, str]] = []
    if not hits:
        spans.append(("task_framing", authored.strip()))
    else:
        if hits[0][0] > 0:
            spans.append(("task_framing", authored[: hits[0][0]].strip()))
        for i, (pos, name) in enumerate(hits):
            end = hits[i + 1][0] if i + 1 < len(hits) else len(authored)
            spans.append((name, authored[pos:end].strip()))
    return [(n, t) for n, t in spans if len(t) >= 8]


def validate_prompt_sections(
    sections: list[tuple[str, str]] | None, prompt: str, min_coverage: float = 0.8
) -> tuple[bool, str]:
    """Every declared text must occur verbatim in `prompt`, and the spans must
    cover >= min_coverage of the AUTHORED region (the transcript is excluded
    from the denominator). Returns (ok, reason)."""
    if not sections:
        return False, "no sections declared"
    authored_len = len(authored_region(prompt))
    covered = [False] * len(prompt)
    for name, text in sections:
        if not name or not text:
            return False, "empty name or text"
        pos = prompt.find(text)
        if pos < 0:
            return False, f"section {name!r} text not found verbatim"
        for i in range(pos, min(pos + len(text), len(covered))):
            covered[i] = True
    frac = sum(covered[:authored_len]) / max(authored_len, 1)
    if frac < min_coverage:
        return False, f"authored coverage {frac:.0%} < {min_coverage:.0%}"
    return True, f"ok ({len(sections)} sections, {frac:.0%} authored coverage)"


def sections_for(
    prompt: str, declared: list[tuple[str, str]] | None = None
) -> list[tuple[str, str]]:
    """Prefer proposer-declared sections when they validate; else structural."""
    if declared:
        ok, _ = validate_prompt_sections(declared, prompt)
        if ok:
            return [(str(n), str(t)) for n, t in declared]
    return structural_sections(prompt)


def load_declared_function(agent_py: str | Path, func_name: str):
    """Extract a module-level function from an agent file WITHOUT importing
    the module. Agent files import harbor/litellm, which the replay venv does
    not have; the contract therefore requires declared functions to be
    self-contained (constants in the body; imports, if any, inside the body
    and limited to the stdlib or this module). The function def is located by
    AST, compiled alone, and returned as a callable — or None when the file
    is missing, unparsable, lacks the function, or the extracted def fails to
    compile."""
    try:
        tree = ast.parse(Path(agent_py).read_text())
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            ns: dict = {}
            mod = ast.Module(body=[node], type_ignores=[])
            try:
                exec(compile(mod, str(agent_py), "exec"), ns)
            except Exception:
                return None
            return ns.get(func_name)
    return None


def load_declared_prompt_sections(agent_py: str | Path):
    """The scaffold declaration: see ``load_declared_function``."""
    return load_declared_function(agent_py, "prompt_sections")


def load_declared_feedback_sections(agent_py: str | Path):
    """The runtime-feedback declaration (optional companion contract): a
    module-level ``feedback_sections(observation_text)`` returning
    (name, exact_text) pairs labeling the HARNESS-AUTHORED spans inside one
    observation — retry nudges, completion-confirmation checklists,
    truncation warnings, tool-result headers. The untagged remainder of the
    observation is environment (terminal) output. Same self-containment
    rules as prompt_sections."""
    return load_declared_function(agent_py, "feedback_sections")


def validate_feedback_sections(
    sections: list[tuple[str, str]] | None, observation: str
) -> tuple[bool, str]:
    """Every declared feedback text must occur verbatim in ``observation``.
    Unlike prompt_sections there is NO coverage requirement: most
    observations are pure terminal output and should yield no spans at all.
    An empty declaration is valid (it asserts 'no harness text here')."""
    if not sections:
        return True, "ok (no harness-authored spans)"
    for name, text in sections:
        if not name or not text:
            return False, "empty name or text"
        if observation.find(text) < 0:
            return False, f"feedback span {name!r} text not found verbatim"
    return True, f"ok ({len(sections)} feedback spans)"
