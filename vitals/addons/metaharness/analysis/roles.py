"""Functional roles of declared sections, assigned from the section name alone."""

from __future__ import annotations

from pathlib import Path

from vitals.core.store import read_regions


def role_of(name: str) -> str:
    """Map a section name to its functional role. Order matters: explicit
    roles first, then the per-label / known-guidance fallbacks."""
    base = name.split(":", 1)[0].lower()
    if base == "format_contract" or base.startswith("format"):
        return "contract"
    if base.startswith("query"):
        return "instance"
    if (base.startswith("directive") or base.startswith("step")
            or base.endswith("_prior")
            or base in ("instructions", "instruction", "directions")):
        return "directive"
    if base in ("task_framing", "header") or base.endswith("_intro"):
        return "framing"
    if "inventory" in base:
        return "inventory"
    if ":" in name or base in (
            "cases", "profile", "signals", "demo", "demos", "coverage_example",
            "instead", "contrastive_pair", "worked", "lesson", "lessons",
            "exemplar", "example", "examples", "memory", "precedent", "rules",
            "rule", "catalogue", "evidence", "store"):
        return "guidance"
    return "other"


def span_footprints(ep: Path) -> dict[str, int]:
    """token_end - token_start per span: section (turn 0)."""
    out: dict[str, int] = {}
    for _t, regs in read_regions(ep).items():
        for r in regs:
            rid = str(r.get("id", ""))
            s, e = r.get("token_start"), r.get("token_end")
            if rid.startswith("span:") and s is not None and e is not None:
                out[rid.removeprefix("span:")] = int(e) - int(s)
    return out
