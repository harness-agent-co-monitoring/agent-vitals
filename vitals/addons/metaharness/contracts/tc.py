"""Structural section annotation for classification harness prompts: named
functional blocks parsed from the BUILT prompt rather than matched against
stored memory. The fallback when a design declares no prompt_sections().

Every evolved prompt has the skeleton
    header -> memory (per-label blocks) -> query -> directions -> format
The per-label memory marker varies by design and we detect all seen forms:
    "### Candidate answer: <label>"   (cases grouped by answer)
    "- **<label>** — ..."             (synthesized answer profiles)
    "Q: <case>\nA: <label>"           (few-shot demos)
This captures synthesized memory (profiles, grouped cases) that verbatim
matching of stored memory.json misses. Regions emitted:
    span:header, span:mem:<label> (many), span:query, span:directions,
    span:format
The replay drivers resolve span requests with ``episode_span_requests``. The
CLI annotates already-replayed stores after the fact: token offsets are
resolved with the serving template via render_turn and appended to each
episode's regions.jsonl (episodes that already carry a span:header region are
skipped unless --force).

Usage: python -m vitals.addons.metaharness.contracts.tc --replay-root <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from vitals.addons.metaharness.config import CFG

FORMAT_RE = re.compile(r'\{"reasoning":\s*"\[[^"]*\]",\s*"final_answer":\s*"\[[^"]*\]"\}')
PROBLEM_RE = re.compile(r'\n\s*\*\*Problem:\*\*\s*', re.I)
INSTR_RE = re.compile(r'\n\s*\*\*Instructions:\*\*', re.I)
# per-label memory block starts
LABEL_MARKERS = [
    re.compile(r'^###\s*Candidate answer:\s*(.+?)\s*$', re.M),
    re.compile(r'^-\s*\*\*(.+?)\*\*\s*[—-]', re.M),
]
QA_RE = re.compile(r'(?m)^Q:\s.*?(?:\nA:\s*(.+?)\s*$)', re.S)


def segment(prompt: str) -> list[dict]:
    """Return [{name, cs, ce}] char spans for the prompt's functional blocks."""
    n = len(prompt)
    spans: list[dict] = []

    fmt = FORMAT_RE.search(prompt)
    format_start = fmt.start() if fmt else n
    if fmt:
        spans.append({"name": "format", "cs": fmt.start(), "ce": n})

    prob = PROBLEM_RE.search(prompt)
    instr = INSTR_RE.search(prompt)
    prob_at = prob.end() if prob else None            # query content start
    instr_at = instr.start() if instr else None
    # memory ends where the query/directions begin
    dir_start = min([x for x in (prob.start() if prob else None,
                                 instr_at, format_start) if x is not None],
                    default=format_start)
    # query: **Problem:** content up to **Instructions:** (or format)
    if prob_at is not None:
        q_end = instr_at if instr_at and instr_at > prob_at else format_start
        spans.append({"name": "query", "cs": prob_at, "ce": q_end})
    # directions: **Instructions:** up to the format block (bullets only)
    if instr_at is not None and instr_at < format_start:
        spans.append({"name": "directions", "cs": instr_at, "ce": format_start})

    # per-label memory markers, in order of appearance
    marks = []
    for rx in LABEL_MARKERS:
        for m in rx.finditer(prompt):
            if m.start() < dir_start:
                marks.append((m.start(), m.group(1).strip()))
    for m in QA_RE.finditer(prompt):
        if m.start() < dir_start and m.group(1):
            marks.append((m.start(), m.group(1).strip()))
    marks.sort()

    if marks:
        # header: leading text before the first memory block
        if marks[0][0] > 20:
            spans.append({"name": "header", "cs": 0, "ce": marks[0][0]})
        # each memory block runs to the next marker
        for i, (pos, label) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else dir_start
            spans.append({"name": f"mem:{label[:24]}", "cs": pos, "ce": end})
    else:
        # no memory blocks: everything before the directions is header + query
        spans.append({"name": "header", "cs": 0, "ce": dir_start})

    return [s for s in spans if s["ce"] > s["cs"]]


def char_to_span_requests(prompt: str, blocks: list[dict]) -> list[dict]:
    """render_turn takes {name, text}; give it the exact substring per block."""
    out = []
    for b in blocks:
        text = prompt[b["cs"]:b["ce"]].strip()
        if len(text) >= 8:
            out.append({"name": b["name"], "text": text})
    return out


def span_requests(prompt: str, declared) -> list[dict]:
    """The declared (name, text) sections found verbatim in the prompt, else
    the structural segmentation."""
    if declared:
        return [{"name": str(nm), "text": str(tx)} for nm, tx in declared if tx and tx in prompt]
    return char_to_span_requests(prompt, segment(prompt))


def episode_span_requests(ep: Path) -> list[dict] | None:
    """Span requests for one ingested classification episode: the declaration
    ingest stored in its manifest, else structural. None if unreadable."""
    try:
        prompt = json.loads((ep / "messages.jsonl").read_text(errors="replace")
                            .splitlines()[0])["messages"][0]["content"]
        declared = None
        try:
            declared = json.loads((ep / "manifest.json").read_text()).get("meta", {}).get("prompt_sections")
        except (OSError, json.JSONDecodeError):
            pass
        return span_requests(prompt, declared)
    except (OSError, IndexError, json.JSONDecodeError, KeyError):
        return None


def annotate_store(backend, store: Path, force: bool = False) -> tuple[int, int]:
    n_ep = n_span = 0
    mts = [p.stat().st_mtime for p in store.glob("ep-*/regions.jsonl")]
    if mts and time.time() - max(mts) < 180:
        print("  SKIP (actively replayed)")
        return 0, 0
    for ep in sorted(store.glob("ep-*")):
        rj = ep / "regions.jsonl"
        if rj.exists() and '"span:header"' in rj.read_text() and not force:
            continue
        try:
            msgs = json.loads((ep / "messages.jsonl").read_text(
                errors="replace").splitlines()[0])["messages"]
        except (OSError, IndexError, json.JSONDecodeError):
            continue
        prompt = msgs[0].get("content") or ""
        declared = None
        man = ep / "manifest.json"
        if man.exists():
            try:
                declared = json.loads(man.read_text()).get("meta", {}).get("prompt_sections")
            except (OSError, json.JSONDecodeError):
                pass
        reqs = span_requests(prompt, declared)
        if not reqs:
            continue
        ctx, toks = backend.render_turn(msgs, spans=reqs)
        recv = ep / "attn_received.jsonl"
        if recv.exists():
            try:
                n_recv = len(json.loads(recv.read_text().splitlines()[0])["vec"])
                if n_recv != len(toks):
                    continue
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
        spans = [r for r in ctx.regions if str(r.id).startswith("span:")]
        if not spans:
            continue
        with rj.open("a") as f:
            f.write(json.dumps({"turn": 0, "regions": [
                r.model_dump(mode="json") for r in ctx.regions]}) + "\n")
        n_ep += 1
        n_span += len(spans)
    return n_ep, n_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-root", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(str(CFG.reference_gguf), n_ctx=32768, n_gpu_layers=0)
    for d in sorted(args.replay_root.iterdir()):
        store = d / "store"
        if not (d.is_dir() and "--" in d.name and store.is_dir()):
            continue
        ne, ns = annotate_store(be, store, force=args.force)
        print(f"{d.name}: {ne} episodes, {ns} spans")


if __name__ == "__main__":
    main()
