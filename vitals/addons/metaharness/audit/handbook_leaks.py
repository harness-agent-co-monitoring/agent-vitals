"""Leak audits for a HANDBOOK.md meta-harness loop.

The benchmark's point is that every task's rules live in a handbook the agent
must find and read in-episode, and that no two tasks share a policy. A
harness optimizer must therefore never (a) carry handbook/instruction/rubric
text into a candidate harness, (b) point the agent outside its workspace and
services (e.g. at the verifier), or (c) pre-load or pre-digest the handbook.
This module makes (a) and (b) mechanical:

  corpus     extract the text of every task's environment documents, the
             instruction, and the verifier files into one corpus (kept under
             runs/, outside any optimizer tree)
  candidate  check a candidate system prompt: verbatim word n-gram overlap
             with the corpus (default 8-grams), capitalized-name overlap,
             absolute paths other than /workdir, and file/verifier references
  trial      scan a Harbor trial's completion logs for agent tool calls that
             touch /tests, /initial_data, /logs, or verifier files
  code       audit a harness candidate MODULE: every string literal, docstring
             and comment goes through the `candidate` checks, and file-system
             access outside the model's own tool calls is listed (forbidden
             trees reject; /workdir access is reported for review)

  python -m vitals.addons.metaharness.audit.handbook_leaks corpus --tasks <tasks dir> --out runs/handbook/audit/corpus.jsonl
  python -m vitals.addons.metaharness.audit.handbook_leaks candidate --corpus ... --prompt <file> [--base <base prompt file>]
  python -m vitals.addons.metaharness.audit.handbook_leaks trial --trial <harbor trial dir> [...]
  python -m vitals.addons.metaharness.audit.handbook_leaks code --corpus ... --module <agent .py> [--base <baseline .py>]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

NGRAM = 8
FAMILY_MAX_TASKS = 9
FORBIDDEN_PATHS = ("/tests", "/initial_data", "/logs", "/app", "/root", "/etc", "/tmp/openhands")
FORBIDDEN_WORDS = ("sop_verifier", "rubrics.json", "task_prompt.txt", "verifier", "rubric", "reward.txt", "harbor")
DOC_EXT = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}
# a forbidden path counts only as a whole path component: '/app' must not match '/workdir/appeals.xlsx'
_FORBIDDEN_RE = re.compile(r"(?<![\w/.])(" + "|".join(re.escape(p) for p in FORBIDDEN_PATHS) + r")(?![\w\-])")


def forbidden_paths(text: str) -> list[str]:
    return sorted({m.group(1) for m in _FORBIDDEN_RE.finditer(text)})


def _pdf(p: Path) -> str:
    try:
        return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def _docx(p: Path) -> str:
    try:
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml").decode("utf8", "replace")
    except Exception:
        return ""
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", " ", xml)


def _html(p: Path) -> str:
    s = p.read_text(errors="replace")
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>|</p>|</h\d>|</li>|</tr>", "\n", s, flags=re.I)
    return re.sub(r"<[^>]+>", " ", s)


def doc_text(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".pdf":
        return _pdf(p)
    if ext == ".docx":
        return _docx(p)
    if ext in (".html", ".htm"):
        return _html(p)
    if ext in (".md", ".txt"):
        return p.read_text(errors="replace")
    return ""


def norm_words(text: str) -> list[str]:
    text = re.sub(r"&amp;", "&", text)
    return re.findall(r"[a-z0-9][a-z0-9'’\-]*", text.lower())


def shingles(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def cap_names(text: str) -> set[str]:
    """Adjacent capitalized word pairs (people, companies, products): every
    pair inside a run of capitalized words, so 'Contact Nicole Ashford' and
    'to Nicole Ashford' both yield 'Nicole Ashford'."""
    out: set[str] = set()
    for m in re.finditer(r"\b(?:[A-Z][a-zA-Z0-9&\-]+\s+)+[A-Z][a-zA-Z0-9&\-]+\b", text):
        ws = m.group(0).split()
        out.update(" ".join(ws[i:i + 2]) for i in range(len(ws) - 1))
    return out


def build_corpus(tasks_dir: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for task in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
            files = [task / "instruction.md", *sorted((task / "tests").glob("*"))]
            files += sorted(p for p in (task / "environment" / "initial_workspace").rglob("*")
                            if p.is_file() and p.suffix.lower() in DOC_EXT)
            for p in files:
                ext = p.suffix.lower()
                if ext in (".json", ".sh", ".py"):
                    text = p.read_text(errors="replace")
                else:
                    text = doc_text(p) if ext in DOC_EXT else ""
                if not text.strip():
                    continue
                f.write(json.dumps({"task": task.name, "file": str(p.relative_to(tasks_dir)), "text": text}) + "\n")
                n += 1
    return n


def load_corpus(path: Path):
    docs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    sh: dict[tuple[str, ...], set[str]] = {}
    names: dict[str, set[str]] = {}
    name_tasks: dict[str, set[str]] = {}
    for d in docs:
        for s in shingles(norm_words(d["text"]), NGRAM):
            sh.setdefault(s, set()).add(d["file"])
        for nm in cap_names(d["text"]):
            names.setdefault(nm, set()).add(d["file"])
            name_tasks.setdefault(nm, set()).add(d["task"])
    # a name is task-specific when only one base handbook's variants (<= 9 of
    # the 65 tasks) mention it: 'Nicole Ashford' yes, 'Human Resources' (two
    # HR bases, 13 tasks) and 'Standard Operating' (every handbook) no
    specific = {nm for nm, ts in name_tasks.items() if len(ts) <= FAMILY_MAX_TASKS}
    return sh, {nm: names[nm] for nm in specific}


def audit_candidate(prompt: str, corpus: Path, base: str = "") -> dict:
    sh, names = load_corpus(corpus)
    base_sh = shingles(norm_words(base), NGRAM) if base else set()
    base_names = cap_names(base) if base else set()
    hits = []
    for s in shingles(norm_words(prompt), NGRAM) - base_sh:
        if s in sh:
            hits.append({"ngram": " ".join(s), "files": sorted(sh[s])[:5]})
    name_hits = [{"name": nm, "files": sorted(names[nm])[:5]}
                 for nm in cap_names(prompt) - base_names if nm in names]
    paths = [p for p in re.findall(r"(?<![\w/])/[A-Za-z][\w./\-]*", prompt) if not p.startswith("/workdir")]
    words = [w for w in FORBIDDEN_WORDS if w in prompt.lower()]
    forbidden = sorted(set(forbidden_paths(prompt)) | set(words))
    verdict = "REJECT" if (hits or name_hits or forbidden) else "OK"
    return {"verdict": verdict, "ngram_hits": hits[:20], "name_hits": name_hits[:20],
            "paths_outside_workdir": sorted(set(paths)), "forbidden": forbidden}


def code_text(source: str) -> tuple[str, list[str]]:
    """(all string literals + docstrings + comments of a module joined by
    newlines, file-access lines). Access lines are `open(`, `Path(...).read*`,
    `os.listdir/walk`, `glob`, `subprocess`, `shutil` calls, with their line."""
    import ast
    import io
    import tokenize
    parts: list[str] = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                parts.append(node.value)
    except SyntaxError:
        parts.append(source)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                parts.append(tok.string.lstrip("# "))
    except (tokenize.TokenError, IndentationError):
        pass
    access = [f"{i}: {ln.strip()[:120]}" for i, ln in enumerate(source.splitlines(), 1)
              if re.search(r"\bopen\(|\.read_text\(|\.read_bytes\(|os\.listdir|os\.walk|os\.scandir|glob\.|\.glob\(|\.rglob\(|subprocess\.|shutil\.|\.iterdir\(", ln)]
    return "\n".join(parts), access


def audit_code(source: str, corpus: Path, base: str = "") -> dict:
    text, access = code_text(source)
    r = audit_candidate(text, corpus, base)
    r["file_access"] = access
    src_forbidden = forbidden_paths(source)
    if src_forbidden:
        r["forbidden"] = sorted(set(r["forbidden"]) | set(src_forbidden))
    if "/data" in source and re.search(r"(?<![\w/.])/data(?![\w\-])", source):
        r["forbidden"] = sorted(set(r["forbidden"]) | {"/data"})
    r["verdict"] = "REJECT" if (r["ngram_hits"] or r["name_hits"] or r["forbidden"]) else "OK"
    return r


def audit_trial(trial: Path) -> dict:
    hits = []
    n_calls = 0
    for f in glob.glob(str(trial / "agent" / "completions" / "*.json")):
        d = json.loads(Path(f).read_text())
        msg = ((d.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
        for tc in msg.get("tool_calls") or []:
            n_calls += 1
            args = tc.get("function", {}).get("arguments") or ""
            bad = forbidden_paths(args) + [w for w in ("sop_verifier", "rubrics.json", "task_prompt.txt") if w in args]
            if bad:
                hits.append({"tool": tc["function"]["name"], "matched": bad, "args": args[:200]})
    return {"trial": trial.name, "tool_calls": n_calls, "hits": hits, "verdict": "REJECT" if hits else "OK"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--tasks", required=True)
    c.add_argument("--out", required=True)
    k = sub.add_parser("candidate")
    k.add_argument("--corpus", required=True)
    k.add_argument("--prompt", required=True)
    k.add_argument("--base", default=None)
    t = sub.add_parser("trial")
    t.add_argument("--trial", required=True, nargs="+")
    c2 = sub.add_parser("code")
    c2.add_argument("--corpus", required=True)
    c2.add_argument("--module", required=True)
    c2.add_argument("--base", default=None, help="a module or prompt whose strings are exempt (the baseline candidate)")
    a = ap.parse_args()
    if a.cmd == "corpus":
        n = build_corpus(Path(a.tasks), Path(a.out))
        print(f"corpus: {n} documents -> {a.out}")
    elif a.cmd == "candidate":
        base = Path(a.base).read_text() if a.base else ""
        r = audit_candidate(Path(a.prompt).read_text(), Path(a.corpus), base)
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["verdict"] == "OK" else 2)
    elif a.cmd == "code":
        base = ""
        if a.base:
            bsrc = Path(a.base).read_text()
            base = code_text(bsrc)[0] if a.base.endswith(".py") else bsrc
        r = audit_code(Path(a.module).read_text(), Path(a.corpus), base)
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["verdict"] == "OK" else 2)
    else:
        bad = 0
        for tr in a.trial:
            r = audit_trial(Path(tr))
            bad += r["verdict"] != "OK"
            print(json.dumps(r))
        sys.exit(0 if not bad else 2)


if __name__ == "__main__":
    main()
