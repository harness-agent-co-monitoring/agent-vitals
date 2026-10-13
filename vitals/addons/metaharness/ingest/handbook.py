"""HANDBOOK.md (Harbor + OpenHands) trials -> Vitals store, from the runner's
completion logs (`--ak log_completions=true`): one JSON per LLM call holding
the exact request (`messages`, OpenHands tool definitions) and the response
(`reasoning_content`, `content`, `tool_calls`). One trial = one episode, one
call = one turn. The prompt is the logged message list (text blocks
flattened, which is what the chat template does) rendered with the OpenAI
tool schemas the SDK derives from the logged definitions
(`agent/tools_openai.json`, produced in-image by
`ToolDefinition.model_validate(t).to_openai_tool()`).

Spans: `sys:*` (the system prompt sentence by sentence, or the candidate's own
declaration), `instruction`, `tool:<name>` (one per schema line), per message
`user:k` / `self:k` / `obs:k:<tool>` plus declared `fb:k:<name>` feedback
spans, and for any tool result that is a long document (the handbook, read via
readPDF/readFile/executePython) its sections `doc:k/<heading>` (else
fixed-size passages) pinned inside that result.

  python -m vitals.addons.metaharness.ingest.handbook --job <harbor job dir> \
      --store runs/handbook/store [--max-prompt-tokens 200000]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from vitals.addons.metaharness.config import CFG
from vitals.addons.metaharness.ingest import detok_engine
from vitals.addons.metaharness.ingest.chat import MIN_SPAN, tool_spans
from vitals.core.schema import EpisodeManifest
from vitals.core.store import TraceStore
from vitals.engine.template import ChatTemplate, template_from_gguf
from vitals.replay import append_forced, forced_text, strip_end_of_turn

DOC_MIN_CHARS = 15000
DOC_MIN_HEADINGS = 5


def flatten_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in (content or []) if isinstance(b, dict))


def _tool_calls(tcs: list[dict]) -> list[dict]:
    return [{"id": tc.get("id"), "type": "function",
             "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}}
            for tc in tcs]


def flatten_message(m: dict) -> dict:
    out = {"role": m["role"], "content": flatten_text(m.get("content"))}
    if m.get("tool_calls"):
        out["tool_calls"] = _tool_calls(m["tool_calls"])
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    if m.get("name"):
        out["name"] = m["name"]
    return out


def load_calls(trial: Path) -> list[dict]:
    docs = []
    for f in (trial / "agent" / "completions").glob("*.json"):
        d = json.loads(f.read_text())
        if "response" in d and d["response"].get("choices"):
            docs.append(d)
    docs.sort(key=lambda d: d.get("timestamp") or 0)
    return docs


_HTML_HEAD = re.compile(r"<h([1-3])[^>]*>(.*?)</h\1>", re.S | re.I)
_NUM_HEAD = re.compile(r"(?<![\d.$])(\d{1,2})\.\s{1,4}(?=[A-Z])")


def _clean(head: str) -> str:
    head = re.sub(r"<[^>]+>", "", head).replace("\\n", " ")
    return re.sub(r"\s+", " ", head).strip()[:48]


_LOWER_OK = {"and", "of", "the", "for", "to", "in", "on", "with", "a", "an", "vs", "or", "at", "by"}


def _is_heading(text: str, pos: int) -> bool:
    """A numbered heading reads as a short Title-Case phrase; a numbered list
    item reads as a sentence ('25. When in doubt about...', '1. Triage intake:')."""
    title = re.split(r"\\n|\n", text[pos:pos + 60])[0]
    body = re.sub(r"^\d{1,2}\.\s+", "", title)
    if ":" in body[:45]:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z&'’-]*", body)[:5]
    if not words:
        return False
    lower = [w for w in words if w[0].islower() and w.lower() not in _LOWER_OK]
    return len(lower) < 2


def _numbered_run(text: str) -> list[tuple[int, str]]:
    """Forward-greedy runs 1., 2., 3., ... over heading-shaped candidates; the
    table of contents forms a short-coverage run, the body the long one."""
    cands = [(int(m.group(1)), m.start()) for m in _NUM_HEAD.finditer(text) if _is_heading(text, m.start())]
    best: list[tuple[int, str]] = []
    best_score = (0, 0)
    for first in [p for n, p in cands if n == 1]:      # a run from every '1.' (ToC and body both start at 1)
        run = [first]
        n = 2
        while True:
            nxt = [p for k, p in cands if k == n and p > run[-1]]
            if not nxt:
                break
            run.append(nxt[0])
            n += 1
        if len(run) < DOC_MIN_HEADINGS:
            continue
        ends = run[1:] + [len(text)]
        score = (sum(1 for a, e in zip(run, ends, strict=True) if e - a >= 300), run[-1] - run[0])
        if score > best_score:   # most real (>= 300-char) sections, then extent
            best_score = score
            best = [(pos, _clean(re.split(r"\\n|\n", text[pos:pos + 70])[0])) for pos in run]
    return best


SYS_NAMES = ["sys:date", "sys:role", "sys:workspace", "sys:no-ask", "sys:begin"]
PARA_CHARS = 1500


def system_spans(text: str) -> list[tuple[str, str]]:
    """The office-assistant prompt sentence by sentence (the units a
    meta-harness edits): date, role+tools, workspace, do-not-ask, begin-by."""
    sents = [x.strip() for x in re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip()) if x.strip()]
    names = SYS_NAMES if len(sents) == len(SYS_NAMES) else [f"sys:{i}" for i in range(len(sents))]
    return list(zip(names, sents, strict=True))


def _subsections(n: int, chunk: str, head: str) -> list[tuple[str, str]]:
    """Split a top-level section at its 'N.1 ', 'N.2 ', ... headings (monotone);
    the text before the first sub-heading keeps the section's own name."""
    pat = re.compile(rf"(?<![\d.]){n}\.(\d{{1,2}})\s+(?=[A-Z])")
    picks, want = [], 1
    for m in pat.finditer(chunk):
        if int(m.group(1)) == want and _is_heading(chunk, m.start()):
            picks.append(m.start())
            want += 1
    if len(picks) < 2:
        return [(head, chunk)]
    out = []
    bounds = [0] + picks + [len(chunk)]
    for i in range(len(bounds) - 1):
        part = chunk[bounds[i]:bounds[i + 1]].rstrip("\\n \n")
        if len(part) < 200:
            continue
        name = head if i == 0 else _clean(re.split(r"\\n|\n", part[:70])[0])
        out.append((name, part))
    return out


def paragraph_chunks(text: str, size: int = PARA_CHARS) -> list[tuple[str, str]]:
    """Fixed-size passages at pseudo-newline boundaries for documents whose
    headings the detector cannot find (inline-heading PDFs, XML dumps)."""
    parts = re.split(r"(\\n|\n)", text)
    out, cur = [], ""
    for part in parts:
        cur += part
        if len(cur) >= size:
            out.append(cur)
            cur = ""
    if cur.strip():
        if out and len(cur) < 300:
            out[-1] += cur
        else:
            out.append(cur)
    return [(f"¶{i}", c) for i, c in enumerate(out) if c.strip()]


def doc_sections(text: str) -> list[tuple[str, str]]:
    """Top-level sections of a document as it appears in the tool result
    (raw: JSON-escaped newlines, HTML tags, PDF text with inline headings).
    HTML: <h1-3> elements. Otherwise: the numbered headings '1. Title',
    '2. Title', ... taken as the last monotone run (a table of contents lists
    the same numbers first; the body's occurrences come later). Chunks under
    300 chars are dropped."""
    heads: list[tuple[int, str]] = []
    if "<html" in text[:400].lower():
        heads = [(m.start(), _clean(m.group(2))) for m in _HTML_HEAD.finditer(text)]
    else:
        heads = _numbered_run(text)
    if len(heads) < DOC_MIN_HEADINGS:
        return []
    out = []
    for i, (pos, head) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        chunk = text[pos:end].rstrip("\\n \n")
        if len(chunk) < 300:
            continue
        m = re.match(r"(\d{1,2})\.", chunk)
        if m:
            out += _subsections(int(m.group(1)), chunk, head or f"s{i}")
        else:
            out.append((head or f"s{i}", chunk))
    return out


def turn_action(resp_msg: dict) -> dict:
    calls = [tc["function"]["name"] for tc in (resp_msg.get("tool_calls") or [])]
    return {"action": calls[0] if calls else "message", "calls": calls,
            "says": bool((resp_msg.get("content") or "").strip()),
            "reasoning_chars": len(resp_msg.get("reasoning_content") or "")}


def declared_system_spans(system_prompt: str, declared) -> list[tuple[str, str]] | None:
    """A candidate's own prompt_sections() declaration, accepted only when
    every span is verbatim in the rendered prompt and there are >= 2 of them;
    names get the sys: prefix the digest groups on."""
    if not declared:
        return None
    try:
        secs = list(declared) if not callable(declared) else declared(system_prompt)
    except Exception:
        return None
    if not secs or len(secs) < 2:
        return None
    out = []
    for name, text in secs:
        text = str(text).strip()
        if not text or text not in system_prompt:
            return None
        name = str(name)
        out.append((name if name.startswith("sys:") else f"sys:{name}", text))
    return out


def feedback_spans(text: str, feedback_fn) -> list[tuple[str, str]]:
    """A candidate's `feedback_sections(observation_text)`: harness-authored
    spans inside one tool result / user message, verbatim or dropped."""
    if not callable(feedback_fn):
        return []
    try:
        out = feedback_fn(text) or []
    except Exception:
        return []
    return [(str(n), str(t)) for n, t in out if str(t) and str(t) in text]


def build_spans(msgs: list[dict], tools_spans: list[tuple[str, str]], system_sections=None, feedback_fn=None) -> list[dict]:
    sys_secs = declared_system_spans(msgs[0]["content"], system_sections) or system_spans(msgs[0]["content"])
    spans = [{"name": n, "text": t, "msg_index": 0} for n, t in sys_secs]
    spans += [{"name": n, "text": t, "msg_index": 0} for n, t in tools_spans]
    seen: dict[str, int] = {}
    for k, m in enumerate(msgs[1:], 1):
        text = str(m.get("content") or "").strip()
        if len(text) < MIN_SPAN:
            continue
        role = m["role"]
        if role == "user":
            name = "instruction" if k == 1 else f"user:{k}"
        elif role == "assistant":
            name = f"self:{k}"
        elif role == "tool":
            name = f"obs:{k}:{str(m.get('name') or 'tool').split('__')[-1][:24]}"
        else:
            continue
        occ = seen.get(text, 0) + 1
        seen[text] = occ
        if role in ("tool", "user"):
            for fb_name, fb_text in feedback_spans(text, feedback_fn):
                spans.append({"name": f"fb:{k}:{fb_name}", "text": fb_text, "msg_index": k,
                              **({"occurrence": occ} if occ > 1 else {})})
        if role == "tool" and len(text) >= DOC_MIN_CHARS:
            # a document: flat passages replace the whole-result span (no
            # double counting): numbered (sub)sections, else fixed paragraphs
            secs = doc_sections(text) or paragraph_chunks(text)
            for head, chunk in secs:
                spans.append({"name": f"doc:{k}/{head}", "text": chunk, "msg_index": k,
                              **({"occurrence": occ} if occ > 1 else {})})
            continue
        spans.append({"name": name, "text": text, "msg_index": k, **({"occurrence": occ} if occ > 1 else {})})
    return spans


HB_IMAGE = os.environ.get("HANDBOOK_IMAGE", "handbook_base")
HB_RUNNER_PYTHON = "/app/openhands-runner/.venv/bin/python"


def logged_tool_names(logged: list[dict]) -> list[str]:
    """Function names of the tool definitions logged with a call, in the
    order the SDK sent them (MCP tools carry the name in `mcp_tool.name`;
    `title` is a display title for some of them; built-ins have only title)."""
    return [str((t.get("mcp_tool") or {}).get("name") or t.get("title")) for t in logged]


def convert_tools_in_image(trial: Path) -> list[dict] | None:
    """Exact OpenAI schemas via the SDK's own converter, run inside the
    benchmark image (ToolDefinition.model_validate(t).to_openai_tool()).
    Writes agent/tools_openai.json into the trial; None if docker/image fail."""
    comp = (trial / "agent" / "completions").resolve()
    code = (
        "import json,glob\n"
        "from openhands.sdk.tool import ToolDefinition\n"
        "for f in sorted(glob.glob('/c/*.json')):\n"
        "    d=json.load(open(f))\n"
        "    if d.get('tools'):\n"
        "        print(json.dumps([ToolDefinition.model_validate(t).to_openai_tool() for t in d['tools']])); break\n"
    )
    try:
        r = subprocess.run(["docker", "run", "--rm", "-e", "OPENHANDS_SUPPRESS_BANNER=1", "-v", f"{comp}:/c:ro",
                            HB_IMAGE, HB_RUNNER_PYTHON, "-c", code], capture_output=True, text=True, timeout=300)
        line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("[")), None)
        tools = json.loads(line) if line else None
    except Exception:
        tools = None
    if tools:
        (trial / "agent" / "tools_openai.json").write_text(json.dumps(tools))
    return tools


def load_tools(trial: Path, calls: list[dict], tools_json: Path | None = None) -> list[dict] | None:
    """The OpenAI schemas the SDK sent, in the logged order: the trial's own
    `tools_openai.json`; else the cached list re-ordered to the logged names
    when every name is cached; else the SDK converter run in the image."""
    own = trial / "agent" / "tools_openai.json"
    if own.exists():
        return json.loads(own.read_text())
    logged = next((d["tools"] for d in calls if d.get("tools")), None)
    if logged is None:
        return None
    names = logged_tool_names(logged)
    if tools_json is not None and Path(tools_json).exists():
        cached = {x["function"]["name"]: x for x in json.loads(Path(tools_json).read_text())}
        if all(n in cached for n in names):
            return [cached[n] for n in names]
    return convert_tools_in_image(trial)


def ingest_trial(trial: Path, store: TraceStore, eid: str, tpl: ChatTemplate, eng,
                 meta_extra: dict, max_prompt_tokens: int, tools_json: Path | None = None,
                 system_sections=None, feedback_fn=None) -> dict | None:
    calls = load_calls(trial)
    if not calls:
        return None
    tools = load_tools(trial, calls, tools_json)
    if tools is None:
        return {"error": "no tool schemas (tools_openai.json missing and the cached list does not match)"}
    final = [flatten_message(m) for m in max(calls, key=lambda d: len(d["messages"]))["messages"]]
    # every call's prompt must be a prefix of the final one (no condensation)
    for d in calls:
        ms = [flatten_message(m) for m in d["messages"]]
        if ms != final[:len(ms)]:
            return {"error": "prompt not a prefix of the final prompt (condensation?)"}
    tspans = tool_spans(tpl, final[0]["content"], tools)
    spans = build_spans(final, tspans, system_sections, feedback_fn)
    actions, skipped, wrote = [], [], 0
    for t, d in enumerate(calls):
        r = d["response"]["choices"][0]["message"]
        actions.append(turn_action(r))
        pt = (d["response"].get("usage") or {}).get("prompt_tokens") or 0
        if pt > max_prompt_tokens:
            skipped.append(t)
            continue
        msgs = [flatten_message(m) for m in d["messages"]]
        a = {"role": "assistant", "content": r.get("content") or ""}
        if r.get("tool_calls"):
            a["tool_calls"] = _tool_calls(r["tool_calls"])
        a["reasoning_content"] = r.get("reasoning_content") or ""
        forced = forced_text(tpl, msgs, a, tools=tools)
        toks = strip_end_of_turn(eng, eng.tokenize(forced, add_special=False))
        if not toks:
            skipped.append(t)
            continue
        store.write_messages(eid, t, msgs, tools=tools)
        append_forced(store, eid, t, toks)
        wrote += 1
    if not wrote:
        return None
    store.flush(eid)
    meta = {"task": "handbook", "provider_generated": True, "memory": meta_extra.get("memory", "base"),
            "n_turns_total": len(calls), "n_turns_replayed": wrote, "skipped_turns": skipped,
            "turn_actions": actions, "n_doc_sections": sum(s["name"].startswith("doc:") for s in spans),
            "system_sections_declared": bool(declared_system_spans(final[0]["content"], system_sections)),
            "prompt_sections_requests": spans, **meta_extra}
    store.write_manifest(EpisodeManifest(episode_id=eid, meta=meta))
    return meta


def trial_meta(trial: Path) -> dict:
    name = None
    for f in ("config.json", "result.json"):
        p = trial / f
        if p.exists():
            m = re.search(r"sop-tasks/([a-z0-9_]+)", p.read_text())
            if m:
                name = m.group(1)
                break
    name = name or trial.name.split("__")[0]
    rp = trial / "verifier" / "reward.txt"
    reward = float(rp.read_text().strip()) if rp.exists() else None
    parts = name.split("_")
    return {"example_idx": name, "industry": parts[0], "handbook": "_".join(parts[:-1]),
            "reward": reward, "strict_pass": bool(reward is not None and reward >= 0.999),
            "trial_dir": str(trial)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--gguf", default=str(CFG.reference_gguf))
    ap.add_argument("--max-prompt-tokens", type=int, default=200_000)
    ap.add_argument("--tools-json", default=None, help="cached OpenAI tool schemas for trials without tools_openai.json")
    args = ap.parse_args()

    tpl = ChatTemplate(template_from_gguf(args.gguf))
    eng = detok_engine()
    store = TraceStore(args.store)
    n_ok = 0
    for trial in sorted(p for p in Path(args.job).iterdir() if "__" in p.name and (p / "agent").is_dir()):
        meta_extra = trial_meta(trial)
        eid = meta_extra["example_idx"]
        if (Path(args.store) / eid / "manifest.json").exists():
            print(f"skip {eid}: exists")
            continue
        try:
            meta = ingest_trial(trial, store, eid, tpl, eng, meta_extra, args.max_prompt_tokens,
                                tools_json=Path(args.tools_json) if args.tools_json else None)
        except ValueError as exc:
            print(f"SKIP {eid}: {exc}")
            continue
        if meta is None or "error" in meta:
            print(f"SKIP {eid}: {(meta or {}).get('error', 'no calls')}")
            continue
        n_ok += 1
        print(f"{eid}: {meta['n_turns_replayed']}/{meta['n_turns_total']} turns"
              f"{' (skipped ' + str(meta['skipped_turns']) + ')' if meta['skipped_turns'] else ''}, "
              f"reward {meta['reward']}, {len(meta['prompt_sections_requests'])} spans, {meta['n_doc_sections']} doc sections")
    print(f"ingested {n_ok} episodes -> {args.store}")


if __name__ == "__main__":
    main()
