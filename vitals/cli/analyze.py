"""Commands that read a replayed store: ``report``, ``dashboard``, ``annotate``, ``sections``, ``sensitivity``,
``gate`` and ``calibrate``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vitals.cli._common import add_cache, add_model, add_store, load_backend
from vitals.core.messages import content_text


def add_report(sub) -> None:
    p = sub.add_parser("report", help="read map of a trace store: per turn and section (declared or automatic), read "
                                      "lift, peak lift with its hottest line, consult ratio and contrast")
    add_store(p)
    p.add_argument("--json", action="store_true", help="print JSON rows instead of a table")
    p.add_argument("--out", default="", help="write the JSON rows here")
    p.add_argument("--diagnostics", action="store_true",
                   help="include consult decomposition, both lift averages, and captured per-step peaks "
                        "and focus JS")
    p.set_defaults(func=run_report)


def run_report(args: argparse.Namespace) -> int:
    from vitals.analysis.readmap import format_readmap, store_readmap

    rows = store_readmap(args.store, diagnostics=args.diagnostics)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2) if args.json else format_readmap(rows))
    if args.diagnostics and not args.json:
        from vitals.analysis.diagnostics import format_diagnostics

        print("\n" + format_diagnostics(rows))
    return 0


def add_dashboard(sub) -> None:
    p = sub.add_parser("dashboard", help="local dashboard over the trace stores and runs found under ROOT: read maps, "
                                         "store overview, drift monitor, pipeline alarms, decision explanations and "
                                         "Meta-Harness evolution")
    p.add_argument("roots", nargs="+", metavar="ROOT", help="trace stores, Meta-Harness run directories "
                   "(logs/<run>), or directories to search for them")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8040, help="port (default 8040)")
    p.add_argument("--cache-dir", default=None, help="keep cached read maps here instead of beside each episode "
                   "(default: $VITALS_READMAP_CACHE, else the episode directory when writable)")
    add_cache(p, "explanations", "--explain-dir", "decision explanations written by vitals explain")
    add_cache(p, "alarms", "--alarm-dir", "flags and explanations written by vitals pipeline")
    p.set_defaults(func=run_dashboard)


def run_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    from vitals.dashboard.app import create_app
    from vitals.dashboard.data import Dashboard

    dash = Dashboard(args.roots, cache_dir=args.cache_dir, explain_dir=args.explain_dir, alarm_dir=args.alarm_dir)
    dash.watch()
    print(f"vitals dashboard: {len(dash.stores)} stores, {len(dash.runs)} runs | http://{args.host}:{args.port}/")
    uvicorn.run(create_app(dash), host=args.host, port=args.port, log_level="warning")
    return 0


def add_annotate(sub) -> None:
    p = sub.add_parser("annotate", help="declare sections for a static prompt file (Markdown headings, paragraphs "
                                        "or inline markers), or check a declaration against a prompt or a stored "
                                        "episode")
    p.add_argument("file", help="the prompt file to annotate; with --check, the prompt file or episode directory "
                   "to check against")
    p.add_argument("--by", choices=["heading", "paragraph", "marker"], default="heading",
                   help="one section per heading block, per paragraph, or per <!-- section: name --> marker")
    p.add_argument("--template", action="store_true", help="the file is a prompt template: split sections at "
                   "placeholders ({{ x }}, {%% ... %%}, {x}, ${x}) and declare only the fixed text")
    p.add_argument("--out", help="write the sections JSON here (default: stdout)")
    p.add_argument("--clean-out", help="with --by marker: write the prompt with the markers removed here")
    p.add_argument("--check", metavar="SECTIONS", help="check this sections JSON against FILE instead of annotating")
    p.add_argument("--gguf", help="with an episode directory: render its last stored request with this model's "
                   "chat template (otherwise the message texts are checked)")
    p.set_defaults(func=run_annotate)


def run_annotate(args: argparse.Namespace) -> int:
    from vitals.core.annotate import check_sections, sections_from_markdown, sections_from_markers

    target = Path(args.file)
    if args.check:
        sections = json.loads(Path(args.check).read_text())
        if target.is_dir():
            records = [json.loads(line) for line in (target / "messages.jsonl").read_text().splitlines() if line.strip()]
            rec = records[-1]
            texts = [content_text(m) for m in rec["messages"]]
            if args.gguf:
                from vitals.engine.template import ChatTemplate, template_from_gguf

                prompt = ChatTemplate(template_from_gguf(args.gguf)).render(
                    rec["messages"], tools=rec.get("tools"), add_generation_prompt=True, **(rec.get("chat_kwargs") or {}))
            else:
                prompt = "\n\n".join(texts)
            report = check_sections(sections, prompt, texts)
        else:
            report = check_sections(sections, target.read_text())
        for row in report["sections"]:
            print(f"{row['status']:<10} {row['name']}")
        print(f"coverage {report['coverage']:.0%} of the prompt")
        return 1 if report["missing"] else 0
    text = target.read_text()
    if args.by == "marker":
        text, sections = sections_from_markers(text, template=args.template)
        if args.clean_out:
            Path(args.clean_out).write_text(text)
        else:
            print("note: serve the prompt without the markers (write it with --clean-out)", file=sys.stderr)
    else:
        sections = sections_from_markdown(text, by=args.by, template=args.template)
    out = json.dumps(sections, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(out)
    else:
        sys.stdout.write(out)
    return 0


def add_sections(sub) -> None:
    p = sub.add_parser("sections", help="compare section reading in failed and successful episodes (needs manifest "
                                        "meta.reward), and label sections by their relation to the answer")
    p.add_argument("root", help="a store dir or a replay root of <name>/store dirs")
    p.add_argument("--min-errors", type=int, default=15,
                   help="minimum rows (episode turns) from failed episodes per section contrast")
    p.add_argument("--min-correct", type=int, default=30,
                   help="minimum rows (episode turns) from successful episodes per section contrast")
    p.add_argument("--top", type=int, default=12, help="sections listed per dataset")
    p.add_argument("--match", choices=["exact", "contains"], default="exact",
                   help="label-to-answer matching for the gradient")
    p.set_defaults(func=run_sections)


def run_sections(args: argparse.Namespace) -> int:
    from vitals.analysis.sections import report

    print(report(args.root, min_errors=args.min_errors, min_correct=args.min_correct, top=args.top, match=args.match))
    return 0


def add_sensitivity(sub) -> None:
    p = sub.add_parser("sensitivity", help="compare output distributions after a controlled section replacement")
    add_store(p)
    p.add_argument("--episode", required=True, help="episode id")
    p.add_argument("--turn", type=int, required=True, help="turn index")
    p.add_argument("--section", required=True, help="declared section name (with or without span:)")
    p.add_argument("--steps", required=True, help="comma-separated output positions, e.g. 0,5,12")
    p.add_argument("--replacements", required=True,
                   help="JSON file mapping condition names to text or {token_ids: [...]}")
    add_model(p, n_ctx=32768)
    p.add_argument("--out", default="", help="write the JSON result here")
    p.set_defaults(func=run_sensitivity)


def run_sensitivity(args: argparse.Namespace) -> int:
    from vitals.probes.sensitivity import sensitivity_from_store

    replacements = json.loads(Path(args.replacements).read_text())
    if not isinstance(replacements, dict) or not replacements:
        raise ValueError("--replacements must contain a nonempty JSON object")
    try:
        steps = [int(t) for t in args.steps.split(",")]
    except ValueError:
        raise ValueError("--steps must contain comma-separated integers") from None
    backend = load_backend(args)
    try:
        result = sensitivity_from_store(backend, args.store, args.episode, turn=args.turn,
                                         section=args.section, replacements=replacements, steps=steps)
    finally:
        backend.close()
    result["reference_gguf"] = str(Path(args.gguf).resolve())
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(output + "\n")
    print(output)
    return 0


def add_gate(sub) -> None:
    p = sub.add_parser("gate", help="check that flash attention on and off produce the same greedy tokens (the "
                                    "diagnostic attention path)")
    p.add_argument("which", choices=["fa"], help="the gate to run")
    p.add_argument("--gguf", required=True, help="model GGUF")
    p.add_argument("--prompt-file", required=True, help="text file with the prompt to continue")
    p.add_argument("--n", type=int, default=200, help="greedy tokens to compare")
    p.add_argument("--n-ctx", type=int, default=16384, help="context window in tokens")
    p.add_argument("--out", default="", help="write the JSON report here")
    p.set_defaults(func=run_gate)


def run_gate(args: argparse.Namespace) -> int:
    from vitals.probes import fa_equivalence

    return _print_report(fa_equivalence(args.gguf, Path(args.prompt_file).read_text(), n=args.n, n_ctx=args.n_ctx),
                         args.out)


def add_calibrate(sub) -> None:
    p = sub.add_parser("calibrate", help="per-model head census: retrieval heads, induction heads, position bias "
                                         "(one-time per model; not the explain --weights file)")
    p.add_argument("which", choices=["retrieval-heads", "induction-heads", "position-bias"], help="the census to run")
    p.add_argument("--gguf", required=True, help="model GGUF")
    p.add_argument("--n-ctx", type=int, default=4096, help="context window in tokens")
    p.add_argument("--out", default="", help="write the JSON report here")
    p.set_defaults(func=run_calibrate)


def run_calibrate(args: argparse.Namespace) -> int:
    from vitals.probes import induction_heads, position_bias, retrieval_heads

    fn = {"retrieval-heads": retrieval_heads, "induction-heads": induction_heads,
          "position-bias": position_bias}[args.which]
    return _print_report(fn(args.gguf, n_ctx=args.n_ctx), args.out)


def _print_report(rep: dict, out: str) -> int:
    text = json.dumps(rep, indent=2)
    if out:
        Path(out).write_text(text)
    print(text)
    return 0
