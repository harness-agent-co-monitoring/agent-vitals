"""agent-vitals: section-level read maps, monitoring and decision explanations for LLM agent harnesses.

Get episodes into a trace store with `vitals import` (logged calls) or `vitals serve` (live), recover attention
with `vitals replay`, then read the store with `vitals report` and `vitals dashboard`, and monitor it with
`vitals monitor` (checks, no model pass) or `vitals pipeline` (ledger, probes, explanations, webhook).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from vitals import SCHEMA_VERSION, __version__
from vitals.cli import analyze, capture, monitor

# in the order a new user meets them
_COMMANDS = (capture.add_import, capture.add_serve, capture.add_replay, analyze.add_report, analyze.add_dashboard,
             analyze.add_annotate, monitor.add_monitor, monitor.add_ledger, monitor.add_probe, monitor.add_pipeline,
             monitor.add_explain, analyze.add_sensitivity, analyze.add_sections, analyze.add_gate,
             analyze.add_calibrate, capture.add_engine)

# optional dependencies and the extra that installs them
_EXTRAS = {"torch": "gpu", "fastapi": "server", "uvicorn": "server", "gepa": "gepa"}


def build_parser() -> argparse.ArgumentParser:
    """The ``vitals`` argument parser with every command."""
    p = argparse.ArgumentParser(prog="vitals", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"vitals {__version__} (schema {SCHEMA_VERSION})")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    for add in _COMMANDS:
        add(sub)
    # listed for --help only: main() hands "metaharness" argv to the add-on before parsing
    sub.add_parser("metaharness", add_help=False,
                   help="Meta-Harness add-on: ingestion, replay drivers, digests, section tables "
                        "and read maps (vitals metaharness --help)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run one command. User errors (a missing file, a bad argument value, a missing optional dependency) print one
    line and return 2; set ``VITALS_DEBUG=1`` for the traceback."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "metaharness":  # the add-on owns its own argv (incl. --help)
        from vitals.addons.metaharness.cli import main as mh_main

        return mh_main(argv[1:])
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if os.environ.get("VITALS_DEBUG") else logging.WARNING,
                        format="%(name)s: %(levelname)s: %(message)s")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as e:
        if os.environ.get("VITALS_DEBUG"):
            raise
        missing = getattr(e, "name", None)
        if isinstance(e, ModuleNotFoundError) and missing:
            top = missing.split(".")[0]
            hint = f"pip install 'agent-vitals[{_EXTRAS[top]}]'" if top in _EXTRAS else f"install {top}"
            print(f"vitals {args.cmd}: needs the {top} package: {hint}", file=sys.stderr)
        else:
            print(f"vitals {args.cmd}: error: {e} (set VITALS_DEBUG=1 for the traceback)", file=sys.stderr)
        return 2
