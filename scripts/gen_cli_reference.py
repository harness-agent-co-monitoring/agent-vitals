"""Write docs/reference/cli.md from the argparse help of every command: python scripts/gen_cli_reference.py"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "reference" / "cli.md"
COMMANDS = [["import"], ["serve"], ["replay"], ["report"], ["dashboard"], ["annotate"], ["monitor"], ["ledger"],
            ["probe", "train"], ["probe", "score"], ["pipeline"], ["explain"], ["sensitivity"], ["sections"],
            ["gate"], ["calibrate"], ["engine", "doctor"]]


def help_text(argv: list[str]) -> str:
    from vitals.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        main([*argv, "--help"])
    return buf.getvalue().rstrip()


def render() -> str:
    old = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = "110"            # argparse wraps to the terminal width
    try:
        return _render()
    finally:
        if old is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = old


def _render() -> str:
    parts = ["# Command line\n", "Generated from the argparse help by `scripts/gen_cli_reference.py`; do not edit.\n",
             "```text", help_text([]), "```\n"]
    for argv in COMMANDS:
        parts += [f"## vitals {' '.join(argv)}\n", "```text", help_text(argv), "```\n"]
    parts += ["## vitals metaharness\n", "The Meta-Harness add-on has its own commands: `vitals metaharness --help` and the "
              "[add-on guide](../../vitals/addons/metaharness/README.md).\n"]
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}", file=sys.stderr)
