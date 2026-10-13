"""The command line: every command's help renders, the top-level help lists every command, user errors exit 2
with a message instead of a traceback, and the CPU-only modules import without the optional extras."""

import subprocess
import sys

import pytest

from vitals.cli import build_parser, main

COMMANDS = ["import", "serve", "replay", "report", "dashboard", "annotate", "monitor", "ledger", "probe", "pipeline",
            "explain", "sensitivity", "sections", "gate", "calibrate", "engine", "metaharness"]


def test_top_level_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert all(c in out for c in COMMANDS)


@pytest.mark.parametrize("argv", [[c, "--help"] for c in COMMANDS if c != "metaharness"]
                         + [["probe", "train", "--help"], ["probe", "score", "--help"], ["engine", "doctor", "--help"]])
def test_every_help_renders(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 0 and "usage: vitals" in capsys.readouterr().out


def test_old_flag_spellings_still_parse():
    args = build_parser().parse_args(["explain", "--store", "s", "--episode", "e", "--gguf", "g", "--top-k", "2"])
    assert args.top_k == 2


def test_user_errors_exit_2_without_a_traceback(tmp_path, capsys):
    assert main(["probe", "score", "--store", str(tmp_path), "--out-dir", str(tmp_path / "none"), "--probe", "p.json"]) == 2
    err = capsys.readouterr().err
    assert "vitals probe: error:" in err and "Traceback" not in err


def test_module_entry_point():
    r = subprocess.run([sys.executable, "-m", "vitals.cli", "--version"], capture_output=True, text=True, check=True)
    assert r.stdout.startswith("vitals ")


BLOCK = """
import importlib, sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {%s}:
            raise ModuleNotFoundError(f"No module named {name!r}")
sys.meta_path.insert(0, Block())
for m in %r:
    importlib.import_module(m)
"""


def test_cpu_modules_import_without_torch():
    mods = ["vitals.cli", "vitals.monitor.checks", "vitals.monitor.turns", "vitals.monitor.flags", "vitals.monitor.ledger",
            "vitals.monitor.pipeline", "vitals.dashboard.data", "vitals.analysis.readmap", "vitals.analysis.drift",
            "vitals.explain", "vitals.replay.importer"]
    code = BLOCK % ('"torch"', mods)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_cli_reference_is_current():
    """docs/reference/cli.md is generated from the parsers: regenerate it (scripts/gen_cli_reference.py) after a
    change to any command's arguments or help."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("gen_cli_reference", root / "scripts" / "gen_cli_reference.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert (root / "docs" / "reference" / "cli.md").read_text() == gen.render(), \
        "docs/reference/cli.md is stale: run python scripts/gen_cli_reference.py"
