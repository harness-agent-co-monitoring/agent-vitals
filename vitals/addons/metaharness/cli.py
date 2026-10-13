"""``vitals metaharness <command> [args]`` — entry points of the Meta-Harness add-on.

Each command forwards its remaining arguments to the module's own argparse
``main``; run ``vitals metaharness <command> --help`` for that module's flags.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    # name: (module, help)
    "config": ("vitals.addons.metaharness.config", "print the resolved paths (env > toml > defaults)"),
    # replay drivers (GPU)
    "replay-inloop": ("vitals.addons.metaharness.replay.inloop", "one-shot in-loop replay of the just-benchmarked systems (classification)"),
    "replay-inloop-tb": ("vitals.addons.metaharness.replay.inloop_tb", "one-shot in-loop replay of a candidate's harbor job (terminal-bench)"),
    "replay-inloop-handbook": ("vitals.addons.metaharness.replay.inloop_handbook", "one-shot in-loop replay of a candidate's harbor job (HANDBOOK.md, OpenHands completion logs)"),
    # the per-section digests for the read-informed run (CPU)
    "digest": ("vitals.addons.metaharness.digest.inloop", "per-section read lift / consult ratio digest for the proposer (classification)"),
    "digest-tb": ("vitals.addons.metaharness.digest.inloop_tb", "per-section read lift / consult ratio digest for the proposer (terminal-bench)"),
    "digest-handbook": ("vitals.addons.metaharness.digest.inloop_handbook", "per-section read digest for the proposer (HANDBOOK.md): harness sentences, handbook passages, per-trial map, delta vs parent"),
    "diff": ("vitals.addons.metaharness.digest.delta", "candidate-vs-parent read delta: token-diff the rendered prompts, lift of retained / removed / added text"),
    # terminal-bench read maps (CPU)
    "readmaps": ("vitals.addons.metaharness.analysis.tb_readmaps", "fine-grained TB read-maps: build | matrices | atlas | traj | events"),
}


def _usage() -> str:
    width = max(len(k) for k in COMMANDS)
    lines = [f"  {k:<{width}}  {h}" for k, (_, h) in COMMANDS.items()]
    return "usage: vitals metaharness <command> [args]\n\ncommands:\n" + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"vitals metaharness: unknown command {name!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module_name, _ = COMMANDS[name]
    if name == "config":
        from vitals.addons.metaharness.config import CFG
        print(CFG.describe())
        return 0
    mod = importlib.import_module(module_name)
    sys.argv = [f"vitals metaharness {name}", *rest]
    result = mod.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
