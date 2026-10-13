"""Paths the Meta-Harness add-on needs, resolved on first use.

Precedence: environment variables > a TOML file > built-in defaults. Every
driver and analysis in the add-on reads ``CFG`` instead of carrying its own
absolute paths. Relative paths resolve against the working directory.

Environment variables (all optional):
  VITALS_MH_REFERENCE_GGUF  the BF16 reference model the replays teacher-force
  VITALS_MH_DETOK_GGUF      a same-vocab small model for CPU tokenization
                            (glob allowed; first match wins)
  VITALS_MH_ROOT            meta-harness/reference_examples (the optimizer repo)
  VITALS_MH_HANDBOOK_TOOLS_JSON
                            cached OpenAI tool schemas for HANDBOOK.md trials
                            without their own tools_openai.json
  VITALS_MH_CONFIG          path to a TOML with the same keys (lowercase, no prefix)
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from glob import glob
from pathlib import Path

_DEFAULTS = {
    "reference_gguf": str(Path.home() / "models/Qwen3.6-27B-GGUF/BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf"),
    "detok_gguf": str(Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF"
                      "/snapshots/*/Qwen3.5-0.8B-BF16.gguf"),
    "root": str(Path.home() / "meta-harness/reference_examples"),
    "handbook_tools_json": "runs/handbook/tools_openai.json",
}


def _load_toml() -> dict:
    path = os.environ.get("VITALS_MH_CONFIG") or str(Path.home() / ".config" / "vitals" / "metaharness.toml")
    p = Path(path)
    if not p.exists():
        return {}
    return tomllib.loads(p.read_text())


def _resolve(key: str, toml: dict) -> str:
    env = os.environ.get(f"VITALS_MH_{key.upper()}")
    if env:
        return env
    return toml.get(key, _DEFAULTS[key])


@dataclass(frozen=True)
class MetaHarnessConfig:
    reference_gguf: Path
    detok_gguf: Path
    metaharness_root: Path
    handbook_tools_json: Path

    # optimizer environments
    @property
    def tc_env(self) -> Path:
        return self.metaharness_root / "text_classification"

    @property
    def tb_env(self) -> Path:
        return self.metaharness_root / "terminal_bench_2"

    @property
    def tb_agents(self) -> Path:
        return self.tb_env / "agents"

    @classmethod
    def load(cls) -> MetaHarnessConfig:
        toml = _load_toml()
        detok = _resolve("detok_gguf", toml)
        hits = sorted(glob(detok))
        return cls(
            reference_gguf=Path(_resolve("reference_gguf", toml)),
            detok_gguf=Path(hits[0] if hits else detok),
            metaharness_root=Path(_resolve("root", toml)),
            handbook_tools_json=Path(_resolve("handbook_tools_json", toml)),
        )

    def describe(self) -> str:
        return "\n".join(f"{k:<18} {v}" for k, v in {
            "reference_gguf": self.reference_gguf,
            "detok_gguf": self.detok_gguf, "metaharness_root": self.metaharness_root,
            "handbook_tools_json": self.handbook_tools_json}.items())


class _LazyConfig:
    """``CFG``: the configuration, read on first use (so environment variables set after import still count)."""

    _cfg: MetaHarnessConfig | None = None

    def __getattr__(self, name: str):
        if self._cfg is None:
            type(self)._cfg = MetaHarnessConfig.load()
        return getattr(self._cfg, name)

    def reload(self) -> MetaHarnessConfig:
        type(self)._cfg = MetaHarnessConfig.load()
        return self._cfg


CFG = _LazyConfig()
