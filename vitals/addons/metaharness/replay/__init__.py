"""Replay drivers: teacher-force ingested episodes through the reference model."""

import json
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def manifest_spans(ep: Path) -> list[dict] | None:
    """The span requests ingest stored in an episode's manifest, or None."""
    try:
        return json.loads((ep / "manifest.json").read_text())["meta"].get("prompt_sections_requests")
    except (OSError, json.JSONDecodeError, KeyError):
        return None
