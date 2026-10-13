"""Trace store: parquet for per-token scalars, zarr for activations, JSON manifests.

The parquet schema is the contract with downstream analysis. ``schema_version``
gates migrations; refuse silently-degraded writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vitals import SCHEMA_VERSION
from vitals.core.io import atomic_write_text
from vitals.core.schema import EpisodeManifest, TokenMetrics

TOKEN_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string()),
        pa.field("turn", pa.int32()),
        pa.field("step", pa.int32()),
        pa.field("token_id", pa.int32()),
        pa.field("ent", pa.float32()),
        pa.field("m2", pa.float32()),
        pa.field("js", pa.float32()),
        pa.field("fork", pa.bool_()),
        pa.field("kl_ref", pa.float32(), nullable=True),
        pa.field("region_id", pa.string()),
        pa.field("attn_share", pa.list_(pa.float32()), nullable=True),
        pa.field("probes", pa.map_(pa.string(), pa.float32())),
        pa.field("t_ms", pa.float32()),
        # v0.2.0 columns (all nullable; None when top_logprobs capture is off)
        pa.field("surprisal", pa.float32(), nullable=True),
        pa.field("top_ids", pa.list_(pa.int32()), nullable=True),
        pa.field("top_lp", pa.list_(pa.float32()), nullable=True),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode()},
)


def attn_tokens_frame(turn: int, token_ids, token_strs, regions: list, recv):
    """One turn's per-prompt-token attention table as a DataFrame (see
    write_attn_tokens)."""
    import pandas as pd

    n = len(token_ids)
    region_of = [None] * n

    def _get(r, k):
        return getattr(r, k) if hasattr(r, k) else r.get(k)

    for r in sorted(regions, key=lambda r: str(_get(r, "id")).startswith("span:")):
        s, e = _get(r, "token_start"), _get(r, "token_end")
        if s is None or e is None:
            continue
        rid = str(_get(r, "id"))
        for i in range(max(0, int(s)), min(n, int(e))):
            region_of[i] = rid
    return pd.DataFrame({
        "turn": [turn] * n,
        "idx": list(range(n)),
        "token_id": [int(t) for t in token_ids],
        "token_str": list(token_strs),
        "region_id": region_of,
        "attn": [round(float(recv[i]), 8) for i in range(n)],
    })


def write_attn_tokens_turns(ep_dir: str | Path, turns: dict) -> None:
    """Write several turns' per-prompt-token attention tables to
    attn_tokens.parquet in one read-modify-write. ``turns`` maps a turn index
    to ``(token_ids, token_strs, regions, recv)``; each listed turn replaces
    its existing rows."""
    import pandas as pd

    if not turns:
        return
    frames = [attn_tokens_frame(t, *args) for t, args in sorted(turns.items())]
    path = Path(ep_dir) / "attn_tokens.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        frames.insert(0, old[~old.turn.isin(list(turns))])
    tmp = path.with_name(path.name + ".tmp")
    pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_attn_tokens(ep_dir: str | Path, turn: int, token_ids, token_strs, regions: list, recv) -> None:
    """Append one turn's per-prompt-token attention table to attn_tokens.parquet:
    each prompt token with its string, the most specific region it belongs to
    (a ``span:`` section over its enclosing message region), and the attention
    it received from the turn's generated tokens. Re-writing a turn replaces
    its rows. ``regions`` may be Region objects or their dict form."""
    write_attn_tokens_turns(ep_dir, {turn: (token_ids, token_strs, regions, recv)})


def check_episode_id(episode_id: str) -> str:
    """An episode id names one directory inside the store: not empty, not ``.`` or ``..``, no path separators or
    NUL. Raises ValueError otherwise (ids reach the store from request headers and log files)."""
    if not isinstance(episode_id, str) or episode_id in ("", ".", "..") or any(c in episode_id for c in "/\\\0"):
        raise ValueError(f"invalid episode id {episode_id!r}: one directory name, no path separators")
    return episode_id


class TraceStore:
    """Writer and reader for one trace store directory (see ``docs/reference/store.md``). Token rows are buffered
    per episode and flushed to ``tokens-NNNNN.parquet`` parts; the other files are appended per turn."""

    def __init__(self, root: str | Path, flush_every: int = 512):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._parts: dict[str, int] = {}

    # -- per-token scalars ---------------------------------------------------

    def _episode_dir(self, episode_id: str, create: bool = True) -> Path:
        d = self.root / check_episode_id(episode_id)
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def episode_dir(self, episode_id: str) -> Path:
        """The directory of an episode (checked, not created)."""
        return self._episode_dir(episode_id, create=False)

    def append(self, tm: TokenMetrics) -> None:
        buf = self._buffers.setdefault(tm.episode_id, [])
        row = tm.to_row()
        row["probes"] = list((row["probes"] or {}).items())  # pyarrow map_ wants pairs
        buf.append(row)
        if len(buf) >= self.flush_every:
            self.flush(tm.episode_id)

    def flush(self, episode_id: str) -> None:
        buf = self._buffers.get(episode_id)
        if not buf:
            return
        table = pa.Table.from_pylist(buf, schema=TOKEN_SCHEMA)
        part = self._parts.get(episode_id)
        if part is None:            # an episode written before this store object was opened: continue after it
            have = [int(f.name[7:12]) for f in self._episode_dir(episode_id).glob("tokens-*.parquet") if f.name[7:12].isdigit()]
            part = max(have) + 1 if have else 0
        path = self._episode_dir(episode_id) / f"tokens-{part:05d}.parquet"
        # written aside and renamed, so a reader (the dashboard) never sees a partial part
        pq.write_table(table, path.with_name(path.name + ".tmp"))
        os.replace(path.with_name(path.name + ".tmp"), path)
        self._parts[episode_id] = part + 1
        self._buffers[episode_id] = []

    def read_tokens(self, episode_id: str) -> pa.Table:
        self.flush(episode_id)
        parts = sorted(self._episode_dir(episode_id, create=False).glob("tokens-*.parquet"))
        if not parts:
            return TOKEN_SCHEMA.empty_table()
        tables = [pq.read_table(p) for p in parts]
        for t in tables:
            meta = (t.schema.metadata or {}).get(b"schema_version", b"").decode()
            if meta != SCHEMA_VERSION:
                raise ValueError(
                    f"schema_version mismatch in {episode_id}: stored={meta!r} current={SCHEMA_VERSION!r}; "
                    "run a migration instead of mixing schemas"
                )
        return pa.concat_tables(tables)

    # -- activations -----------------------------------------------------------

    def save_activations(self, episode_id: str, layer: int, turn: int, arr: np.ndarray) -> None:
        """Append activation rows [n_tokens, d] for (episode, layer, turn) to zarr."""
        import zarr

        grp = zarr.open_group(str(self._episode_dir(episode_id) / "activations.zarr"), mode="a")
        key = f"layer{layer}/turn{turn}"
        arr = np.asarray(arr, dtype=np.float16)
        if key in grp:
            grp[key].append(arr)
        else:
            grp.create_array(key, data=arr, chunks=(min(1024, max(1, arr.shape[0])), arr.shape[1]))

    # -- raw messages (opt-in; what replay renders the prompt from) ------------

    def write_messages(self, episode_id: str, turn: int, messages: list[dict],
                       tools: list[dict] | None = None, chat_kwargs: dict | None = None) -> None:
        """Persist the full request message history at a given turn. Opt-in
        (raw content is large and privacy-sensitive) — gates forced-prefix
        replay and observation-content audits. `tools` (OpenAI function
        schemas) travel with the record so replay renders the same prompt the
        provider saw. `chat_kwargs` are chat
        template variables the agent's request set (e.g. enable_thinking=False);
        replay renders with them."""
        path = self._episode_dir(episode_id) / "messages.jsonl"
        rec = {"turn": turn, "messages": messages}
        if tools:
            rec["tools"] = tools
        if chat_kwargs:
            rec["chat_kwargs"] = chat_kwargs
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def read_messages(self, episode_id: str) -> list[dict]:
        path = self._episode_dir(episode_id, create=False) / "messages.jsonl"
        if not path.exists():
            return []
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    # -- harness events ---------------------------------------------------------

    def append_events(self, episode_id: str, events: list) -> None:
        """Append boundary/harness events (jsonl) — the raw data for observation-
        boundary analyses. Accepts HarnessEvent models."""
        if not events:
            return
        path = self._episode_dir(episode_id) / "events.jsonl"
        with path.open("a") as f:
            for e in events:
                f.write(e.model_dump_json() + "\n")

    def read_events(self, episode_id: str) -> list:
        from vitals.core.schema import HarnessEvent

        path = self._episode_dir(episode_id, create=False) / "events.jsonl"
        if not path.exists():
            return []
        return [HarnessEvent.model_validate_json(ln) for ln in path.read_text().splitlines() if ln.strip()]

    # -- regions and received attention -----------------------------------------

    def append_attn_received(self, episode_id: str, turn: int, vec) -> None:
        """Per-turn received-attention vector: mean attention mass each PROMPT
        position drew from the turn's generated tokens (heads/layers averaged,
        rows normalized). One small vector per turn -- the 'which words did the
        model actually read' channel for prompt/observation highlighting."""
        path = self._episode_dir(episode_id) / "attn_received.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"turn": turn,
                                "vec": [round(float(x), 6) for x in np.asarray(vec)]}) + "\n")

    def read_attn_received(self, episode_id: str) -> dict[int, list[float]]:
        return read_received_attention(self._episode_dir(episode_id, create=False))

    def write_regions(self, episode_id: str, turn: int, regions: list) -> None:
        """Persist the region list (with token spans) as seen at a given turn —
        the interpretation key for attn_share vectors (region order + trailing
        generated bucket)."""
        path = self._episode_dir(episode_id) / "regions.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"turn": turn, "regions": [r.model_dump(mode="json") for r in regions]}) + "\n")

    def append_attn_tokens(self, episode_id: str, turn: int, token_ids, token_strs,
                           regions: list, recv) -> None:
        """Per-prompt-token attention table for a turn (see write_attn_tokens)."""
        write_attn_tokens(self._episode_dir(episode_id), turn, token_ids, token_strs, regions, recv)

    def read_regions(self, episode_id: str) -> list[dict]:
        """The raw regions.jsonl records of an episode, one per recorded turn (``read_regions(ep)`` gives them
        keyed by turn)."""
        path = self._episode_dir(episode_id, create=False) / "regions.jsonl"
        if not path.exists():
            return []
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    # -- manifests -------------------------------------------------------------

    def write_manifest(self, manifest: EpisodeManifest) -> Path:
        path = self._episode_dir(manifest.episode_id) / "manifest.json"
        atomic_write_text(path, json.dumps(manifest.model_dump(mode="json"), indent=2))
        return path

    def read_manifest(self, episode_id: str) -> EpisodeManifest:
        path = self._episode_dir(episode_id, create=False) / "manifest.json"
        return EpisodeManifest.model_validate_json(path.read_text())


# -- episode readers (one episode directory) ---------------------------------


def is_episode_dir(path: str | Path) -> bool:
    """Whether a directory holds one stored episode."""
    path = Path(path)
    return path.is_dir() and ((path / "manifest.json").exists() or (path / "regions.jsonl").exists())


def episode_dirs(store_dir: str | Path) -> list[Path]:
    """The episode directories of a store, in name order."""
    store_dir = Path(store_dir)
    return sorted(p for p in store_dir.iterdir() if is_episode_dir(p)) if store_dir.is_dir() else []


def read_regions(ep: str | Path) -> dict[int, list[dict]]:
    """Regions of every turn of an episode, from regions.jsonl, keyed by turn."""
    out: dict[int, list[dict]] = {}
    p = Path(ep) / "regions.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
                out[int(rec.get("turn", 0))] = rec.get("regions", [])
            except (json.JSONDecodeError, ValueError):
                continue
    return out


def read_token_rows(ep: str | Path):
    """All token rows of an episode as one pandas frame, or None without parquet parts."""
    import pandas as pd

    parts = sorted(Path(ep).glob("tokens-*.parquet"))
    if not parts:
        return None
    return pd.concat([pd.read_parquet(p) for p in parts])


def read_received_attention(ep: str | Path) -> dict[int, list[float]]:
    """Mean attention received by each prompt position, per turn, from attn_received.jsonl."""
    out: dict[int, list[float]] = {}
    p = Path(ep) / "attn_received.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
                out[int(rec["turn"])] = rec["vec"]
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def read_messages(ep: str | Path) -> dict[int, dict]:
    """The messages.jsonl records of an episode keyed by turn (``messages``, and ``tools`` / ``chat_kwargs`` when
    recorded); malformed lines are skipped."""
    from vitals.core.io import read_jsonl

    out = {}
    for rec in read_jsonl(Path(ep) / "messages.jsonl"):
        try:
            out[int(rec.get("turn", 0))] = rec
        except (TypeError, ValueError):
            continue
    return out


def turn_prompt_text(ep: str | Path, turn: int, messages: dict[int, dict] | None = None) -> str | None:
    """All message text the model saw at ``turn`` (content strings and text parts, one message per line), for
    finding the prompt line a token belongs to. ``messages``: ``read_messages(ep)`` when already loaded."""
    from vitals.core.messages import content_text

    rec = (messages if messages is not None else read_messages(ep)).get(int(turn))
    if rec is None:
        return None
    return "\n".join(content_text(m) for m in rec.get("messages", []))


def turn_inputs(ep_dir: str | Path, turn: int, backend=None) -> dict:
    """One stored turn as token lists: ``prompt`` (from attn_tokens.parquet), ``output`` (the generated tokens)
    with their stored ``top_ids``, and ``labels``, a section name per prompt and output position
    (``vitals.core.sections.section_labels``). ``backend`` is accepted for compatibility and unused."""
    import pandas as pd

    from vitals.core.sections import section_labels

    ep = Path(ep_dir)
    at = pd.read_parquet(ep / "attn_tokens.parquet", columns=["turn", "idx", "token_id"], filters=[("turn", "==", turn)])
    prompt = at.sort_values("idx").token_id.astype(int).tolist()
    tk = pd.concat([pd.read_parquet(p, columns=["turn", "step", "region_id", "token_id", "top_ids"],
                                    filters=[("turn", "==", turn)]) for p in sorted(ep.glob("tokens-*.parquet"))])
    g = tk[tk.region_id == "gen"].sort_values("step")
    out = g.token_id.astype(int).tolist()
    return {"prompt": prompt, "output": out, "top_ids": [list(map(int, x)) for x in g.top_ids],
            "labels": section_labels(read_regions(ep).get(turn, []), len(prompt), len(out))}


def span_index(regs: list) -> dict[str, tuple[int, int]]:
    """section -> (region index, token span size) for the named spans of one turn."""
    out = {}
    for i, r in enumerate(regs):
        rid = str(r.get("id", ""))
        s, e = r.get("token_start"), r.get("token_end")
        if rid.startswith("span:") and s is not None and e is not None and e > s:
            out[rid.removeprefix("span:")] = (i, int(e) - int(s))
    return out
