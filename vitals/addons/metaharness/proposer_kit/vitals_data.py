"""Model-free loader for the raw replay artifacts of a Vitals store.

The proposer analyzes how the frozen worker read the scaffold its candidates
built. This module loads the raw per-token measurements as plain DataFrames --
no model, no GPU, pure file reads -- and provides NO derived metrics or
judgments: every analytical choice is left to the caller. See VITALS_SCHEMA.md
for the file/column contract.

Primary object (`load_attention`): one row per PROMPT token, with its decoded
string, the harness region that wrote it, and the attention it received from
the worker's forced output. This is the "attention token to harness" table.

  from vitals_data import load_attention, load_generation, list_episodes
  df = load_attention(store_dir, episode_id)   # idx, token_str, region_id, attn
  # ... write whatever analysis you want on top ...
"""

from __future__ import annotations

from pathlib import Path


def list_stores(replay_root: str | Path) -> list[str]:
    """`<candidate>--<dataset>` store dirs under a run's replay root."""
    root = Path(replay_root)
    return sorted(d.name for d in root.glob("*--*") if (d / "store").is_dir())


def list_episodes(store_dir: str | Path) -> list[str]:
    """Episode ids in a store that have a replayed attention table."""
    store = Path(store_dir)
    return sorted(p.parent.name for p in store.glob("ep-*/attn_tokens.parquet"))


def load_attention(store_dir: str | Path, episode_id: str):
    """Per-PROMPT-token attention table for one episode. Columns:
      turn, idx, token_id, token_str, region_id, attn
    `attn` is the mean mass this prompt position received from the forced
    output. Across the prompt, it sums to the prompt-side share of attention
    for that turn, which can be less than 1. `region_id` is the finest harness
    region covering the token (span:<name>), or a message region.
    Returns an empty frame if the episode was not replayed."""
    import pandas as pd

    p = Path(store_dir) / episode_id / "attn_tokens.parquet"
    if not p.exists():
        return pd.DataFrame(
            columns=["turn", "idx", "token_id", "token_str", "region_id", "attn"])
    return pd.read_parquet(p)


def load_generation(store_dir: str | Path, episode_id: str):
    """Per-GENERATED-token table (the worker's own answer tokens). Columns
    include: step, token_id, ent (entropy), m2, js, kl_ref (divergence from the
    reference model), surprisal, top_ids, top_lp, region_id, attn_share."""
    import pandas as pd

    parts = sorted((Path(store_dir) / episode_id).glob("tokens-*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def load_manifest(store_dir: str | Path, episode_id: str) -> dict:
    """Episode manifest meta: task, memory (candidate), reward (1.0 correct /
    0.0 wrong), prediction, target, and prompt_sections if the harness declared
    them."""
    import json

    p = Path(store_dir) / episode_id / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("meta", {})


def iter_attention(store_dir: str | Path):
    """Yield (episode_id, attention_df) for every replayed episode in a store.
    Convenience for pooling across a candidate's episodes."""
    store = Path(store_dir)
    for eid in list_episodes(store):
        yield eid, load_attention(store, eid)
