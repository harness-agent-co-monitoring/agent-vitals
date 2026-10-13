"""The read map of a stored episode.

One row per (turn, section) with four measures, computed from the store alone:

  lift              read lift, normalized after averaging (``reading.episode_turns``)
  peak / hot_line   peak lift over the section's most-read content tokens and
                    the line that holds them (needs attn_tokens.parquet)
  consult_ratio     entropy-weighted mean share over plain mean share
  consult_contrast  share at the most uncertain steps over the most confident

The server report, ``vitals report``, the dashboard and the Meta-Harness digests all read through this module, so
the numbers agree wherever they are shown. ``cached_readmap`` keeps the rows in a cache file that is recomputed
when the episode's files change.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

from vitals.core.paths import cache_dir as _cache_dir
from vitals.core.store import turn_prompt_text

_turn_prompt_text = turn_prompt_text     # the name older code imports

PEAK_K = 8          # tokens averaged for the peak
PEAK_MIN_TOK = 32   # shorter sections report no peak: their lift already is one


def content_token(t: str) -> bool:
    """True for a token that carries content: it has a letter, digit or CJK
    character (or is a byte-fallback fragment of one) and is not a lone ASCII
    letter or digit. Punctuation, whitespace and brackets attract delimiter
    heads and would dominate a top-K over all tokens."""
    st = t.strip()
    if "�" in st:
        return True
    return any(ch.isalnum() for ch in st) and not re.fullmatch(r"[A-Za-z0-9]", st)


def prompt_line(prompt: str, joined: str) -> str | None:
    """The prompt line containing the token-joined text. Token strings split
    multibyte characters, so anchor on the longest clean run of the joined
    text and return the full line around it."""
    runs = sorted((m.group(0) for m in re.finditer(r"[^�]{6,}", joined)), key=len, reverse=True)
    for key in runs[:3]:
        i = prompt.find(key.strip())
        if i >= 0:
            a = prompt.rfind("\n", 0, i) + 1
            b = prompt.find("\n", i)
            return prompt[a:(b if b >= 0 else len(prompt))].strip()
    return None


def hottest_line(tokens: list[str], top: list[int], weights: list[float], prompt: str | None) -> str | None:
    """The line holding most of the ``weights`` of the ``top`` tokens (indices into ``tokens``, one section's token
    strings in order), as the full prompt line when it can be found in ``prompt``, else as the joined tokens."""
    line_of, cur = [], 0
    for t in tokens:
        line_of.append(cur)
        cur += t.count("\n")
    mass: dict[int, float] = {}
    for i, w in zip(top, weights, strict=True):
        mass[line_of[i]] = mass.get(line_of[i], 0.0) + float(w)
    best = max(mass, key=mass.get)
    joined = "".join(t for t, li in zip(tokens, line_of, strict=True) if li == best).strip().replace("\n", " ")
    text = prompt_line(prompt, joined) if prompt else None
    return (text or joined.replace("\ufffd", "")) or None


def section_peaks(ep: str | Path, regions: list, L: int, turn: int = 0, *, attn=None,
                  prompt: str | None = None) -> dict[str, tuple[float | None, str | None]]:
    """Per section at ``turn``: (peak lift, hottest line). Peak is the
    mean context-normalized lift of the section's PEAK_K most-read content
    tokens. Raw attention is averaged before dividing by context mass, so
    output steps receive context-attention mass weights. The line is the one
    holding most of that top-K mass. Sections under PEAK_MIN_TOK tokens or
    with fewer than PEAK_K content tokens return (None, None). ``attn``: the
    turn's attn_tokens rows and ``prompt`` its message text, when the caller
    already has them (``episode_readmap`` reads each file once per episode)."""
    import pandas as pd

    ep = Path(ep)
    if attn is None:
        p = ep / "attn_tokens.parquet"
        if not p.exists():
            return {}
        attn = pd.read_parquet(p, columns=["turn", "idx", "token_str", "attn"], filters=[("turn", "==", turn)])
        prompt = turn_prompt_text(ep, turn)
    a = attn.sort_values("idx")
    if not len(a):
        return {}
    vec, toks = a.attn.to_numpy(dtype=float), a.token_str.tolist()
    ctx = float(vec.sum()) or 1.0
    out: dict[str, tuple[float | None, str | None]] = {}
    for r in regions:
        rid = str(r.get("id", ""))
        st, en = r.get("token_start"), r.get("token_end")
        if not rid.startswith("span:") or st is None or en is None:
            continue
        st, en = int(st), min(int(en), len(vec))
        name = rid[5:]
        if en - st < PEAK_MIN_TOK:
            out[name] = (None, None)
            continue
        tl = vec[st:en] * L / ctx
        cand = np.flatnonzero([content_token(t) for t in toks[st:en]])
        if len(cand) < PEAK_K:
            out[name] = (None, None)
            continue
        top = cand[np.argpartition(tl[cand], -PEAK_K)[-PEAK_K:]]
        peak = float(tl[top].mean())
        text = hottest_line(toks[st:en], top.tolist(), tl[top].tolist(), prompt)
        out[name] = (peak, text[:60] if text else None)
    return out


def episode_readmap(ep_dir: str | Path, tokens=None, *, diagnostics: bool = False) -> list[dict]:
    """Rows (turn, section, tokens, lift, peak, hot_line, consult_ratio,
    consult_contrast, L, n_gen) for every declared section of every turn."""
    import pandas as pd

    from vitals.analysis.reading import episode_turns, token_uptake_turns
    from vitals.core.store import read_messages, read_regions, read_token_rows

    ep = Path(ep_dir)
    if diagnostics and tokens is None:
        tokens = read_token_rows(ep)
        if tokens is None:
            return []
    turns = episode_turns(ep, tokens=tokens)
    if not turns:
        return []
    regions = read_regions(ep)
    at_path = ep / "attn_tokens.parquet"
    attn = (dict(tuple(pd.read_parquet(at_path, columns=["turn", "idx", "token_str", "attn"]).groupby("turn")))
            if at_path.exists() else {})
    messages = read_messages(ep)
    consult = {(r["turn"], r["section"]): r for r in token_uptake_turns(ep, tokens=tokens)}
    extra, concentration = {}, {}
    if diagnostics:
        from vitals.analysis.concentration import CONCENTRATION_FIELDS, read_concentration
        from vitals.analysis.diagnostics import episode_attention_diagnostics

        extra = episode_attention_diagnostics(ep, tokens=tokens)
        concentration = read_concentration(ep)
    rows: list[dict] = []
    for tr in turns:
        t = tr["turn"]
        regs = regions.get(t, [])
        foot = {}
        for r in regs:
            rid = str(r.get("id", ""))
            s, e = r.get("token_start"), r.get("token_end")
            if rid.startswith("span:") and s is not None and e is not None:
                foot[rid[5:]] = int(e) - int(s)
        a = attn.get(t)
        peaks = section_peaks(ep, regs, tr["L"], turn=t, attn=a, prompt=turn_prompt_text(ep, t, messages)) \
            if a is not None else {}
        for name, d in tr["sections"].items():
            c = consult.get((t, name), {})
            pk, hot = peaks.get(name, (None, None))
            rows.append({
                "turn": t, "section": name, "tokens": foot.get(name, 0),
                "lift": round(float(d["lift"]), 4),
                "peak": None if pk is None else round(pk, 4), "hot_line": hot,
                "consult_ratio": c.get("consult_ratio"),
                "consult_contrast": c.get("consult_contrast"),
                "L": int(tr["L"]), "n_gen": int(tr["n_gen"]),
            })
            if diagnostics:
                d = {**extra.get((t, name), {"status": "unavailable"}),
                     **dict.fromkeys(CONCENTRATION_FIELDS), "peak_status": "not_captured",
                     "focus_status": "not_captured"}
                peak_row = concentration.get((t, name))
                region = next((r for r in regs if r.get("id") == f"span:{name}"), {})
                if peak_row is not None:
                    if (peak_row["L"] == tr["L"] and peak_row["n_gen"] == tr["n_gen"]
                            and peak_row["token_start"] == region.get("token_start")
                            and peak_row["token_end"] == region.get("token_end")):
                        d.update({k: peak_row.get(k) for k in (*CONCENTRATION_FIELDS, "peak_status", "peak_k", "focus_status")})
                        d["concentration_version"] = peak_row["version"]
                    else:
                        d["peak_status"] = "capture_mismatch"
                        d["focus_status"] = "capture_mismatch"
                rows[-1]["diagnostics"] = d
    _join_linear(ep, rows)
    return rows


def _join_linear(ep: Path, rows: list[dict]) -> None:
    """Add the linear-attention measures (``lin_retained``, ``lin_mem_share``, ``lin_read``,
    ``lin_retrieval``) to each row whose section has them (replays run with linear reading)."""
    from vitals.analysis.linear import read_linear

    lin = read_linear(ep)
    if lin is None:
        return
    by = {(int(r.turn), r.section): r for r in lin.itertuples(index=False)}
    for row in rows:
        r = by.get((row["turn"], row["section"]))
        if r is not None:
            row.update(lin_retained=r.retained, lin_mem_share=r.mem_share, lin_read=r.read,
                       lin_retrieval=r.retrieval)


READMAP_FILE = "readmap.parquet"
READMAP_VERSION = 1
_READMAP_INPUTS = ("attn_received.jsonl", "regions.jsonl", "attn_tokens.parquet", "linear_reading.parquet",
                   "messages.jsonl")


def _readmap_signature(ep: Path) -> str:
    parts = [f"v{READMAP_VERSION}"]
    for name in _READMAP_INPUTS:
        if (p := ep / name).exists():
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    if tokens := sorted(ep.glob("tokens-*.parquet")):
        st = tokens[-1].stat()
        parts.append(f"tokens:{len(tokens)}:{st.st_size}:{st.st_mtime_ns}")
        # replay rewrites earlier parts in place too (logits fill entropy, which consult uses)
        parts += [f"{t.name}:{t.stat().st_size}:{t.stat().st_mtime_ns}" for t in tokens[:-1]]
    return "|".join(parts)


def _readmap_cache_path(ep: Path, cache_dir: str | Path | None) -> Path:
    cache_dir = cache_dir or os.environ.get("VITALS_READMAP_CACHE")
    if not cache_dir and os.access(ep, os.W_OK):
        return ep / READMAP_FILE
    name = f"{hashlib.sha1(str(ep.resolve()).encode()).hexdigest()}.parquet"
    path = _cache_dir("readmaps", cache_dir) / name
    if not cache_dir and not path.exists():
        # read maps cached before the cache root moved under ~/.cache/vitals
        legacy = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "agent-vitals" / "readmaps" / name
        if legacy.exists():
            return legacy
    return path


def cached_readmap(ep_dir: str | Path, cache_dir: str | Path | None = None, *, compute: bool = True) -> list[dict] | None:
    """``episode_readmap`` with a cache file, recomputed whenever the episode's inputs change.

    The cache is ``readmap.parquet`` in the episode, or a file in ``cache_dir`` (or ``$VITALS_READMAP_CACHE``) when
    given, or in a user cache directory when the episode is not writable. With ``compute=False`` a missing or stale
    cache returns None.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    ep = Path(ep_dir)
    sig, path = _readmap_signature(ep), _readmap_cache_path(ep, cache_dir)
    if path.exists():
        try:
            table = pq.read_table(path)
            if (table.schema.metadata or {}).get(b"vitals_signature", b"").decode() == sig:
                return [{k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
                        for row in table.to_pylist()]
        except (OSError, ValueError, KeyError):   # unreadable cache file: recompute below
            pass
    if not compute:
        return None
    rows = episode_readmap(ep)
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"vitals_signature": sig.encode()})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)
    except OSError:
        pass
    return rows


def store_readmap(store_dir: str | Path, *, diagnostics: bool = False) -> list[dict]:
    """episode_readmap over every episode of a store, with an ``episode`` field."""
    from vitals.core.store import episode_dirs

    out: list[dict] = []
    for ep in episode_dirs(store_dir):
        for r in episode_readmap(ep, diagnostics=diagnostics):
            out.append({"episode": ep.name, **r})
    return out


def format_readmap(rows: list[dict]) -> str:
    """A compact text table of read-map rows."""
    if not rows:
        return "(no declared sections with attention)"
    def num(v):
        return "" if v is None else f"{v:.2f}"

    has_ep = "episode" in rows[0]
    head = (["episode"] if has_ep else []) + ["turn", "section", "tokens", "lift", "peak", "consult", "contrast", "hottest line"]
    body = [([r["episode"]] if has_ep else []) + [
        str(r["turn"]), r["section"][:28], str(r["tokens"]), num(r["lift"]), num(r["peak"]),
        num(r["consult_ratio"]), num(r["consult_contrast"]), (r["hot_line"] or "")[:48]] for r in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *body, strict=True)]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True))
                     for cells in [head, *body])
