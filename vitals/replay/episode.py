"""Teacher-forced replay of logged episodes: attention and logit channels.

Episodes may be generated anywhere (the local server, or a provider running
the same model). Replay renders each turn's stored request, prefills the
prompt, then decodes the stored generated tokens teacher-forced, recovering:

  attention channel (needs the fa-off diagnostic backend):
    - attn_received.jsonl rows (mean mass each prompt position drew from the
      turn's forced output), attn_tokens.parquet (the same per prompt token,
      with strings and region ids), attn_sparse.parquet (top-K positions per
      forced token) and the parquet attn_share per generated token, in the
      live-capture convention (token k's shares = the query row that sampled it)
  logits channel (works fa-on at full prefill speed):
    - parquet ent / m2 / surprisal / top_ids / top_lp per generated token,
      same prev-row attribution (token k's stats = the distribution it was
      sampled from)

Multi-turn episodes: templates strip prior <think> content, so the state
after a turn's forced output cannot be continued into the next turn -- but the
state after its PROMPT can. With kv_reuse (backend n_seq_max >= 2) seq 0 keeps
each turn's prompt state, the forced output runs on a scratch copy (seq_cp),
and the next turn decodes only its new prompt tokens when they strictly extend
the previous prompt (append-only agent histories); otherwise a full reset.

By default only MISSING channels are written; ``sidecar=True`` writes
replay_* files instead (replay-vs-live validation).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from vitals.analysis.concentration import (
    CONCENTRATION_VERSION,
    SectionConcentration,
    read_concentration,
    write_concentration_turns,
)
from vitals.analysis.linear import linear_turns_done, write_linear_turns
from vitals.core.schema import AttentionMode, AttentionSpec, CaptureSpec, Region
from vitals.core.store import episode_dirs, read_regions, write_attn_tokens_turns
from vitals.replay.pooling import (
    chunk_logit_stats,
    layer_head_mean,
    logit_rows,
    logit_stats,
    pad_cols,
    pool_chunk,
    pool_row,
    sparse_row,
    stats_device,
    take_device_rows,
)
from vitals.replay.writers import jsonl_turns, write_attn_sparse_turns, write_back, write_reasoning_steps

try:
    import torch as _torch
except ImportError:  # the per-row numpy paths stay complete without torch
    _torch = None


def _attention_capture(backend):
    """Capture plan and session for the attention channel.

    PER_HEAD keeps full (heads, T_q, n_kv) matrices; each engine.decode() clears
    the capture accumulator on entry, so the replay's small chunks bound the
    transient per layer. Replay only uses the head/layer mean, so heads are
    pooled on the device when possible (self-checked; falls back to full
    copies). "layer_mean" also sums the pooled rows over the layers on the
    device and the replay reduces them there (region shares, received
    attention, sparse top-K), so no attention row crosses the bus.
    VITALS_REPLAY_HEAD_MEAN=0 forces the full per-head host copies;
    VITALS_REPLAY_DEVICE_POOL=0 keeps host copies of the pooled rows and pools
    each row on the host."""
    from vitals.engine.capture import CaptureSession, plan_capture

    spec = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.PER_HEAD))
    plan = plan_capture(spec, backend.node_map)
    if os.environ.get("VITALS_REPLAY_HEAD_MEAN", "1") != "0":
        mode = "layer_mean" if os.environ.get("VITALS_REPLAY_DEVICE_POOL", "1") != "0" else "head_mean"
        for name in list(plan.modes):
            if plan.modes[name] == "full":
                plan.modes[name] = mode
    return plan, CaptureSession(plan, backend.node_map)


def _linear_reading(backend, opts: dict):
    """The backend's LinearReading, reused across episodes while its spec and options are unchanged."""
    from vitals.engine.nodes import linear_spec
    from vitals.probes.linear_reading import LinearReading

    spec = linear_spec(backend.engine, backend.node_map)
    if spec is None:
        raise ValueError(f"linear reading: {backend.node_map.arch} has no linear-attention layers")
    prev = getattr(backend, "_replay_linear", None)
    if prev is not None and prev.spec == spec and prev.opts == opts:
        return prev
    return LinearReading(backend.engine, spec, **opts)


def _saved_sections(ep: Path) -> tuple[dict, dict]:
    """The store's section declarations per turn, and the stored prompt token ids per turn they were
    resolved against."""
    import pandas as pd

    regions = read_regions(ep)
    prompt_tokens = {}
    if (ep / "attn_tokens.parquet").exists():
        at = pd.read_parquet(ep / "attn_tokens.parquet", columns=["turn", "idx", "token_id"])
        prompt_tokens = {int(u): a.sort_values("idx").token_id.tolist() for u, a in at.groupby("turn")}
    return regions, prompt_tokens


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k


def replay_episode(
    backend,
    store_dir: str | Path,
    episode_id: str,
    *,
    spans: list[dict] | None = None,
    auto_sections: bool = True,
    sidecar: bool = False,
    force: bool = False,
    chunk: int | None = None,
    channels: tuple[str, ...] = ("attention", "logits"),
    top_k: int = 32,
    token_attn: bool = False,
    sparse_attn_k: int = 64,
    kv_reuse: bool = False,
    turns: set[int] | None = None,
    next_prompt: list[int] | None = None,
    section_diagnostics: bool = False,
    linear: bool = False,
    linear_opts: dict | None = None,
) -> dict:
    """Replay one episode's turns teacher-forced. For the attention channel
    ``backend`` must be constructed with diagnostic_attention=True (fa off);
    the logits channel works under either mode. token_attn=True also emits a
    self-describing attn_tokens.parquet (per-prompt-token string + region +
    received attention).

    section_diagnostics=True accumulates exact pooled/per-step section peaks
    and attention-focus JS before sparse truncation, writes
    section_diagnostics.parquet and enables token_attn. Older captures missing
    the current diagnostics version are replayed to fill it.

    sparse_attn_k > 0 (default 64) also stores attn_sparse.parquet: per
    FORCED token, the top-K prompt positions of its attention row, the
    prompt/generated mass totals, and the top-K_gen (256) generated positions it
    read (gpos/gw, as generated-step indices: the reasoning-to-answer joint).
    Sparse rows support approximate analysis after changing section bounds.
    A section's share can be undercounted by up to the omitted prompt tail;
    exact per-section peaks are not generally recoverable from this table.
    The optional section diagnostics accumulate peaks before this truncation.
    Set 0 to disable.

    kv_reuse=True (backend built with n_seq_max >= 2): a turn whose prompt
    tokens strictly extend the previous turn's prompt decodes only the new
    tokens on seq 0; the forced output is decoded on a scratch seq (seq_cp of
    the prefix state, dropped after the turn) so seq 0 keeps the prefix. Any
    divergence (edited history, retokenization) falls back to a full reset.
    Prefill capture is armed only for the final prompt chunk (the query row
    of forced token 0 is the only prompt row used). ``turns`` restricts replay
    to those turn indices (others are reported as skipped 'not sampled').
    The result carries ``timing`` (per turn: prompt / decoded / reused tokens,
    forced tokens, seconds).

    Reuse also spans EPISODES: the resident prefix lives on the backend
    (``backend._replay_kept``), so the next episode of a store extends it when
    its first prompt shares that prefix (episodes of one harness share its
    scaffold head). ``next_prompt`` is the token list of the prompt that will
    follow this episode's last turn (see ``plan_episode_order``); it decides
    how much of the last turn's prompt stays resident.

    linear=True (hybrid models: Qwen3.5/3.6, GLM-5.3-Flash) also measures the
    linear-attention layers per section -- retained, memory share, read and
    retrieval -- and writes linear_reading.parquet / linear_steps.parquet
    (vitals.probes.linear_reading; ``linear_opts`` go to LinearReading)."""
    import pandas as pd

    want_attn = "attention" in channels
    want_logits = "logits" in channels
    if section_diagnostics and not want_attn:
        raise ValueError("section diagnostics require the attention replay channel")
    token_attn = token_attn or section_diagnostics
    if chunk is None:
        # forced tokens per engine call (see replay_store)
        chunk = int(os.environ.get("VITALS_REPLAY_CHUNK", "64"))
    if chunk < 1:
        raise ValueError("chunk must be positive")
    ep = Path(store_dir) / episode_id
    msgs_by_turn: dict[int, list[dict]] = {}
    tools_by_turn: dict[int, list[dict] | None] = {}
    kwargs_by_turn: dict[int, dict | None] = {}
    for line in (ep / "messages.jsonl").read_text(errors="replace").splitlines():
        rec = json.loads(line)
        msgs_by_turn[int(rec["turn"])] = rec["messages"]
        tools_by_turn[int(rec["turn"])] = rec.get("tools")
        kwargs_by_turn[int(rec["turn"])] = rec.get("chat_kwargs")
    parts = sorted(ep.glob("tokens-*.parquet"))
    if not parts or not msgs_by_turn:
        return {"episode": episode_id, "turns": 0, "skipped": "no messages/tokens"}
    tokens = pd.concat([pd.read_parquet(p) for p in parts])
    gen = tokens[tokens.region_id == "gen"]
    write_reasoning_steps(backend, ep, gen)

    # Channels already stored are skipped per turn (unless forced), so an
    # extended episode resumes without re-decoding its completed turns.
    recv_path = ep / ("replay_attn_received.jsonl" if sidecar else "attn_received.jsonl")
    have_recv = jsonl_turns(recv_path) if want_attn and recv_path.exists() and not force else set()
    have_tok_attn: set[int] = set()     # turns whose per-prompt-token attention is stored
    if token_attn and want_attn and not force and not sidecar and (ep / "attn_tokens.parquet").exists():
        have_tok_attn = {int(t) for t in pd.read_parquet(ep / "attn_tokens.parquet", columns=["turn"]).turn.unique()}
    have_diagnostics: set[int] = set()
    if section_diagnostics and not force and not sidecar:

        captured = read_concentration(ep)
        have_diagnostics = {t for t, _ in captured if all(
            r["version"] == CONCENTRATION_VERSION for (u, _), r in captured.items() if u == t)}
    have_lg: set[int] = set()      # turns whose generated rows already carry top_ids
    if want_logits and not force and not sidecar and len(gen):
        have_lg = {int(t) for t, g in gen.groupby("turn") if len(g) and g.top_ids.notna().all()}

    plan = session = None
    if want_attn:
        plan, session = _attention_capture(backend)
    lin = None                    # linear-attention reading (hybrid models)
    have_lin: set[int] = set()
    if linear:

        lin = _linear_reading(backend, linear_opts or {})
        if not force:
            have_lin = linear_turns_done(ep)
    lin_names = lin.capture_names if lin is not None else set()

    shares_by_turn: dict[int, list[list[float]]] = {}
    lstats_by_turn: dict[int, list[dict]] = {}
    done, skipped = [], []
    eng = backend.engine
    reuse_ok = bool(kv_reuse) and int(getattr(eng, "_ctx_opts", {}).get("n_seq_max", 1)) >= 2
    # prompt tokens resident in seq 0 (the shared-prefix state), carried across
    # episodes on the backend when the capture plan is unchanged
    kept, kept_valid = [], False
    plan_sig = ((tuple(sorted(plan.names)), tuple(sorted(plan.modes.items()))) if plan is not None else None,
                tuple(sorted(lin_names)))
    if reuse_ok:
        prev = getattr(backend, "_replay_kept", None)
        if prev and prev[2] == plan_sig and eng.n_past == len(prev[0]):
            kept, kept_valid = list(prev[0]), bool(prev[1])
    order = [u for u in sorted(msgs_by_turn) if turns is None or u in turns]

    need_sections = section_diagnostics or linear
    saved_regions, saved_prompt_tokens = _saved_sections(ep) if need_sections and spans is None else ({}, {})
    stored_regions = (saved_regions or read_regions(ep)) if spans is None and auto_sections else {}
    rendered: dict[int, tuple] = {}
    auto_used: dict[int, bool] = {}

    def render(u: int):
        if u not in rendered:
            # automatic sections where neither a spans file nor the store declares any for this turn, and where
            # the stored sections are themselves automatic (they are a function of the request: same regions)
            stored_spans = [r for r in stored_regions.get(u, []) if str(r.get("id", "")).startswith("span:")]
            use_auto = auto_sections and spans is None and all((r.get("meta") or {}).get("auto") for r in stored_spans)
            rendered[u] = backend.render_turn(msgs_by_turn[u], tools=tools_by_turn.get(u),
                                              chat_kwargs=kwargs_by_turn.get(u), spans=spans, auto_sections=use_auto)
            auto_used[u] = use_auto and not stored_spans and any(
                str(r.id).startswith("span:") for r in rendered[u][0].regions)
            if need_sections:
                prepared, prompt_ids = rendered[u]
                if not any(str(r.id).startswith("span:") for r in prepared.regions):
                    # reuse the store's own declarations when its stored prompt matches
                    declared = saved_regions.get(u, [])
                    usable = (any(str(r.get("id", "")).startswith("span:") for r in declared)
                              and saved_prompt_tokens.get(u) == prompt_ids)
                    if section_diagnostics and not usable:
                        raise ValueError("section diagnostics need named spans whose stored prompt matches the "
                                         "rendered one; provide spans or --spans-file")
                    if usable:     # linear reading without declarations measures message regions only
                        prepared.regions = [Region.model_validate(r) for r in declared]
        return rendered[u]

    tm = {"kv_reuse": reuse_ok, "turns": [], "prompt_tokens": 0, "decoded_prompt_tokens": 0,
          "reused_tokens": 0, "gen_tokens": 0, "seconds": 0.0, "resets": 0, "extensions": 0}
    # Everything a turn writes is buffered and written once per episode (one
    # read-modify-write per parquet table instead of one per turn), in a finally
    # so an exception mid-episode still leaves the completed turns consistent.
    pending: dict = {"recv": [], "regions": [], "tokens": {}, "sparse": {}, "conc": {}, "linear": {}}
    tok_strs: dict[int, str] = {}      # token id -> piece, once per distinct id per episode

    def piece(tok: int) -> str:
        s = tok_strs.get(tok)
        if s is None:
            s = tok_strs[tok] = backend.detokenize([tok])
        return s

    def flush() -> int:
        if pending["recv"]:
            with recv_path.open("a") as f:
                f.write("\n".join(pending["recv"]) + "\n")
        if pending["regions"]:
            with (ep / ("replay_regions.jsonl" if sidecar else "regions.jsonl")).open("a") as f:
                f.write("\n".join(pending["regions"]) + "\n")
        if pending["tokens"]:
            write_attn_tokens_turns(ep, pending["tokens"])
        if pending["sparse"]:
            write_attn_sparse_turns(ep, pending["sparse"])
        if pending["conc"]:
            write_concentration_turns(ep, pending["conc"], sidecar=sidecar)
        if pending["linear"]:
            write_linear_turns(ep, pending["linear"])
        for k in pending:
            pending[k].clear()
        return write_back(ep, parts, gen, shares_by_turn, lstats_by_turn, sidecar=sidecar, force=force)

    lin_done: list[int] = []
    try:
        for t in sorted(msgs_by_turn):
            if turns is not None and t not in turns:
                skipped.append((t, "not sampled"))
                continue
            g = gen[gen.turn == t].sort_values("step")
            if len(g) == 0:
                skipped.append((t, "no generated tokens"))
                continue
            turn_attn = want_attn and (t not in have_recv or sidecar
                                       or (section_diagnostics and t not in have_diagnostics)
                                       or (token_attn and not sidecar and t not in have_tok_attn))
            turn_logits = want_logits and t not in have_lg
            turn_lin = lin is not None and t not in have_lin
            if not turn_attn and not turn_logits and not turn_lin:
                skipped.append((t, "channels already present"))
                continue
            out_toks = [int(x) for x in g.token_id]
            ctx, toks = render(t)
            # long transcripts can exceed the per-sequence context capacity
            # (llama_decode fails past it): skip the turn, not the episode
            cap = getattr(eng, "n_ctx_per_seq", 0)
            if cap and len(toks) + len(out_toks) + 8 > cap:
                skipped.append((t, f"prompt+gen {len(toks) + len(out_toks)} "
                                   f"exceeds per-seq capacity {cap}"))
                continue
            t_start = time.time()
            n_prompt = len(toks)
            concentration = None
            if turn_attn and section_diagnostics:

                concentration = SectionConcentration([piece(int(tok)) for tok in toks], ctx.regions)
            # Prefix reuse. The prompt of turn t ends with the generation suffix
            # (e.g. "<|im_start|>assistant\n<think>\n"); the next turn replaces
            # that suffix with the assistant's message, so the FULL prompt never
            # extends into the next one -- only its common prefix with the next
            # turn's prompt does. All turns are known up front, so seq 0 keeps
            # exactly that shared prefix; the turn-specific tail and the forced
            # output run on a scratch seq copied from it.
            split = n_prompt
            if reuse_ok:
                nxt = next((u for u in order if u > t), None)
                ntoks = render(nxt)[1] if nxt is not None else next_prompt
                if ntoks is not None:
                    split = _common_prefix_len(toks, ntoks)
            # the resident prefix cannot shrink (no rollback), so keep at least
            # it; a later prompt sharing less will reset then
            if reuse_ok and kept_valid and len(kept) > split:
                split = len(kept)
            # the tail must hold >= 1 token in both paths: its last decoded row
            # is the query of forced token 0, the only prompt row ever captured
            split = max(1, min(split, n_prompt - 1))
            shared = toks[:split]
            ext = (reuse_ok and kept_valid and len(toks) > len(kept)
                   and toks[:len(kept)] == kept)
            if not ext:
                if lin is None:
                    eng.reset_context(
                        capture_names=plan.names if turn_attn else None,
                        capture_modes=plan.modes if turn_attn else None)
                else:
                    eng.reset_context(
                        capture_names=(set(plan.names) if turn_attn else set()) | lin_names,
                        capture_modes={**(plan.modes if turn_attn else {}), **lin.capture_modes})
                    lin.reset()
                kept = []
                kept_valid = False
                tm["resets"] += 1
            else:
                tm["extensions"] += 1
            new = shared[len(kept):]        # shared prefix not yet resident (seq 0)
            tail = toks[split:]             # this turn's own suffix (scratch seq)
            region_spans = [(r.token_start, r.token_end) for r in ctx.regions]
            cap_names = plan.names if turn_attn else None
            if lin is not None:
                # every turn keeps the linear memories in step with seq 0, written or not
                lin.begin_turn(ctx.regions, n_prompt, len(out_toks))
            # 1) extend the resident prefix on seq 0, attention capture off (no row
            #    of it is used) -- so full batches, not the small capture-bounded chunks
            big = max(chunk, int(getattr(eng, "_ctx_opts", {}).get("n_batch", chunk)))
            if cap_names is not None or lin is not None:
                eng.set_capture(lin_names)
            for off in range(0, len(new), big):
                eng.decode(new[off:off + big])
                if lin is not None:
                    lin.feed("resident", len(kept) + off, len(new[off:off + big]))
            kept = list(shared)
            kept_valid = reuse_ok
            # 2) the tail on the scratch seq (or on seq 0 without reuse): full
            #    batches with capture off up to the last <= chunk tokens, whose
            #    final row -- the query of forced token 0 -- is the one captured
            if reuse_ok:
                eng.branch_copy(1, 0)
            if lin is not None:
                lin.fork()
            prev_att = prev_lg = None          # host carries: the row / logits that sample the next token
            prev_dev = prev_lg_dev = None      # the same on the device, when the engine pools there
            try:
                pos = len(shared)
                cut = max(0, len(tail) - chunk)
                pieces = [tail[o:min(o + big, cut)] for o in range(0, cut, big)] + [tail[cut:]]
                off = 0
                for part in pieces:
                    last = off + len(part) >= len(tail)
                    if cap_names is not None or lin is not None:
                        eng.set_capture(((cap_names or set()) if last else set()) | lin_names)
                    if reuse_ok:
                        eng.decode(part, pos0=pos + off, seq_id=1)
                        eng.n_past = pos + off + len(part)
                    else:
                        eng.decode(part)
                    if lin is not None:
                        lin.feed("scratch", pos + off, len(part), want=(n_prompt - 1, n_prompt))
                    off += len(part)
                    if last:
                        if turn_attn:
                            dev = take_device_rows(eng)
                            if dev is not None:
                                prev_dev = dev[-1:].clone()
                            else:
                                _, attn = session.harvest(eng)
                                if attn:
                                    prev_att = layer_head_mean(attn)[-1]
                        if turn_logits:
                            row = np.array(eng.logits(-1), copy=True)
                            if _torch is not None:
                                prev_lg_dev = _torch.from_numpy(row).to(stats_device())[None]
                            else:
                                prev_lg = row
                if cap_names is not None or lin is not None:
                    eng.set_capture((cap_names or set()) | lin_names)
                if lin is not None:
                    lin.gen_start()
                tm["prompt_tokens"] += n_prompt
                tm["decoded_prompt_tokens"] += len(new) + len(tail)
                tm["reused_tokens"] += len(shared) - len(new)
                if turn_attn and prev_att is None and prev_dev is None:
                    skipped.append((t, "no attention captured -- backend must be "
                                       "diagnostic (fa off)"))
                    if lin is not None:
                        lin.abort_turn()
                    continue
                # prev-row attribution matching live capture: token k's channels come
                # from the row whose logits sampled it; the final output token's own
                # row sampled nothing recorded and is dropped
                recv_sum = np.zeros(n_prompt, dtype=np.float64)
                recv_dev = None
                shares: list[list[float]] = []
                lstats: list[dict] = []
                sparse_rows: list[tuple] = []
                n_done = 0
                for off in range(0, len(out_toks), chunk):
                    part = out_toks[off:off + chunk]
                    if reuse_ok:
                        eng.decode(part, pos0=n_prompt + off, seq_id=1, logits_all=turn_logits)
                        eng.n_past = n_prompt + off + len(part)   # harvest slices attention to n_past
                    else:
                        eng.decode(part, logits_all=turn_logits)
                    if lin is not None:
                        lin.feed("scratch", n_prompt + off, len(part), want=(n_prompt, n_prompt + len(out_toks) - 1))
                    if turn_attn:
                        att_rows = None
                        dev_rows = take_device_rows(eng)
                        if dev_rows is None:
                            _, attn = session.harvest(eng)
                            if not attn:
                                break
                            att_rows = layer_head_mean(attn)
                        if dev_rows is not None:
                            # the rows that sampled part[j]: the carried row, then this
                            # chunk's rows but its last, which becomes the next carry;
                            # everything is reduced on the device, only shares and
                            # sparse top-K come back
                            if prev_dev is None:
                                prev_dev = _torch.as_tensor(np.asarray(prev_att, dtype=np.float32),
                                                            device=dev_rows.device)[None]
                            rows = _torch.cat([pad_cols(prev_dev, dev_rows.shape[1]), dev_rows[:-1]])
                            c_shares, c_recv, c_sparse = pool_chunk(rows, region_spans, n_prompt,
                                                                     sparse_attn_k, n_done)
                            shares.extend(c_shares)
                            recv_dev = c_recv if recv_dev is None else recv_dev + c_recv
                            if concentration is not None:
                                for row in rows[:, :n_prompt].cpu().numpy():
                                    concentration.update(row)
                            sparse_rows.extend(c_sparse)
                            prev_dev = dev_rows[-1:].clone()
                        else:
                            if prev_att is None:
                                prev_att, prev_dev = prev_dev[0].cpu().numpy(), None
                            for j in range(len(part)):
                                shares.append(pool_row(prev_att, region_spans, gen_start=n_prompt))
                                recv_sum += prev_att[:n_prompt]
                                if concentration is not None:
                                    concentration.update(prev_att[:n_prompt])
                                if sparse_attn_k:
                                    sparse_rows.append(sparse_row(prev_att, n_prompt, sparse_attn_k, n_done + j))
                                prev_att = att_rows[j]
                    if turn_logits:
                        # token j's stats come from the row that sampled it (the carried
                        # previous row), then row j becomes the carry; rows must be
                        # copied out before the next decode call
                        if _torch is not None:
                            lrows = _torch.from_numpy(logit_rows(eng, len(part))).to(prev_lg_dev.device)
                            lstats.extend(chunk_logit_stats(_torch.cat([prev_lg_dev, lrows[:-1]]), part, top_k))
                            prev_lg_dev = lrows[-1:].clone()
                        else:
                            for j, tok in enumerate(part):
                                lstats.append(logit_stats(prev_lg, tok, top_k))
                                prev_lg = np.array(eng.logits(j), copy=True)
                    n_done += len(part)
            finally:
                if reuse_ok:
                    eng.branch_drop(1)
                    eng.n_past = len(shared)
            secs = time.time() - t_start
            tm["gen_tokens"] += n_done
            tm["seconds"] += secs
            tm["turns"].append({"turn": t, "prompt_tokens": n_prompt, "decoded_prompt_tokens": len(new) + len(tail),
                                "gen_tokens": n_done, "seconds": round(secs, 3), "extended": bool(ext)})
            if n_done != len(out_toks):
                skipped.append((t, f"replayed rows {n_done} != output tokens {len(out_toks)}"))
                if lin is not None:
                    lin.abort_turn()
                continue
            if lin is not None:
                if turn_lin:
                    pending["linear"][t] = lin.end_turn(t)
                    lin_done.append(t)
                else:
                    lin.abort_turn()
            if turn_attn:
                if recv_dev is not None:
                    recv_sum = recv_sum + recv_dev.cpu().numpy()
                recv_vec = recv_sum / n_done
                pending["recv"].append(json.dumps({"turn": t, "vec": [round(float(x), 6) for x in recv_vec],
                                                   "replayed": True}))
                if token_attn and not sidecar:
                    pending["tokens"][t] = (toks, [piece(int(x)) for x in toks], ctx.regions, recv_vec)
                if sparse_attn_k and sparse_rows and not sidecar:
                    pending["sparse"][t] = sparse_rows
                # Replacing attention also invalidates any older concentration
                # summary when this replay did not request the optional capture.
                pending["conc"][t] = concentration.finish(t) if concentration is not None else []
                shares_by_turn[t] = shares
                if (spans or auto_used.get(t)) and any(str(r.id).startswith("span:") for r in ctx.regions):
                    pending["regions"].append(json.dumps({"turn": t, "regions": [
                        r.model_dump(mode="json") for r in ctx.regions]}))
            if turn_logits:
                lstats_by_turn[t] = lstats
            done.append(t)
    finally:
        n_upd = flush()

    if reuse_ok:
        backend._replay_kept = (kept, kept_valid, plan_sig)
        backend._replay_linear = lin
    return {"episode": episode_id, "turns": len(done), "turns_replayed": done,
            "skipped": skipped, "rows_updated": n_upd, "sidecar": sidecar, "timing": tm,
            "channels": [c for c, present in (("attention", shares_by_turn), ("logits", lstats_by_turn),
                                              ("linear", lin_done)) if present]}


def first_prompt_tokens(backend, store_dir: str | Path, episode_id: str) -> list[int]:
    """Token ids of an episode's first turn prompt (no spans needed: tokens do
    not depend on section declarations)."""
    ep = Path(store_dir) / episode_id
    for line in (ep / "messages.jsonl").read_text(errors="replace").splitlines():
        rec = json.loads(line)
        _, toks = backend.render_turn(rec["messages"], tools=rec.get("tools"), chat_kwargs=rec.get("chat_kwargs"))
        return list(toks)
    return []


def plan_episode_order(backend, store_dir: str | Path, episode_ids: list[str]) -> list[tuple[str, list[int] | None]]:
    """Order episodes so consecutive ones share the longest prompt prefixes
    (lexicographic order of first-turn token ids groups a design's shared
    scaffold head), and pair each with the NEXT episode's first prompt so the
    replay keeps exactly the shared prefix resident. Returns
    [(episode_id, next_prompt_tokens_or_None), ...]."""
    firsts = {eid: first_prompt_tokens(backend, store_dir, eid) for eid in episode_ids}
    ordered = sorted(episode_ids, key=lambda e: firsts[e])
    return [(e, firsts[ordered[i + 1]] if i + 1 < len(ordered) else None)
            for i, e in enumerate(ordered)]


def replay_store(
    gguf: str,
    store_dir: str | Path,
    *,
    episodes: list[str] | None = None,
    spans_file: str | Path | None = None,
    sidecar: bool = False,
    force: bool = False,
    n_ctx: int = 32768,
    lib_path: str | None = None,
    channels: tuple[str, ...] = ("attention", "logits"),
    top_k: int = 32,
    token_attn: bool = True,
    section_diagnostics: bool = False,
    chunk: int | None = None,
    linear: bool = False,
    linear_opts: dict | None = None,
    auto_sections: bool = True,
    n_gpu_layers: int = 999,
    log=print,
) -> list[dict]:
    """Replay every (or the named) episode(s) of a store. The backend loads
    once; fa-off diagnostic mode only when the attention channel is wanted.
    ``n_ctx`` is the context per sequence (the backend holds two).
    ``token_attn`` also writes attn_tokens.parquet (the attention each prompt
    token receives: peaks, hottest lines, the ledger and explanations need it).
    ``chunk`` forced tokens are decoded per engine call (default
    ``$VITALS_REPLAY_CHUNK``, else 64); larger chunks amortize
    the capture callback's per-call cost (256 measured 10-20% faster than 64 on
    an H200) at the price of a larger attention transient on the device."""
    from vitals.backends.llamacpp import LlamaCppBackend

    if section_diagnostics and "attention" not in channels:
        raise ValueError("section diagnostics require the attention replay channel")
    spans = None
    if spans_file is not None:
        spans = json.loads(Path(spans_file).read_text())
    # two sequences: seq 0 keeps the prefix shared with the next turn/episode,
    # the forced output runs on a scratch seq (per-seq capacity = n_ctx / 2)
    names = episodes or [p.name for p in episode_dirs(store_dir) if (p / "messages.jsonl").exists()]
    if not names:
        log(f"[replay] no episode with stored messages in {store_dir}")
        return []
    be = LlamaCppBackend(gguf, n_ctx=2 * n_ctx, n_seq_max=2,
                         diagnostic_attention="attention" in channels,
                         lib_path=lib_path, n_gpu_layers=n_gpu_layers)
    try:
        out = []
        for name, nxt in plan_episode_order(be, store_dir, names):
            r = replay_episode(be, store_dir, name, spans=spans, auto_sections=auto_sections,
                               sidecar=sidecar, force=force,
                               channels=channels, top_k=top_k, token_attn=token_attn,
                               kv_reuse=True, next_prompt=nxt, section_diagnostics=section_diagnostics,
                               chunk=chunk, linear=linear, linear_opts=linear_opts)
            log(f"[replay] {name}: {r['turns']} turns, {r.get('rows_updated', 0)} "
                  f"rows updated ({'+'.join(r.get('channels', [])) or 'none'})"
                  + (f", skipped {len(r['skipped'])}" if r["skipped"] else ""))
            out.append(r)
        return out
    finally:
        be.close()
