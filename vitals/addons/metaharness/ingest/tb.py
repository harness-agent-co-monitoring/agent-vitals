"""Ingest terminal-bench (harbor) trajectories into vitals stores (multi-turn).

Source: harbor trajectory.json per trial -- step 1 is the initial user prompt
(the harness-authored scaffold + task), then one step per agent turn carrying
message (visible analysis/plan text), reasoning_content (the deliberation --
stored by harbor, NOT resent to the provider on later turns), tool_calls
(structured commands), and observation (the environment's reply).

Reconstruction per turn t:
  messages prefix = [user initial] + for each prior turn k<t:
      assistant: message_k + rendered tool-call JSON   (no reasoning -- matches
                 what the provider actually conditioned on)
      user:      observation_k content
  forced generation = reasoning_t + message_t + tool-call JSON_t
      (the worker's actual generation order under the reasoning API)

FIDELITY: the chat template is the local serving template over plain
user/assistant text, not the provider's exact tool-calling render. Absolute
logit stats therefore carry template-glue noise; section-attention over the
scaffold and the within-trial dynamics are the measurement targets, and they
are internally consistent across trials/agents.

Sections: proposer-declared scaffold spans when resolvable (the agent file's
module-level prompt_sections, else baseline_kira's, whose scaffold the kira
lineage shares byte for byte -- each candidate declaration validated
verbatim against the actually-logged prompt before use), else structural
spans (contracts.tb); plus one span per observation (obs:<k>) for
recency/grounding analyses; plus fb:<k>:<name> spans for HARNESS-AUTHORED
text inside observations (completion-confirmation checklists, truncation
warnings, tool-result headers), resolved from the agent file's optional
module-level feedback_sections (else baseline_kira's), validated verbatim
per observation and anchored "within" their observation so recurring
templates resolve to their own occurrence. Manifest meta: task="tb2",
memory=<agent job>, example_idx=<task name>, reward,
n_turns_total / n_turns_replayed (turn cap).
"""

from __future__ import annotations

import json
from pathlib import Path

from vitals.addons.metaharness.config import CFG
from vitals.addons.metaharness.contracts.tb import (
    load_declared_feedback_sections,
    load_declared_prompt_sections,
    sections_for,
    validate_feedback_sections,
    validate_prompt_sections,
)
from vitals.addons.metaharness.ingest import detok_engine
from vitals.core.schema import EpisodeManifest
from vitals.core.store import TraceStore
from vitals.replay import append_forced

# Candidates benchmarked outside the optimizer repo keep their agent files in
# their own tree: callers (replay-inloop-tb) add those dirs here, searched
# before the default; otherwise a candidate's feedback_sections would be
# silently replaced by baseline_kira's.
EXTRA_AGENT_DIRS: list[Path] = []


def _agent_file(name: str) -> Path | None:
    for d in [*EXTRA_AGENT_DIRS, CFG.tb_agents]:
        p = Path(d) / f"{name}.py"
        if p.exists():
            return p
    return None


def _declared_sections(agent: str | None, initial: str):
    """Proposer-declared fine-grained sections for a trial's initial prompt.

    Ladder: the trial agent's own file, else baseline_kira (the kira lineage
    shares its byte-identical scaffold, so the parent's declaration applies
    verbatim). Declarations are extracted by AST (agent files import harbor,
    which the replay environment need not have) and each candidate is
    validated against the actually-logged prompt before use, so the ladder can
    only refine labels, never mislabel: on any mismatch the caller's
    sections_for() falls back to structural spans."""
    for name in dict.fromkeys([agent, "baseline_kira"]):
        if not name:
            continue
        p = _agent_file(name)
        fn = load_declared_prompt_sections(p) if p else None
        if fn is None:
            continue
        try:
            decl = fn(initial)
        except Exception:
            continue
        if decl and validate_prompt_sections(decl, initial)[0]:
            return [(str(n), str(t)) for n, t in decl]
    return None


def _declared_feedback(agent: str | None):
    """Runtime-feedback declaration for a trial's agent (or None): the agent
    file's module-level feedback_sections, else baseline_kira's (the kira
    lineage inherits its injected templates). Every span it yields is
    validated per observation in _fb_spans, so a wrong declaration can only
    leave harness text untagged, never mislabel terminal output."""
    for name in dict.fromkeys([agent, "baseline_kira"]):
        if not name:
            continue
        p = _agent_file(name)
        fn = load_declared_feedback_sections(p) if p else None
        if fn is not None:
            return fn
    return None


def _fb_spans(fb_fn, obs: str) -> list[tuple[str, str]]:
    """Validated harness-authored feedback spans inside one observation."""
    if fb_fn is None:
        return []
    try:
        decl = fb_fn(obs)
    except Exception:
        return []
    if not decl or not validate_feedback_sections(decl, obs)[0]:
        return []
    return [(str(n), str(t)) for n, t in decl]


def _observation_spans(t: int, obs: str, obs_seen: dict[str, int], fb_fn) -> list[dict]:
    """The obs:<t> span of turn t's observation and the fb:<t>:<name> spans
    inside it; none for observations under 24 characters. Span text is
    stripped (chat templates trim trailing message whitespace, which breaks a
    verbatim find on raw terminal output); `occurrence` disambiguates
    byte-identical observations at different turns. Updates `obs_seen`."""
    if len(obs) < 24:
        return []
    obs_span = obs.strip()
    occ = obs_seen.get(obs_span, 0) + 1
    obs_seen[obs_span] = occ
    extra = {"occurrence": occ} if occ > 1 else {}
    spans = [{"name": f"obs:{t}", "text": obs_span, "msg_index": 2 * t + 2, **extra}]
    for fb_name, fb_text in _fb_spans(fb_fn, obs):
        spans.append({"name": f"fb:{t}:{fb_name}", "text": fb_text,
                      "msg_index": 2 * t + 2, "within": obs_span, **extra})
    return spans


def _obs_text(step: dict) -> str:
    raw = step.get("observation")
    if not raw:
        return "(no output)"
    try:
        doc = json.loads(raw) if isinstance(raw, str) else raw
        parts = [r.get("content", "") for r in doc.get("results", [])]
        return "\n".join(p for p in parts if p) or "(no output)"
    except (json.JSONDecodeError, AttributeError, TypeError):
        return str(raw)


def _assistant_text(step: dict) -> str:
    parts = [step.get("message") or ""]
    tc = step.get("tool_calls")
    if tc:
        try:
            parts.append(json.dumps(tc if not isinstance(tc, str)
                                    else json.loads(tc), ensure_ascii=False))
        except (json.JSONDecodeError, TypeError):
            parts.append(str(tc))
    return "\n".join(p for p in parts if p)


def _forced_text(step: dict, capture_era: bool = False) -> str:
    # verbatim serialization: the rendered generation prompt ends "<think>\n",
    # and the worker generated reasoning, closed the think block, then wrote
    # the message ("<think>\n{R}</think>\n\n{msg}" per the qwen3 template;
    # kira stores R ending "\n" and msg stripped).
    # In a capture-era trajectory (some step carries reasoning_content) a step
    # WITHOUT one was served without a think block: the template then reads
    # "<think>\n\n</think>\n\n{msg}", i.e. "\n</think>\n\n" + msg after the
    # prompt's "<think>\n". Legacy trajectories force the message alone.
    r = step.get("reasoning_content") or ""
    body = _assistant_text(step)
    if not r:
        return "\n</think>\n\n" + body if capture_era else body
    if not r.endswith("\n"):
        r += "\n"
    return r + "</think>\n\n" + body if body else r + "</think>"


def _capture_era(agent_steps: list[dict]) -> bool:
    return any(s.get("reasoning_content") for s in agent_steps)


def extend_trial(traj_path: str | Path, store_dir: str | Path, eid: str,
                 eng=None, turn_cap: int = 999) -> int:
    """Append turns beyond an episode's existing n_turns_replayed (an earlier
    capped ingest). Rebuilds the message prefix from the trajectory, writes
    messages + token rows for the NEW turns only, refreshes the manifest's
    turn count and full span list. Returns the number of turns added.
    Replay resumes per-turn (attention and logits done-markers are per turn),
    so completed turns are never re-decoded."""
    store_dir = Path(store_dir)
    epdir = store_dir / eid
    man = json.loads((epdir / "manifest.json").read_text())
    meta = man.get("meta") or {}
    k_done = int(meta.get("n_turns_replayed", 0))
    if eng is None:
        eng = detok_engine()
    doc = json.loads(Path(traj_path).read_text())
    steps = doc.get("steps", [])
    initial = str(steps[0].get("message") or "") if steps else ""
    agent_steps = [s for s in steps[1:] if s.get("source") == "agent"][:turn_cap]
    if len(agent_steps) <= k_done:
        return 0

    store = TraceStore(store_dir)
    # flush() numbers parts from an in-memory counter -- seed it from disk so
    # the appended rows land in a NEW parquet part instead of clobbering 00000
    store._parts[eid] = len(list(epdir.glob("tokens-*.parquet")))
    msgs = [{"role": "user", "content": initial}]
    # spans for turns already replayed must stay EXACTLY as those turns used
    # them (replay bakes span region ids into attn_tokens.parquet per turn):
    # keep the manifest list wholesale and only APPEND spans for new turns.
    # Resolve fresh only when the manifest has no spans at all.
    spans = list(meta.get("prompt_sections_requests") or [])
    if not spans:
        declared = _declared_sections(meta.get("memory"), initial)
        spans = [{"name": n, "text": t, "msg_index": 0}
                 for n, t in sections_for(initial, declared)]
    fb_fn = _declared_feedback(meta.get("memory"))
    obs_seen: dict[str, int] = {}
    added = 0
    cap = _capture_era(agent_steps)
    for t, step in enumerate(agent_steps):
        if t >= k_done:
            forced = _forced_text(step, cap)
            toks = eng.tokenize(forced, add_special=False)
            if toks:
                store.write_messages(eid, t, list(msgs))
                append_forced(store, eid, t, toks)
                added += 1
        msgs.append({"role": "assistant", "content": _assistant_text(step)})
        obs = _obs_text(step)
        msgs.append({"role": "user", "content": obs})
        # occurrences count over ALL turns (so appended spans get the right
        # fallback index), but only the NEW turns' spans are appended
        new_spans = _observation_spans(t, obs, obs_seen, fb_fn if t >= k_done else None)
        if t >= k_done:
            spans += new_spans
    if added:
        store.flush(eid)
        meta["n_turns_replayed"] = len(agent_steps)
        meta["prompt_sections_requests"] = spans
        store.write_manifest(EpisodeManifest(episode_id=eid, meta=meta))
    return added


def ingest_trial(traj_path: str | Path, store_dir: str | Path, eid: str,
                 meta_extra: dict, eng=None, turn_cap: int = 14) -> bool:
    """One trial -> one multi-turn episode. Returns False if unusable."""
    if eng is None:
        eng = detok_engine()
    doc = json.loads(Path(traj_path).read_text())
    steps = doc.get("steps", [])
    if not steps or steps[0].get("source") != "user":
        return False
    initial = str(steps[0].get("message") or "")
    agent_steps = [s for s in steps[1:] if s.get("source") == "agent"]
    if not agent_steps:
        return False
    n_total = len(agent_steps)
    agent_steps = agent_steps[:turn_cap]

    store = TraceStore(store_dir)
    msgs = [{"role": "user", "content": initial}]
    declared = _declared_sections(meta_extra.get("memory"), initial)
    # msg_index pins each span to its message (initial prompt = 0, obs:t =
    # 2t+2): replay searches only that message's rendered range, so a short
    # observation whose text recurs verbatim inside an earlier message (e.g.
    # embedded in a confirmation echo) can never mislocate, and a span whose
    # message is not yet in the prompt is skipped, not false-matched.
    spans = [{"name": n, "text": t, "msg_index": 0}
             for n, t in sections_for(initial, declared)]
    fb_fn = _declared_feedback(meta_extra.get("memory"))
    obs_seen: dict[str, int] = {}
    wrote_any = False
    cap = _capture_era(agent_steps)
    for t, step in enumerate(agent_steps):
        forced = _forced_text(step, cap)
        toks = eng.tokenize(forced, add_special=False)
        if toks:
            store.write_messages(eid, t, list(msgs))
            append_forced(store, eid, t, toks)
            wrote_any = True
        # extend the prefix for the next turn
        msgs.append({"role": "assistant", "content": _assistant_text(step)})
        obs = _obs_text(step)
        msgs.append({"role": "user", "content": obs})
        spans += _observation_spans(t, obs, obs_seen, fb_fn)
    if not wrote_any:
        return False
    store.flush(eid)
    meta = {"task": "tb2", "provider_generated": True,
            "n_turns_total": n_total,
            "n_turns_replayed": len(agent_steps),
            "prompt_sections_requests": spans, **meta_extra}
    store.write_manifest(EpisodeManifest(episode_id=eid, meta=meta))
    return True
