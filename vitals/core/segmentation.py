"""Message-role → region segmentation and compaction detection.

The chat template is the region annotation: OpenAI-compatible
message structure already tags every span with a role, and off-policy regions
are precisely the non-assistant content that entered mid-episode. Harnesses may
pass richer annotations; they never must.
"""

from __future__ import annotations

import json
import re
from typing import Any

from vitals.core.messages import content_text
from vitals.core.schema import Region, RegionKind

Message = dict[str, Any]

_ROLE_TO_KIND = {
    "system": RegionKind.SYSTEM,
    "developer": RegionKind.SYSTEM,
    "user": RegionKind.USER,
    "assistant": RegionKind.REASONING,
    "tool": RegionKind.OBSERVATION,
    "function": RegionKind.OBSERVATION,
}

# Heuristic error detector for observation content. Harnesses can override via
# an explicit `is_error` flag in message metadata; this is the fallback for
# tool-role results without structured markers. Phrasings verified against real
# harness output: pi says "Command exited with code N", Hermes "Script exited
# with code N" / "failed with exit code N"; Python exception classes
# (SyntaxError:) don't match \bError: because the word boundary fails mid-word.
_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|\bFAILED\b|\bError:|\berror:"
    r"|\b[A-Z]\w*Error\b|\bException\b"
    r"|exit(?:ed)?\s+(?:with\s+)?(?:code|status)\s+[1-9]\d*"
    r"|failed with exit code [1-9]\d*"
    r"|non-zero exit|Segmentation fault|panicked at",
)


def looks_like_error(text: str) -> bool:
    return bool(_ERROR_RE.search(text))


# Flat-history harnesses (mini-SWE-agent and kin) inject tool output as USER-role
# messages wrapped in distinctive markup. These markers reclassify such messages
# as OBSERVATION regions — and carry the actual exit code, which beats the
# generic error regex. Covers mini-swe-agent's textbased (<returncode>...) and
# JSON ({"returncode": ...) observation templates.
_OBS_MARKER_RE = re.compile(r"^\s*(?:<(?:exception|returncode|output|warning)>|\{\s*\"returncode\")")
_RETURNCODE_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>|\"returncode\":\s*(-?\d+)")
_EXCEPTION_RE = re.compile(r"^\s*<exception>")

# Terminus-2 (Harbor) uses a text protocol: injected feedback arrives as
# USER-role messages in three flavours we tag via meta["obs_type"] so downstream
# analysis can attribute each to the scaffold component that governs it:
#   terminal     -> "New Terminal Output:" / "Current terminal state:"  (context mgmt)
#   budget_error -> the model exceeded the output-length cap             (output budget)
#   parse_error  -> the action block failed to parse as JSON/XML         (completion/format)
# All three are OBSERVATION regions (externally injected). Markers are specific
# enough not to collide with the system-prompt template (which is the first
# user message and is guarded separately).
_TERMINUS_OBS_RE = re.compile(r"New Terminal Output:|Current [Tt]erminal (?:state|Screen):")
_TERMINUS_BUDGET_RE = re.compile(
    r"exceeded the maximum output length|NONE of the actions you just requested were performed")
_TERMINUS_PARSE_RE = re.compile(
    r"No valid JSON found|parsing errors|Extra text detected before JSON"
    r"|Previous response had warnings")


def _structured_exit_code(text: str) -> int | None:
    """Exit code from a JSON tool-result body (Hermes: {"output":..,"exit_code":N};
    also handles "returncode"/"exit_status" keys). None if not JSON or no such key."""
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    for k in ("exit_code", "returncode", "exit_status", "return_code"):
        if isinstance(obj.get(k), int):
            return obj[k]
    return None


def classify_observation(text: str) -> tuple[bool, bool | None]:
    """(is_observation, is_error) from harness observation markup.
    is_error is None when the text is not recognizable observation content."""
    code = _structured_exit_code(text)
    if code is not None:  # JSON tool result with an exit-code field (Hermes)
        return True, code != 0
    if not _OBS_MARKER_RE.search(text):
        return False, None
    if _EXCEPTION_RE.search(text):
        return True, True
    m = _RETURNCODE_RE.search(text)
    if m:
        c = int(m.group(1) or m.group(2))
        return True, c != 0
    return True, None


def terminus_feedback(text: str) -> tuple[str | None, bool]:
    """Terminus-2 injected-feedback classifier.
    Returns (obs_type, is_error): obs_type in {terminal, budget_error, parse_error}
    or None if the text is not recognizable Terminus feedback."""
    if _TERMINUS_BUDGET_RE.search(text):
        return "budget_error", True
    if _TERMINUS_PARSE_RE.search(text):
        return "parse_error", True
    if _TERMINUS_OBS_RE.search(text):
        return "terminal", looks_like_error(text)
    return None, False


def _content_text(message: Message) -> str:
    return content_text(message, tool_calls=True)


def segment_messages(messages: list[Message], tools: list[dict] | None = None) -> list[Region]:
    """Parse a message history into regions with char spans.

    Token spans are resolved later by the backend at prefill (tokenizer-aware).
    Turn counting: a turn is one assistant message and everything since the
    previous assistant message.
    """
    regions: list[Region] = []
    counters: dict[RegionKind, int] = {}
    turn = 0
    cursor = 0
    first_user_seen = False  # the initial prompt/task blob is never an observation

    if tools:
        regions.append(
            Region(id="schema_0", kind=RegionKind.TOOL_SCHEMA, role="tools", turn=0,
                   char_start=0, char_end=0, meta={"n_tools": len(tools)})
        )

    for msg in messages:
        role = str(msg.get("role", "user"))
        kind = _ROLE_TO_KIND.get(role, RegionKind.USER)
        text = _content_text(msg)
        marker_error: bool | None = None
        obs_type: str | None = None
        # The first user message is Terminus-2's system prompt (it merges the
        # task and EXPLAINS the feedback markers), so skip the Terminus classifier
        # there; generic flat-history markers (mini-SWE, Hermes) still apply at
        # any position.
        is_first_user = kind is RegionKind.USER and not first_user_seen
        if kind in (RegionKind.USER, RegionKind.OBSERVATION):
            if not is_first_user:
                obs_type, marker_error = terminus_feedback(text)
            if obs_type is not None:
                if kind is RegionKind.USER:
                    kind = RegionKind.OBSERVATION
            else:
                is_obs, marker_error = classify_observation(text)
                if is_obs and kind is RegionKind.USER:
                    kind = RegionKind.OBSERVATION
        if is_first_user:
            first_user_seen = True
        n = counters.get(kind, 0)
        counters[kind] = n + 1
        prefix = {
            RegionKind.SYSTEM: "sys",
            RegionKind.USER: "user",
            RegionKind.OBSERVATION: "obs",
            RegionKind.REASONING: "gen",
            RegionKind.TOOL_SCHEMA: "schema",
            RegionKind.FEW_SHOT: "shot",
        }[kind]
        if "is_error" in msg:
            is_error = bool(msg.get("is_error"))
        elif marker_error is not None:
            is_error = marker_error
        else:
            is_error = kind is RegionKind.OBSERVATION and looks_like_error(text)
        regions.append(
            Region(
                id=f"{prefix}_{n}",
                kind=kind,
                role=role,
                turn=turn,
                char_start=cursor,
                char_end=cursor + len(text),
                is_error=is_error,
                meta={"obs_type": obs_type} if obs_type else {},
            )
        )
        cursor += len(text)
        if role == "assistant":
            turn += 1

    return regions


def _pairs(messages: list[Message]) -> list[tuple[str, str]]:
    return [(str(m.get("role", "")), _content_text(m)) for m in messages]


def extends(prev: list[Message], new: list[Message]) -> bool:
    """True when `new` extends `prev` (same-episode continuation).

    The last prev message may be extended in place (streaming assistant text),
    so it must be a prefix of its counterpart rather than strictly equal.
    """
    p, q = _pairs(prev), _pairs(new)
    if len(q) < len(p):
        return False
    head = len(p) - 1
    if p[:head] != q[:head]:
        return False
    if head >= 0:
        (pr, pc), (qr, qc) = p[head], q[head]
        if pr != qr or not qc.startswith(pc):
            return False
    return True


def detect_compaction(prev: list[Message], new: list[Message]) -> bool:
    """Compaction heuristic: the new history is *shorter* than,
    but semantically derived from, the tracked history — same system head, fewer
    or rewritten middle messages. Fires the COMPACTION event upstream.

    Guardrail: a fresh task opener
    ([system, user]) from the same harness shares the system prompt and plenty
    of generic wording with any prior transcript — a 2-message history is a NEW
    episode, never a compaction. Real compactions carry the compressed past, so
    they are longer and overlap much harder."""
    if extends(prev, new):
        return False
    p, q = _pairs(prev), _pairs(new)
    if not p or len(q) < 3 or len(q) >= len(p):
        return False
    same_head = p[0] == q[0]
    prev_texts = " ".join(c for _, c in p)
    body = q[1:]
    overlap = sum(1 for _, c in body if c and (c in prev_texts or _token_overlap(c, prev_texts) > 0.5))
    return same_head and overlap >= max(1, len(body) // 2)


def _token_overlap(a: str, b: str) -> float:
    wa = set(a.split())
    if not wa:
        return 0.0
    wb = set(b.split())
    return len(wa & wb) / len(wa)
