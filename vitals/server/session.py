"""Episode tracking.

Identity resolution order: explicit `X-Vitals-Episode` header / `metadata.episode_id`
in the body, then the fingerprint heuristic — a request whose messages *extend* a
tracked episode's history belongs to that episode; one whose history is shorter but
derived from it fires a COMPACTION event on the same episode.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from vitals.core.events import compaction, events_for_new_regions
from vitals.core.metrics import MetricsAccumulator
from vitals.core.schema import HarnessEvent
from vitals.core.segmentation import detect_compaction, extends, segment_messages


@dataclass
class Episode:
    id: str
    messages: list[dict] = field(default_factory=list)
    turn: int = 0
    created_at: float = field(default_factory=time.time)
    accumulator: MetricsAccumulator = field(default_factory=MetricsAccumulator)
    manifest_written: bool = False
    compactions: int = 0


class EpisodeTracker:
    """In-memory registry of live episodes (unbounded)."""

    def __init__(self) -> None:
        self.episodes: dict[str, Episode] = {}

    def _new(self, episode_id: str | None = None) -> Episode:
        ep = Episode(id=episode_id or f"ep-{uuid.uuid4().hex[:12]}")
        self.episodes[ep.id] = ep
        return ep

    def resolve(self, messages: list[dict], explicit_id: str | None = None) -> tuple[Episode, list[HarnessEvent]]:
        """Attach this request to an episode and derive its boundary events.
        The returned events cover messages that entered since the last turn.

        Fingerprint matching prefers the episode with the LONGEST matching
        prefix (most specific extension), not the first one found; otherwise,
        when many episodes share a system prompt, every continuation funnels
        into whichever episode was created first. Explicit `X-Vitals-Episode` /
        `episode_id` bypasses the heuristic entirely and is the recommended
        path for batch experiments."""
        ep = None
        if explicit_id:
            ep = self.episodes.get(explicit_id) or self._new(explicit_id)
        else:
            extenders = [c for c in self.episodes.values() if extends(c.messages, messages)]
            if extenders:
                ep = max(extenders, key=lambda c: len(c.messages))  # most specific
            else:
                ep = next(
                    (c for c in self.episodes.values() if detect_compaction(c.messages, messages)),
                    None,
                ) or self._new()

        events: list[HarnessEvent] = []
        if ep.messages and detect_compaction(ep.messages, messages):
            ep.compactions += 1
            events.append(compaction(ep.turn))
            new_msgs = messages  # post-compaction history is segmented again wholesale
        elif extends(ep.messages, messages):
            new_msgs = messages[len(ep.messages) :]
            # the shared boundary message may have grown in place; only fully new
            # messages become new regions
        else:
            new_msgs = messages

        if new_msgs:
            regions = segment_messages(new_msgs)
            events.extend(events_for_new_regions(regions, ep.turn))

        ep.messages = [dict(m) for m in messages]
        return ep, events
