"""Harness event model: what the loop and policies react to."""

from __future__ import annotations

from vitals.core.schema import EventKind, HarnessEvent, Region, RegionKind


def events_for_new_regions(regions: list[Region], turn: int) -> list[HarnessEvent]:
    """Derive events from regions that entered the context this turn."""
    out: list[HarnessEvent] = []
    for r in regions:
        if r.kind is RegionKind.OBSERVATION:
            out.append(HarnessEvent(kind=EventKind.OBSERVATION_INJECTED, turn=turn, region_id=r.id))
            if r.is_error:
                out.append(HarnessEvent(kind=EventKind.ERROR_BLOCK, turn=turn, region_id=r.id))
    return out


def compaction(turn: int) -> HarnessEvent:
    return HarnessEvent(kind=EventKind.COMPACTION, turn=turn)
