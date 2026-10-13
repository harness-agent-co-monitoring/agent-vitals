# Dashboard

`vitals dashboard` serves a local page over every trace store and Meta-Harness run found under the directories you
give it. It needs the `server` extra.

```bash
vitals dashboard runs/traces --port 8040        # then open http://127.0.0.1:8040/
```

## Tabs

- **Read map.** One episode's sections by turn, colored by read lift, peak lift, consult ratio or consult contrast.
  Hover a cell for its values and hottest line. Click a cell to pin it: the side panel shows the section's text
  colored token by token, its measures across turns, and **Why this action** when `vitals explain` has explained
  that turn. Drag across the turn strip to zoom, click legend ranges to keep only those cells.
- **Store.** Every episode's mean reading of each section, with outcomes above the columns. Click a row to order
  episodes by it, or a cell to open that episode pinned on the section.
- **Monitor.** One store's episodes against a reference store (`vitals.analysis.drift`). An episode's section
  reading gets a robust z against the reference: its mean log2 lift minus the reference median, over 1.4826 × MAD
  floored at 0.25, for sections in at least three reference episodes. Alarms:
  - **over-read** and **under-read** when |z| passes the threshold slider;
  - **missing** when a section that 90% of reference episodes have is absent;
  - **new** when a declared section the reference never had appears (groups whose members vary by episode, like
    per-task documents, do not raise it).

  The control chart shows the selected section per episode against the reference band, and a sections × episodes
  map colors every cell by z.
- **Alarms.** The flags `vitals pipeline` wrote for the store, and for each flagged decision its explanation: the
  read map before the action next to each probe's split by section, with the claims
  ([monitoring](monitoring.md)).
- **Evolution.** A Meta-Harness run (a `logs/<run>` directory with `evolution_summary.jsonl`): score by iteration
  with parent links, score against context size, each candidate's section reading from its `vitals_digest.parquet`
  (absolute, or against its parent or a chosen reference), and per-task scores. Selecting a candidate shows its
  hypothesis, lineage, largest reading changes, and links to its stores.

Every setting lives in the URL, so a view can be shared as a link. Stores are checked every two seconds, so the page
follows a `vitals serve` store or a replay in progress.

## Caches

The dashboard never computes anything in the request path that a cache can hold:

- read maps are cached per episode in `readmap.parquet` beside the episode, or under `--cache-dir` (or
  `$VITALS_READMAP_CACHE`) when the stores should stay untouched, and recomputed when the episode's files change;
- explanations are read from the `vitals explain` cache (`--explain-dir`, default the explanations cache) and flags
  from the `vitals pipeline` cache (`--alarm-dir`, default the alarms cache). See
  [environment variables](reference/env.md) for the cache locations.

## API

The page reads JSON from the same server: `/api/sources`, `/api/changes`, `/api/stores/{sid}/episodes`,
`/api/stores/{sid}/episodes/{eid}/readmap`, `.../text`, `.../explanations`, `.../flags`, `/api/stores/{sid}/alarms`,
`/api/stores/{sid}/matrix`, `/api/monitor?target=&ref=&threshold=&harness_only=`, `/api/runs/{rid}`.
