"""vitals add-on for the Meta-Harness optimizer.

vitals proper is the instrument (engine, replay, read maps).
This add-on is the integration with one harness optimizer, Meta-Harness, in
three adapters plus the analyses built on them:

- ``contracts``: the section-declaration contracts (``prompt_sections``,
  ``feedback_sections``) the instrument asks a harness to implement, with
  validators and structural fallbacks for both task families.
- ``ingest``: Meta-Harness logs and harbor trajectories -> vitals episodes.
- ``replay``: drivers that teacher-force those episodes through the BF16
  reference (the in-loop replay for classification, Terminal-Bench and HANDBOOK.md).
- ``digest``: the per-section read digest (lift, peak, consult measures, the
  delta against the parent) fed back to the proposer in the read-informed run.
- ``analysis``: the section tables built from replayed stores, the
  terminal-bench read maps and event views, and the HANDBOOK.md reports.

The same three-adapter shape (contract, ingest, digest hook) is what another
optimizer would implement to sit under vitals. Paths come from ``config.CFG``.
Entry point: ``vitals metaharness <command>``.
"""
