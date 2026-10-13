# Vitals proposer kit

This directory is the complete Vitals interface provided to a Meta-Harness
proposer in a run with Vitals feedback.

It contains two files:

- `VITALS_SCHEMA.md`, which explains the raw replay artifacts and their
  interpretation
- `vitals_data.py`, a standalone pandas/pyarrow loader that requires neither
  the worker model nor the installed Vitals package

The kit deliberately contains no interpretation of past results, so the
proposer forms its own conclusions from the measurements.

The schema mirrors `../VITALS_SCHEMA.md`, with paths adjusted for the
standalone loader. The loader mirrors the relevant data-loading utilities.
Keep both copies synchronized when the source schema or store format changes.
