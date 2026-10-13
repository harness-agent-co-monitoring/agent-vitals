"""Ingest Meta-Harness text-classification eval outputs into a vitals store.

Input: a val.json/test.json written by the inner loop with per-prediction
logging -- its ``predictions`` list carries per-example ``prompt_text`` and
``full_response`` from the provider (or per-call ``llm_calls``). Output: one
vitals episode per worker call (messages.jsonl + schema-correct token parquet
with the generated tokens teacher-forceable), labels stamped in the manifest.
`vitals replay` then fills the attention and logit channels locally.

Token ids come from re-tokenizing the response with the same-vocab tokenizer;
generation-side stats columns are placeholders (ent/m2/js = 0) until the
replay logits pass overwrites them.

Usage:
  python -m vitals.addons.metaharness.ingest.tc --val-json <path> --store <dir> [--tag NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vitals.addons.metaharness.ingest import detok_engine
from vitals.core.schema import EpisodeManifest
from vitals.core.store import TraceStore
from vitals.replay import append_forced


def ingest(val_json: str | Path, store_dir: str | Path, tag: str = "ep") -> list[str]:
    doc = json.loads(Path(val_json).read_text())
    preds = doc.get("predictions") or []
    if not preds:
        raise SystemExit(f"{val_json}: no per-prediction records -- was the run "
                         "made with per-prediction logging?")
    eng = detok_engine()
    store = TraceStore(store_dir)
    made, workerless = [], 0
    # capture-era run: the reasoning channel was recorded per call, so a call
    # WITHOUT one was served by the upstream without a think block (the
    # "thinking" then sits in the content). Legacy runs never carry the key.
    capture_era = any("reasoning" in c for p in preds for c in (p.get("llm_calls") or []))
    for i, p in enumerate(preds):
        calls = p.get("llm_calls") or []
        if not calls and p.get("prompt_text") and p.get("full_response"):
            calls = [{"prompt": p["prompt_text"], "response": p["full_response"]}]
        if not calls:
            # worker-free prediction (e.g. a gated harness answered from
            # memory alone): no worker internals exist -- count it, no episode
            workerless += 1
            continue
        for k, call in enumerate(calls):
            eid = f"ep-{tag}-{i:04d}-c{k}"
            # teacher-force the assistant turn as the worker serialized it; the
            # qwen3-family generation prompt ends "<|im_start|>assistant\n<think>\n"
            reasoning = call.get("reasoning") or ""
            n_reason_toks = 0
            think_mode = "reasoning" if reasoning else ("nothink" if capture_era else "legacy")
            if reasoning:
                # verbatim reconstruction: the provider splits the generation at
                # "</think>", returning reasoning (ends "\n") and content (which
                # keeps whatever followed the tag, usually "\n\n"). Re-join with
                # nothing added so the forced sequence is the produced one.
                if not reasoning.endswith("\n"):
                    reasoning += "\n"
                think = reasoning + "</think>"
                forced = think + call["response"]
                n_reason_toks = len(eng.tokenize(think, add_special=False))
            elif capture_era:
                # served without a think block: the qwen3 template then reads
                # "<think>\n\n</think>\n\n{content}"; after the prompt's
                # "<think>\n" that is "\n</think>\n\n" + content, verbatim
                forced = "\n</think>\n\n" + call["response"]
            else:
                forced = call["response"]
            toks = eng.tokenize(forced, add_special=False)
            if not toks:
                continue
            store.write_messages(eid, 0, [{"role": "user", "content": call["prompt"]}])
            append_forced(store, eid, 0, toks)
            store.flush(eid)
            meta = {"task": doc.get("dataset"), "memory": doc.get("memory"),
                    "provider_model": doc.get("model"),
                    "example_idx": i, "call_idx": k, "n_calls": len(calls),
                    "n_reasoning_toks": n_reason_toks, "think_mode": think_mode,
                    "provider": call.get("provider"),
                    "reward": 1.0 if p.get("was_correct") else 0.0,
                    "prediction": p.get("prediction"), "target": p.get("target"),
                    "provider_generated": True}
            # proposer-declared fine-grained sections, attached to the call(s)
            # the declaration validated against in the eval loop (recorded in
            # prompt_sections_calls); older val.json without that field
            # declared against the last prompt built, i.e. the final call
            sec_calls = p.get("prompt_sections_calls")
            if sec_calls is None:
                sec_calls = [len(calls) - 1] if p.get("prompt_sections") else []
            if k in sec_calls and p.get("prompt_sections"):
                meta["prompt_sections"] = p["prompt_sections"]
            store.write_manifest(EpisodeManifest(episode_id=eid, meta=meta))
            made.append(eid)
    print(f"ingested {len(made)} episodes from {len(preds)} predictions "
          f"({workerless} worker-free) -> {store_dir}")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag", default="ep")
    a = ap.parse_args()
    ingest(a.val_json, a.store, a.tag)
