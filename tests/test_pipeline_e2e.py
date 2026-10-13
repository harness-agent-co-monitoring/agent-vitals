"""End to end on the 0.8B model: import a small OpenAI log, run the pipeline with a probe, get flags whose
explanations carry both the read map and the probe split (plus the ablation confirmation), webhook deliveries,
and nothing new on a second run."""

from __future__ import annotations

import json
import os
import threading
from glob import glob
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS or not torch.cuda.is_available(),
                    reason="pinned engine build, 0.8B GGUF or CUDA not available")
def test_import_then_pipeline(tmp_path):
    from test_importer import log_lines

    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.monitor.checks import alarm_cache
    from vitals.monitor.pipeline import run_pipeline
    from vitals.monitor.probes import LinearProbe
    from vitals.replay.importer import import_log

    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/hook"
    log = tmp_path / "calls.jsonl"
    log.write_text("".join(json.dumps(x) + "\n" for x in log_lines()))
    store = tmp_path / "store"
    be = LlamaCppBackend(GGUFS[0], n_ctx=8192, n_seq_max=2, diagnostic_attention=True)
    try:
        assert import_log(log, store, be)["imported"] == 4
        layer = 23
        rng = np.random.default_rng(0)
        n_embd = be.engine.n_embd
        probe = LinearProbe("toy", layer, np.zeros(n_embd), np.ones(n_embd), rng.normal(size=n_embd) / np.sqrt(n_embd),
                            0.0, threshold=-1e9)                       # flags every decision
        kw = dict(backend=be, probes=[probe], webhook=url, out_dir=tmp_path / "alarms", ledger_out=tmp_path / "ledger",
                  confirm=True, log=lambda *_: None)
        r1 = run_pipeline(store, GGUFS[0], **kw)
        n_posts = len(got)
        r2 = run_pipeline(store, GGUFS[0], **kw)
    finally:
        be.close()
        srv.shutdown()
    assert r1["decisions"] >= 4 and r1["flags"] >= r1["decisions"] and r1["explained"] >= 4
    assert r1["webhook_sent"] == n_posts == r1["flags"] and r1["webhook_failed"] == 0
    assert r2["new"] == 0 and r2["explained"] == 0 and len(got) == n_posts
    cache = alarm_cache(tmp_path / "alarms", store)
    flags = [json.loads(x) for x in (cache / "flags.jsonl").read_text().splitlines()]
    assert all(f["explained"] for f in flags if f["kind"] == "probe")
    ex = [json.loads(x) for p in (cache / "explanations").glob("*.jsonl") for x in p.read_text().splitlines()]
    for e in ex:
        assert e["read_map"]["rows"], "read map view missing"
        assert e["probes"] and e["probes"][0]["rows"], "probe split view missing"
        assert "toy" in e["confirmed"] and "error" not in e["confirmed"]["toy"]
        assert any(c.startswith("Read map") for c in e["claims"]) and any("Probe 'toy'" in c for c in e["claims"])
    assert got[0]["read_map"] and got[0]["probes"][0]["top"]
    for e in ex:                                                    # read measures before the action
        assert e["read_map"]["lift_by"] == "reasoning_lift" and all(r["reasoning_lift"] for r in e["read_map"]["rows"])
        assert all(r["read_layers"] is not None for r in e["probes"][0]["rows"])
    from vitals.monitor.ledger import ledger_dir

    for f in ledger_dir(tmp_path / "ledger", store).glob("*/*.npz"):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        assert "reason_peaks" in meta and len(z["span_pos"]) == meta["prompt_tokens"]
        prompt_share = z["span"].astype(np.float64).mean(0)[np.array([n != "output" for n in z["sections"]])].sum()
        assert abs(float(z["span_pos"].sum()) - prompt_share) < 0.02
