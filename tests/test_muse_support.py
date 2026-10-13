"""Support for post-b9910 llama.cpp builds and Harmony-style templates (Muse Glimmer 30B).

Layout detection and the verbatim parser run anywhere. The template / region checks need the Muse
GGUF and its llama.cpp build; they run in a subprocess (two libllama builds must not share a process)
and are skipped unless VITALS_TEST_MUSE_LIB (that build's libllama) and VITALS_TEST_MUSE_GGUF (a glob
of the GGUF files) are set.
"""

from __future__ import annotations

import ctypes as C
import json
import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import pytest

from vitals.engine import binding as B
from vitals.server.toolcalls import parse_assistant_output

MUSE_LIB = os.environ.get("VITALS_TEST_MUSE_LIB", "")
MUSE_GGUF = sorted(glob(os.environ.get("VITALS_TEST_MUSE_GGUF", "")))


class _DefaultParams:
    """Stands in for llama_context_default_params() of a given build."""

    def __init__(self, cls: type, **values):
        self.restype = None
        self.struct = cls()
        for k, v in values.items():
            setattr(self.struct, k, v)

    def __call__(self):
        return B.LlamaContextParams.from_buffer_copy(bytes(self.struct))


def _defaults(cls: type, **extra) -> _DefaultParams:
    return _DefaultParams(cls, n_ctx=512, n_batch=2048, n_ubatch=512, n_seq_max=1, n_threads=4, n_threads_batch=4,
                          flash_attn_type=-1, **extra)


def test_context_params_layout_detection(monkeypatch):
    monkeypatch.delenv("VITALS_LLAMA_CTX_ABI", raising=False)
    assert C.sizeof(B.LlamaContextParamsV2) == C.sizeof(B.LlamaContextParams)
    old = type("Lib", (), {})()
    old.llama_context_default_params = _defaults(B.LlamaContextParams)
    assert B.context_params_layout(old) is B.LlamaContextParams
    new = type("Lib", (), {})()
    new.llama_context_default_params = _defaults(B.LlamaContextParamsV2, n_outputs_max_per_seq=1)
    assert B.context_params_layout(new) is B.LlamaContextParamsV2
    # flash_attn_type sits 4 bytes later in the newer layout
    assert B.LlamaContextParamsV2.flash_attn_type.offset == B.LlamaContextParams.flash_attn_type.offset + 4
    assert B.LlamaContextParamsV2.cb_eval.offset == B.LlamaContextParams.cb_eval.offset
    monkeypatch.setenv("VITALS_LLAMA_CTX_ABI", "v1")
    assert B.context_params_layout(new) is B.LlamaContextParams


def test_channel_parser_verbatim_mode():
    gen = " to=self<|message|>\nTHINK\n\n<|eom|><|start|>assistant to=user<|message|>\n\nANS\n<|eot|>"
    assert parse_assistant_output(gen) == ("ANS", "THINK", [])
    assert parse_assistant_output(gen, strip=False) == ("\n\nANS\n", "\nTHINK\n\n", [])


MUSE_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from vitals.backends.llamacpp import LlamaCppBackend
from vitals.engine.binding import LlamaEngine
from vitals.engine.template import ChatTemplate
be = LlamaCppBackend.__new__(LlamaCppBackend)
be.engine = LlamaEngine(sys.argv[2], n_ctx=512, n_gpu_layers=0)
be.template = ChatTemplate.from_engine(be.engine)
tools = [{"type": "function", "function": {"name": n, "description": n,
          "parameters": {"type": "object", "properties": {"p": {"type": "string"}}}}} for n in ("ls", "cat")]
conv = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q1 about the files"},
        {"role": "assistant", "content": "", "reasoning_content": "R1 thinking",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ls", "arguments": {"p": "/"}}},
                        {"id": "c2", "type": "function", "function": {"name": "cat", "arguments": {"p": "a"}}}]},
        {"role": "tool", "content": "OBS1 listing", "tool_call_id": "c1"},
        {"role": "tool", "content": "OBS2 contents", "tool_call_id": "c2"},
        {"role": "assistant", "content": "A1 answer", "reasoning_content": "R2"},
        {"role": "user", "content": "Q2 follow-up"}]
ctx, toks = be.render_turn(conv, tools=tools, spans=[{"name": "obs2", "text": "OBS2 contents"}])
regions = [[r.id, r.token_start, r.token_end] for r in ctx.regions]
text = {r.id: be.engine.detokenize(toks[r.token_start:r.token_end]) for r in ctx.regions if r.token_start is not None}
p1 = be.template.render(conv[:2], add_generation_prompt=True)
p2 = be.template.render(conv, add_generation_prompt=True)
be._tok_cache = None
be._tokenize_prompt(p1)
t2, offs, pieces = be._tokenize_prompt(p2)
print(json.dumps({"regions": regions, "text": text, "bos": be._bos_id(), "first": toks[0],
                  "reuse_equal": t2 == be.engine.tokenize(p2), "bytes_ok": b"".join(pieces) == p2.encode(),
                  "reused": be.tokenize_stats["reused_tokens"], "layout": be.engine.lib.vitals_ctx_params.__name__}))
"""


@pytest.mark.skipif(not (MUSE_LIB and Path(MUSE_LIB).exists() and MUSE_GGUF),
                    reason="set VITALS_TEST_MUSE_LIB and VITALS_TEST_MUSE_GGUF to the Muse libllama and GGUF")
def test_muse_regions_bos_and_prefix_tokenization():
    env = {**os.environ, "VITALS_LLAMA_LIB": MUSE_LIB}
    out = subprocess.run([sys.executable, "-c", MUSE_SCRIPT, str(Path(__file__).resolve().parents[1]), MUSE_GGUF[0]],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["layout"] == "LlamaContextParamsV2"
    ids = [r[0] for r in res["regions"]]
    assert ids[:8] == ["sys_0", "user_0", "gen_0", "obs_0", "obs_1", "gen_1", "user_1", "cue_0"]
    t = res["text"]
    assert "to=self" in t["gen_0"] and "to=ls" in t["gen_0"] and "to=cat" in t["gen_0"]   # one message, three blocks
    assert t["obs_1"].startswith("<|start|>tool cat") and t["cue_0"] == "<|start|>assistant"
    assert t["span:obs2"] == "OBS2 contents"
    assert res["first"] == res["bos"] and t["sys_0"].startswith("<|begin_of_text|>")      # BOS kept, zero width
    assert res["reuse_equal"] and res["bytes_ok"] and res["reused"] > 0
