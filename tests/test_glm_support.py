"""Support for GLM-5.3-Flash (llama.cpp arch `glm5-next`) and the binding changes it needed.

The parser, template-filter and struct checks run anywhere. The region / tokenizer check needs the GGUF and a
llama.cpp build with glm5-next; it runs in a subprocess (two libllama builds must not share a process) and is
skipped unless VITALS_TEST_GLM_LIB (that build's libllama) and VITALS_TEST_GLM_GGUF (a glob of the GGUF shards)
are set.
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
from vitals.engine.nodes import for_model
from vitals.engine.template import ChatTemplate
from vitals.server.toolcalls import parse_tool_calls

GLM_LIB = os.environ.get("VITALS_TEST_GLM_LIB", "")
GLM_GGUF = sorted(glob(os.environ.get("VITALS_TEST_GLM_GGUF", "")))


def test_glm_tool_call_format():
    tools = [{"type": "function", "function": {"name": "cat", "parameters": {"properties": {"n": {"type": "integer"}}}}}]
    text = ("ok<tool_call>ls<arg_key>p</arg_key><arg_value>/</arg_value></tool_call>"
            "<tool_call>cat<arg_key>p</arg_key><arg_value>a && b < c</arg_value><arg_key>n</arg_key><arg_value>3</arg_value></tool_call>")
    content, calls = parse_tool_calls(text, tools)
    assert content == "ok"
    assert [c["function"]["name"] for c in calls] == ["ls", "cat"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"p": "a && b < c", "n": 3}
    assert parse_tool_calls("<tool_call>noargs</tool_call>")[1][0]["function"]["arguments"] == "{}"
    # the other tool-call conventions still parse
    assert parse_tool_calls('<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call>')[1][0]["function"]["name"] == "bash"
    assert parse_tool_calls("<tool_call>\n<function=w>\n<parameter=p>\nx\n</parameter>\n</function>\n</tool_call>")[1][0]["function"]["name"] == "w"


def test_tojson_filter_follows_the_template():
    hf = ChatTemplate("{{ x | tojson(ensure_ascii=False) }}{% for i in [1, 2] %}{% if i == 1 %}{% break %}{% endif %}{% endfor %}")
    assert hf.hf_tojson
    assert hf.render([], x={"b": "<é>", "a": 1}) == '{"b": "<é>", "a": 1}'    # key order kept, nothing escaped
    plain = ChatTemplate("{{ x | tojson }}")
    assert not plain.hf_tojson                                               # jinja's filter
    assert plain.render([], x={"b": "<", "a": 1}) == '{"a": 1, "b": "\\u003c"}'


def test_model_params_are_passed_opaquely():
    assert C.sizeof(B.LlamaModelParams) >= 80                    # the glm5-next era struct is 80 bytes
    assert B.LlamaModelParams.n_gpu_layers.offset == 16          # same offset in every known layout


def test_glm_node_map():
    nm = for_model("glm5-next", 45)
    assert nm.full_attn_layers == tuple(range(3, 45, 4))
    with pytest.raises(ValueError):
        nm.attn_node(4)                                          # a KDA (linear) layer


GLM_SCRIPT = r"""
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
ctx, toks = be.render_turn(conv, tools=tools, chat_kwargs={"reasoning_effort": "low"},
                           spans=[{"name": "obs2", "text": "OBS2 contents"}])
regions = [[r.id, r.token_start, r.token_end] for r in ctx.regions]
text = {r.id: be.engine.detokenize(toks[r.token_start:r.token_end]) for r in ctx.regions if r.token_start is not None}
p1 = be.template.render(conv[:2], add_generation_prompt=True)
p2 = be.template.render(conv, add_generation_prompt=True, tools=tools)
be._tok_cache = None
be._tokenize_prompt(p1)
t2, offs, pieces = be._tokenize_prompt(p2)
print(json.dumps({"regions": regions, "text": text, "arch": be.engine.arch, "n_layer": be.engine.n_layer,
                  "reuse_equal": t2 == be.engine.tokenize(p2), "bytes_ok": b"".join(pieces) == p2.encode(),
                  "reused": be.tokenize_stats["reused_tokens"]}))
"""


@pytest.mark.skipif(not (GLM_LIB and Path(GLM_LIB).exists() and GLM_GGUF),
                    reason="set VITALS_TEST_GLM_LIB and VITALS_TEST_GLM_GGUF to a glm5-next libllama and the GGUF")
def test_glm_regions_and_prefix_tokenization():
    env = {**os.environ, "VITALS_LLAMA_LIB": GLM_LIB}
    out = subprocess.run([sys.executable, "-c", GLM_SCRIPT, str(Path(__file__).resolve().parents[1]), GLM_GGUF[0]],
                         capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["arch"] == "glm5-next" and res["n_layer"] == 45
    ids = [r[0] for r in res["regions"]]
    assert ids == ["schema_0", "sys_0", "user_0", "gen_0", "obs_0", "obs_1", "gen_1", "user_1", "cue_0", "span:obs2"]
    t = res["text"]
    assert t["schema_0"].startswith("[gMASK]<sop><|system|>Reasoning Effort: Low<|system|>")   # synthesized blocks merged
    assert t["gen_0"].startswith("<|assistant|><think>R1 thinking</think><tool_call>ls")
    assert t["obs_0"] == "<|observation|><tool_response>OBS1 listing</tool_response>"
    assert t["obs_1"] == "<tool_response>OBS2 contents</tool_response>"
    assert t["cue_0"] == "<|assistant|><think>" and t["span:obs2"] == "OBS2 contents"
    assert res["reuse_equal"] and res["bytes_ok"] and res["reused"] > 0
