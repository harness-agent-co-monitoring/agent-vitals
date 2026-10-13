"""End-to-end integration: the tested decode loop over the real llama.cpp backend.

Skipped automatically when the pinned engine build or the 0.8B GGUF is absent
(CI without GPU/model assets).
"""

import os
import subprocess
from glob import glob
from pathlib import Path

import numpy as np
import pytest

from vitals.core.schema import ActivationSpec, AttentionMode, AttentionSpec, CaptureSpec, CaptureUnsupported
from vitals.engine.sampler import SamplerConfig
from vitals.loop.decode import run_turn

VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))
pytestmark = pytest.mark.skipif(
    not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS,
    reason="pinned engine build or 0.8B GGUF not available",
)

PROMPT = "The three most common failure modes of coding agents are"
RAW = [{"role": "raw", "content": PROMPT}]


@pytest.fixture(scope="module")
def backend():
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(GGUFS[0], n_ctx=2048)
    yield be
    be.close()


def stock_completion(extra: list[str], n: int = 24) -> str:
    out = subprocess.run(
        [str(VENDOR / "llama-completion"), "-m", GGUFS[0], "-p", PROMPT, "-n", str(n),
         "-ngl", "99", "--no-warmup", "-no-cnv", *extra],
        capture_output=True, text=True, timeout=300,
    ).stdout
    if out.startswith(PROMPT):
        out = out[len(PROMPT):]
    return out.strip("\n")


def test_golden_greedy_equals_stock_tool(backend):
    """Empty CaptureSpec + no policy + no probes == stock llama.cpp output,
    token for token."""
    backend.sampler_config = SamplerConfig(temperature=0.0)
    res = run_turn(backend, RAW, max_tokens=24)
    ours = backend.detokenize(res.tokens).strip("\n")
    stock = stock_completion(["--temp", "0", "-s", "42"])
    assert ours == stock or stock.startswith(ours)


def test_stochastic_parity_with_native_chain(backend):
    """Same sampler chain + seed => same stochastic stream as the stock tool."""
    backend.sampler_config = SamplerConfig(temperature=0.8, top_k=40, top_p=0.95, min_p=0.05, seed=7)
    res = run_turn(backend, RAW, max_tokens=24)
    ours = backend.detokenize(res.tokens).strip("\n")
    stock = stock_completion(
        ["--temp", "0.8", "--top-k", "40", "--top-p", "0.95", "--min-p", "0.05", "-s", "7"]
    )
    assert ours == stock or stock.startswith(ours)


def test_capture_is_non_perturbing_and_shaped(backend):
    backend.sampler_config = SamplerConfig(temperature=0.0)
    base = run_turn(backend, RAW, max_tokens=16)

    captured: dict[int, dict] = {}
    spec = CaptureSpec(activations=ActivationSpec(layers=[5, 11]))
    res = run_turn(backend, RAW, capture=spec, max_tokens=16,
                   capture_sink=lambda i, s: captured.__setitem__(i, s.resid))

    assert res.tokens == base.tokens, "activation capture changed sampled tokens"
    assert set(captured) == set(range(16))
    first = captured[0]                      # prefill forward: (n_prompt, d)
    assert set(first) == {5, 11} and first[5].shape[-1] == 1024 and first[5].ndim == 2
    later = captured[3]                      # single-token forward: (d,) or (1, d)
    assert np.asarray(later[11]).reshape(-1).shape == (1024,)
    assert all(np.isfinite(m.ent) for m in res.metrics)


def test_long_prompt_capture_spans_all_ubatches(backend):
    """Regression: prompts longer than n_ubatch (512) are evaluated as several
    micro-batch graphs; capture must concatenate them, not keep only the last."""
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 80  # ~720 tokens
    captured: dict[int, dict] = {}
    spec = CaptureSpec(activations=ActivationSpec(layers=[11]))
    backend.sampler_config = SamplerConfig(temperature=0.0)
    ctx_probe = backend.prefill([{"role": "raw", "content": long_prompt}], CaptureSpec())
    assert ctx_probe.n_prompt > 512, "prompt must exceed n_ubatch for this test"
    res = run_turn(backend, [{"role": "raw", "content": long_prompt}], capture=spec, max_tokens=4,
                   capture_sink=lambda i, s: captured.__setitem__(i, s.resid))
    assert res.tokens
    assert captured[0][11].shape == (ctx_probe.n_prompt, 1024)


def test_attention_capture_needs_diagnostic_mode(backend):
    spec = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.PER_HEAD, layers=[3]))
    with pytest.raises(CaptureUnsupported, match="diagnostic"):
        run_turn(backend, RAW, capture=spec, max_tokens=4)


def test_attention_capture_in_diagnostic_mode():
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(GGUFS[0], n_ctx=2048, diagnostic_attention=True)
    try:
        # linear-attention layer refused with a useful message
        bad = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.PER_HEAD, layers=[4]))
        with pytest.raises(CaptureUnsupported, match="linear-attention"):
            run_turn(be, RAW, capture=bad, max_tokens=2)

        got: dict[int, dict] = {}
        spec = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.PER_HEAD, layers=[3, 11]))
        res = run_turn(be, RAW, capture=spec, max_tokens=6,
                       capture_sink=lambda i, s: got.__setitem__(i, s.attn))
        assert res.tokens and set(got[0]) == {3, 11}
        a0 = got[0][3]          # prefill: (heads, T_q, n_kv), already sliced un-padded
        n_prompt = a0.shape[-2]
        assert a0.shape == (8, n_prompt, n_prompt)
        assert np.allclose(a0.astype(np.float32).sum(-1), 1.0, atol=1e-3)
        a3 = got[3][11]         # step: (heads, n_kv) or (heads, 1, n_kv)
        rows = a3.astype(np.float32).reshape(8, -1)
        assert np.allclose(rows.sum(-1), 1.0, atol=1e-3)
    finally:
        be.close()


def test_region_pooled_attention_on_long_prompts():
    """REGION_POOLED uses cheap last-row capture and must survive multi-ubatch
    prefills (>512 tokens) where full-matrix parts can't concat."""
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(GGUFS[0], n_ctx=2048, diagnostic_attention=True)
    try:
        be.sampler_config = SamplerConfig(temperature=0.0)
        long_prompt = "The quick brown fox jumps over the lazy dog. " * 80  # ~720 tokens
        spec = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.REGION_POOLED))
        res = run_turn(be, [{"role": "raw", "content": long_prompt}], capture=spec, max_tokens=6)
        shares = [tm.attn_share for tm in res.metrics]
        n_regions = len(be.last_ctx.regions)  # raw path: 1 region
        assert all(s is not None and len(s) == n_regions + 1 for s in shares)
        assert all(abs(sum(s) - 1.0) < 2e-2 for s in shares)
        assert shares[0][-1] == 0.0 and shares[-1][-1] > 0.0
    finally:
        be.close()


def test_per_head_full_matrices_assemble_across_ubatches():
    """PER_HEAD keeps full matrices; multi-ubatch prefill parts are zero-padded
    on the kv axis and concatenated on the query axis."""
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(GGUFS[0], n_ctx=2048, diagnostic_attention=True)
    try:
        be.sampler_config = SamplerConfig(temperature=0.0)
        long_prompt = "A stitch in time saves nine. " * 90  # ~720 tokens
        got: dict[int, dict] = {}
        spec = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.PER_HEAD, layers=[11]))
        run_turn(be, [{"role": "raw", "content": long_prompt}], capture=spec, max_tokens=2,
                 capture_sink=lambda i, s: got.__setitem__(i, s.attn))
        a0 = got[0][11].astype(np.float32)  # (heads, n_prompt, n_kv)
        n_prompt = be.last_ctx.n_prompt
        assert n_prompt > 512
        assert a0.shape[0] == 8 and a0.shape[1] == n_prompt
        # every query row's used-kv prefix sums to ~1 (zero padding beyond mask)
        rows = a0.sum(axis=-1).reshape(-1)
        assert np.allclose(rows, 1.0, atol=1e-2)
    finally:
        be.close()


def test_chat_template_applied_for_message_lists(backend):
    backend.sampler_config = SamplerConfig(temperature=0.0)
    msgs = [{"role": "user", "content": "Say OK and nothing else."}]
    ctx = backend.prefill(msgs, CaptureSpec())
    assert "Say OK and nothing else." in ctx.prompt_text
    assert ctx.prompt_text != msgs[0]["content"], "template should wrap the message"
    assert ctx.n_prompt > len(backend.engine.tokenize(msgs[0]["content"]))


def test_region_token_spans_exact_and_contiguous(backend):
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test in utils.py."},
        {"role": "assistant", "content": "Let me look at the file."},
        {"role": "user", "content": "<returncode>1</returncode>\n<output>\nAssertionError\n</output>"},
    ]
    ctx = backend.prefill(msgs, CaptureSpec())
    regions = ctx.regions
    assert regions[-1].id == "cue_0" and regions[-1].meta.get("generation_cue")
    spans = [(r.token_start, r.token_end) for r in regions]
    assert all(s is not None and e is not None for s, e in spans), "spans must resolve"
    assert spans[0][0] == 0 and spans[-1][1] == ctx.n_prompt
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 == s2, "spans must tile the prompt contiguously"
    # each message's content is inside its span's detokenization
    for region, msg in zip(regions[:-1], msgs, strict=True):
        toks = backend.engine.tokenize(ctx.prompt_text)[region.token_start:region.token_end]
        assert msg["content"] in backend.detokenize(toks)
    # marker-tagged user message became an observation with error flag
    assert regions[3].kind.value == "observation" and regions[3].is_error


def test_merged_tool_blocks_get_subdivided_spans(backend):
    """qwen-family templates render consecutive `tool` messages as ONE user
    block of <tool_response> sections. Each observation region must still get
    its own token span."""
    tools = [{"type": "function", "function": {
        "name": "bash", "description": "run a shell command",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Run the checks."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c0", "type": "function",
             "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
        {"role": "tool", "content": "obs one: a.py"},
        {"role": "tool", "content": "obs two: b.py"},
        {"role": "tool", "content": "obs three: c.py"},
        {"role": "assistant", "content": "Now the tests."},
        {"role": "tool", "content": "obs four: 3 passed"},
    ]
    ctx = backend.prefill(msgs, CaptureSpec(), tools=tools)
    spans = [(r.token_start, r.token_end) for r in ctx.regions]
    assert all(s is not None and e is not None for s, e in spans), "all regions must resolve"
    assert spans[0][0] == 0 and spans[-1][1] == ctx.n_prompt
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 == s2, "spans must tile the prompt contiguously"
    toks = backend.engine.tokenize(ctx.prompt_text)
    obs = [r for r in ctx.regions if r.kind.value == "observation"]
    assert len(obs) == 4
    for region, content in zip(obs, ["obs one: a.py", "obs two: b.py",
                                     "obs three: c.py", "obs four: 3 passed"], strict=True):
        assert content in backend.detokenize(toks[region.token_start:region.token_end])


def test_jinja_template_tools_and_thinking(backend):
    """Multi-harness surface: tools render into the prompt, tool-call history
    renders, and enable_thinking controls the generation cue — none of which the
    C-API fallback can do."""
    assert backend.template is not None, "GGUF should carry a jinja chat template"
    tools = [{"type": "function", "function": {
        "name": "bash", "description": "run a shell command",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]
    msgs = [{"role": "user", "content": "List the files."}]
    ctx = backend.prefill(msgs, CaptureSpec(), tools=tools)
    assert "<tools>" in ctx.prompt_text and '"bash"' in ctx.prompt_text
    # no system message -> template synthesizes one for the tools: schema region
    assert ctx.regions[0].kind.value == "tool_schema"
    assert ctx.regions[0].meta.get("includes_tool_schema") == 1
    spans = [(r.token_start, r.token_end) for r in ctx.regions]
    assert all(s is not None for s, _ in spans) and spans[-1][1] == ctx.n_prompt
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 == s2

    # tool-call round trip in history: assistant tool_calls + tool result render
    history = msgs + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
        {"role": "tool", "content": "a.py\nb.py"},
    ]
    ctx2 = backend.prefill(history, CaptureSpec(), tools=tools)
    assert "<tool_call>" in ctx2.prompt_text and "a.py" in ctx2.prompt_text
    kinds = [r.kind.value for r in ctx2.regions]
    assert "observation" in kinds and "reasoning" in kinds

    # thinking control via template kwargs
    forced = backend.prefill(msgs, CaptureSpec(), chat_kwargs={"enable_thinking": True})
    assert forced.prompt_text.rstrip("\n").endswith("<think>")
    skipped = backend.prefill(msgs, CaptureSpec(), chat_kwargs={"enable_thinking": False})
    assert "<think>" in skipped.prompt_text and "</think>" in skipped.prompt_text


def test_attn_share_end_to_end_in_diagnostic_mode():
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(GGUFS[0], n_ctx=2048, diagnostic_attention=True)
    try:
        be.sampler_config = SamplerConfig(temperature=0.0)
        msgs = [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Name one prime number."},
        ]
        spec = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.PER_HEAD, layers=[3, 11]))
        res = run_turn(be, msgs, capture=spec, max_tokens=6)
        n_regions = len(be.last_ctx.regions)
        shares = [tm.attn_share for tm in res.metrics]
        assert all(s is not None and len(s) == n_regions + 1 for s in shares)
        assert all(abs(sum(s) - 1.0) < 1e-2 for s in shares)
        assert shares[0][-1] == 0.0, "no generated tokens yet at the first step"
        assert shares[-1][-1] > 0.0, "later steps attend to generated tokens"
    finally:
        be.close()


def test_named_spans_become_pooled_regions():
    """vitals.spans anchors resolve to extra regions with sane token spans, and
    attn_share vectors grow to cover them (regions + generated bucket)."""
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(GGUFS[0], n_ctx=2048, diagnostic_attention=True)
    try:
        pin = "Never delete the original files."
        msgs = [{"role": "system", "content": "You are a careful assistant."},
                {"role": "user", "content": f"Some preamble. {pin} Now say hi."}]
        spec = CaptureSpec(attention=AttentionSpec(mode=AttentionMode.REGION_POOLED, layers=[3]))
        res = run_turn(be, msgs, capture=spec, max_tokens=4,
                       spans=[{"name": "pin", "text": pin},
                              {"name": "ghost", "text": "NOT IN THE PROMPT"}])
        regions = be.last_ctx.regions
        named = [r for r in regions if r.id == "span:pin"]
        assert len(named) == 1, [r.id for r in regions]
        r = named[0]
        assert r.token_end > r.token_start > 0
        # the span's detokenized text round-trips to the anchor
        toks = be.engine.tokenize(be.last_ctx.prompt_text)
        assert pin in be.engine.detokenize(toks[r.token_start:r.token_end])
        # absent anchor is skipped, not fabricated
        assert not any(r.id == "span:ghost" for r in regions)
        # attn_share covers every region + the generated bucket
        shares = [tm.attn_share for tm in res.metrics]
        assert all(s is not None and len(s) == len(regions) + 1 for s in shares)
        # the pin share is a sane probability and keyed by region order
        pin_idx = regions.index(r)
        assert all(0.0 <= s[pin_idx] <= 1.0 for s in shares)
    finally:
        be.close()


def test_prompt_scores_capture():
    """CaptureSpec.prompt_scores: the prefill itself yields per-position
    ent/m2/surprisal for prompt tokens, aligned with the token ids."""
    from vitals.backends.llamacpp import LlamaCppBackend

    be = LlamaCppBackend(GGUFS[0], n_ctx=2048, n_gpu_layers=0)
    try:
        msgs = [{"role": "user", "content": "Name three primary colors and stop."}]
        res = run_turn(be, msgs, capture=CaptureSpec(prompt_scores=True), max_tokens=4)
        ctx = be.last_ctx
        assert ctx.prompt_stats is not None and ctx.prompt_token_ids
        n = len(ctx.prompt_token_ids)
        assert ctx.prompt_stats.shape == (n - 1, 3)
        ents, m2s, sups = ctx.prompt_stats.T
        assert (ents >= 0).all() and (sups >= 0).all()
        assert ((m2s >= -1e-9) & (m2s <= 1.0 + 1e-9)).all()
        # generation still works on top of the scored prefill
        assert res.tokens
        # and without the flag, nothing is populated
        run_turn(be, msgs, capture=CaptureSpec(), max_tokens=2)
        assert be.last_ctx.prompt_stats is None
    finally:
        be.close()


def test_prefix_reused_tokenization_matches_full(backend):
    """Consecutive turns of an append-only history: the tokens reused up to the
    last special-token boundary plus the freshly tokenized suffix equal a full
    tokenization, with exact byte offsets, and detokenize goes through the same
    pieces."""
    hist = [{"role": "system", "content": "You are terse. " * 40},
            {"role": "user", "content": "First question about " + "apples " * 30}]
    _, t1 = backend.render_turn(hist)
    assert backend.tokenize_stats["full"] >= 1
    hist = hist + [{"role": "assistant", "content": "Apples are fruit."},
                   {"role": "user", "content": "Second question about " + "pears " * 25}]
    before = dict(backend.tokenize_stats)
    ctx2, t2 = backend.render_turn(hist)
    assert backend.tokenize_stats["reused_tokens"] > before["reused_tokens"]
    assert backend.tokenize_stats["full"] == before["full"]          # the second render reused
    assert t2 == backend.engine.tokenize(ctx2.prompt_text)           # identical to a full tokenization
    assert t2[:len(t1) // 2] == t1[:len(t1) // 2]
    assert backend.detokenize(t2) == backend.engine.detokenize(t2)
    # a divergence before any boundary falls back to a full tokenization
    other = [{"role": "user", "content": "Completely different."}]
    ctx3, t3 = backend.render_turn(other)
    assert t3 == backend.engine.tokenize(ctx3.prompt_text)
