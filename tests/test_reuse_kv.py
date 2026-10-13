"""KV-reuse (opt-in) golden equivalence + safety.

The reuse path must NEVER change outputs: when the new prompt exactly extends
the resident tokens it decodes only the delta; on any divergence it silently
falls back to a full re-prefill. Verified: greedy tokens and first-step logits
are bitwise identical with reuse_kv on vs off, across a 2-turn episode.

Known limitation (measured, structural): on the Qwen think-family templates the
generation cue writes think tokens into KV that history re-renders strip
(3.5: no-think block; 3.6: <think> opener), so cross-turn hits do not occur
there — every turn is a clean-miss fallback. Partial rollback cannot fix it:
llama_memory_recurrent::seq_rm rejects partial ranges on hybrid state. Reuse
pays off only for exact-extension workloads / pure-attention models.
"""

import os
from glob import glob

import numpy as np
import pytest

GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF"
             "/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))
pytestmark = pytest.mark.skipif(not GGUFS, reason="0.8B CI gguf not cached")


def _episode(reuse: bool):
    from vitals.backends.llamacpp import LlamaCppBackend
    from vitals.core.schema import CaptureSpec

    b = LlamaCppBackend(GGUFS[0], n_ctx=2048, reuse_kv=reuse)
    spec = CaptureSpec()
    msgs = [{"role": "user", "content": "Name three primary colors, one per line."}]
    out = []
    for _turn in range(2):
        ctx = b.prefill(msgs, spec)
        toks, first_logits = [], None
        while not b.done(ctx) and len(toks) < 24:
            step = b.step(ctx)
            if first_logits is None:
                first_logits = np.array(step.logits, dtype=np.float32).copy()
            t = int(np.argmax(step.logits))
            toks.append(t)
            b.advance(ctx, t)
        reply = b.detokenize([t for t in toks if not b.engine.is_eog(t)])
        out.append((toks, first_logits))
        msgs = msgs + [{"role": "assistant", "content": reply},
                       {"role": "user", "content": "Now one more."}]
    stats = dict(b.reuse_stats)
    b.close()
    return out, stats


def test_reuse_kv_golden_equivalent():
    base, _ = _episode(reuse=False)
    fast, stats = _episode(reuse=True)
    for turn in range(2):
        assert base[turn][0] == fast[turn][0], f"turn {turn}: token stream differs"
        assert np.array_equal(base[turn][1], fast[turn][1]), f"turn {turn}: logits differ"
    # bookkeeping sane: every decode accounted, no crash on miss-fallback
    assert stats["decoded_tokens"] > 0 and stats["hits"] + stats["misses"] >= 1
