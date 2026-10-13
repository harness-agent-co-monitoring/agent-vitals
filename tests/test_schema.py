
from vitals.core.schema import (
    AttentionMode,
    AttentionSpec,
    CaptureSpec,
    TokenMetrics,
)


def test_capture_spec_empty_property():
    assert CaptureSpec().is_empty is True
    assert CaptureSpec(top_logprobs=20).is_empty is False
    assert CaptureSpec(attention=AttentionSpec(mode=AttentionMode.REGION_POOLED)).is_empty is False
    assert CaptureSpec(probes=["commit:tool_call"]).is_empty is False


def test_token_metrics_row_contract():
    tm = TokenMetrics(episode_id="e", turn=1, step=2, token_id=3, ent=0.1, m2=0.9, js=0.0, fork=False)
    row = tm.to_row()
    assert set(row) == {
        "episode_id", "turn", "step", "token_id", "ent", "m2", "js", "fork",
        "kl_ref", "region_id", "attn_share", "probes", "t_ms",
        # v0.2.0: signed evidence (nullable; populated when top_logprobs > 0)
        "surprisal", "top_ids", "top_lp",
    }
