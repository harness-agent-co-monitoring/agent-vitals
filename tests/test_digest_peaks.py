"""Top-K token averaging within a section: peak lift + hottest line."""

import json

import pandas as pd
import pytest

from vitals.analysis.readmap import PEAK_K, PEAK_MIN_TOK, section_peaks
from vitals.analysis.readmap import content_token as _content_token


def test_content_token_rule():
    assert _content_token("盗窃") and _content_token(" word") and _content_token("\ufffd\ufffd")
    assert not _content_token(" (") and not _content_token("x") and not _content_token("4") and not _content_token("\n\n")


def test_peaks(tmp_path):
    L = 100
    toks = [f"w{i} " for i in range(L)]
    attn = [0.001] * L
    # section A = [0, 50): flat 0.001 except 8 hot tokens on line 2 (tokens 20-27) at 0.02
    for i in range(20, 28):
        attn[i] = 0.02
    toks[9] = "w9\n"
    toks[19] = "w19\n"
    toks[29] = "w29\n"  # lines: 0-9, 10-19, 20-29, 30-49
    # section B = [50, 60): 10 tokens, under the floor
    pd.DataFrame({"turn": [0] * L, "idx": range(L), "token_id": range(L), "token_str": toks,
                  "region_id": ["user_0"] * L, "attn": attn}).to_parquet(tmp_path / "attn_tokens.parquet")
    regions = [{"id": "user_0", "turn": 0, "token_start": 0, "token_end": L},
               {"id": "span:A", "turn": 0, "token_start": 0, "token_end": 50},
               {"id": "span:B", "turn": 0, "token_start": 50, "token_end": 60}]
    # the hot line is anchored in the prompt text (token strings can carry replacement chars)
    prompt = "intro\n" + "".join(toks[:20]) + "".join(toks[20:29]) + "w29 tail 攻击\n" + "".join(toks[30:])
    (tmp_path / "messages.jsonl").write_text(json.dumps({"turn": 0, "messages": [{"role": "user", "content": prompt}]}) + "\n")
    pk = section_peaks(tmp_path, regions, L)
    peak, line = pk["A"]
    assert line.startswith("w20") and line.endswith("tail 攻击")
    (tmp_path / "messages.jsonl").unlink()
    pk = section_peaks(tmp_path, regions, L)
    peak, line = pk["A"]
    assert PEAK_K == 8 and PEAK_MIN_TOK == 32
    ctx = sum(attn)                               # lift is against uniform reading of the context
    assert peak == pytest.approx(0.02 * L / ctx)  # the 8 hot tokens, digest convention lift = attn * L / ctx
    assert line.startswith("w20") and "w27" in line and "w29" in line
    assert pk["B"] == (None, None)                # too short: lift is already the peak
    # structural tokens never enter the top-K: make the hot tokens punctuation and the peak moves
    toks2 = list(toks)
    for i in range(20, 28):
        toks2[i] = " ( "
    pd.DataFrame({"turn": [0] * L, "idx": range(L), "token_id": range(L), "token_str": toks2,
                  "region_id": ["user_0"] * L, "attn": attn}).to_parquet(tmp_path / "attn_tokens.parquet")
    peak2, _ = section_peaks(tmp_path, regions, L)["A"]
    assert peak2 == pytest.approx(0.001 * L / ctx)
    (tmp_path / "regions.jsonl").write_text(json.dumps({"turn": 0, "regions": regions}) + "\n")
