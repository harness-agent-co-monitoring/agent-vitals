"""vitals.analysis.phases on a synthetic single-turn episode: think/answer
split at </think>, per-phase lift + consult, and the answer flow (direct vs
through the reasoning) from the generated-side joint."""

import json
import random

import pandas as pd
import pytest

from vitals.analysis import phases

THINK, END = 248068, 248069


@pytest.fixture()
def ep(tmp_path):
    # prompt: 10 tokens; span A = [0,4) (4 tok), span B = [4,10) (6 tok); cue after
    L = 10
    regs = [{"id": "user_0", "turn": 0, "token_start": 0, "token_end": L},
            {"id": "span:A", "turn": 0, "token_start": 0, "token_end": 4},
            {"id": "span:B", "turn": 0, "token_start": 4, "token_end": 10}]
    (tmp_path / "regions.jsonl").write_text(json.dumps({"turn": 0, "regions": regs}) + "\n")
    (tmp_path / "attn_received.jsonl").write_text(json.dumps({"turn": 0, "vec": [0.1] * L, "replayed": True}) + "\n")
    pd.DataFrame({"turn": [0] * L, "idx": range(L), "token_id": range(L), "token_str": [f"t{i}" for i in range(L)],
                  "region_id": ["user_0"] * L, "attn": [0.1] * L}).to_parquet(tmp_path / "attn_tokens.parquet")
    # 4 think steps (last is </think>) then 2 answer steps.
    # attn_share per step = [user_0, A, B] shares; think reads A (0.4) not B (0.1); answer reads B (0.3), A (0.1)
    steps = list(range(6))
    ids = [7, 7, 7, END, 9, 9]
    share = [[0.5, 0.4, 0.1]] * 4 + [[0.4, 0.1, 0.3]] * 2
    ent = [1.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    pd.DataFrame({"episode_id": ["e"] * 6, "turn": [0] * 6, "step": steps, "token_id": ids, "ent": ent,
                  "fork": [False] * 6, "region_id": ["gen"] * 6, "attn_share": share}).to_parquet(tmp_path / "tokens-00000.parquet")
    # sparse joint: answer step 4 reads reasoning steps 0,1 (weights 0.3, 0.2);
    # answer step 5 reads reasoning step 2 (0.4) and answer step 4 (0.1)
    rows = [dict(turn=0, step=s, pos=[0], w=[0.05], prompt_mass=0.5, gen_mass=0.5, gpos=[], gw=[]) for s in range(4)]
    rows += [dict(turn=0, step=4, pos=[0], w=[0.05], prompt_mass=0.3, gen_mass=0.7, gpos=[0, 1], gw=[0.3, 0.2]),
             dict(turn=0, step=5, pos=[0], w=[0.05], prompt_mass=0.3, gen_mass=0.7, gpos=[2, 4], gw=[0.4, 0.1])]
    pd.DataFrame(rows).to_parquet(tmp_path / "attn_sparse.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({"meta": {"n_reasoning_toks": 4}}))
    return tmp_path


def test_think_split(ep):
    assert phases.think_end_id(ep) == phases.THINK_END_FALLBACK   # no <think> in this cue-less prompt
    g = pd.read_parquet(ep / "tokens-00000.parquet")
    assert phases.think_steps(g, END) == 4
    assert phases.think_steps(g[g.step >= 4], END) == 0


def test_section_phase_rows(ep):
    rows = {r["section"]: r for r in phases.section_phase_rows(ep)}
    a, b = rows["A"], rows["B"]
    assert a["n_think"] == 4 and a["n_answer"] == 2
    # lift = share / (size/L): A fair share 0.4, B 0.6
    assert a["lift_think"] == pytest.approx(0.4 / 0.4) and a["lift_answer"] == pytest.approx(0.1 / 0.4)
    assert b["lift_think"] == pytest.approx(0.1 / 0.6) and b["lift_answer"] == pytest.approx(0.3 / 0.6)
    # consult needs >= 8 tokens: n/a here
    assert a["consult_think"] is None and a["consult_answer"] is None


def test_answer_flow(ep):
    f = phases.answer_flow(ep)
    assert f["available"]
    A, B = f["sections"]["A"], f["sections"]["B"]
    assert A["direct"] == pytest.approx(0.1 / 0.4) and B["direct"] == pytest.approx(0.3 / 0.6)
    # one hop: step4 -> 0.3*share_0(A)+0.2*share_1(A) = 0.20; step5 -> 0.4*share_2(A) + 0.1*share_4(A) = 0.17
    assert A["indirect"] == pytest.approx(((0.20 + 0.17) / 2) / 0.4)
    # full recursion, captured only: reasoning steps have no generated edges here, so F_r = share_r and the
    # answer's hop through step 4 adds w(5->4) * F_4(A) = 0.1 * (0.1 + 0.20)
    f4 = 0.1 + 0.20
    f5 = 0.1 + 0.4 * 0.4 + 0.1 * f4
    assert A["reach_captured"] == pytest.approx(((f4 + f5) / 2) / 0.4)
    # renormalized: captured gen weights scale to gen_mass 0.7 (step4: 0.5 -> x1.4; step5: 0.5 -> x1.4)
    f4r = 0.1 + 1.4 * 0.20
    f5r = 0.1 + 1.4 * (0.4 * 0.4 + 0.1 * f4r)
    assert A["reach"] == pytest.approx(((f4r + f5r) / 2) / 0.4)
    assert A["total"] == A["reach"]
    assert f["answer_row"]["gen"] == pytest.approx(0.7) and f["answer_row"]["reasoning"] == pytest.approx((0.5 + 0.4) / 2)
    assert f["answer_row"]["answer"] == pytest.approx(0.05)
    assert f["answer_row"]["captured"] == pytest.approx(0.5 / 0.7)


def test_answer_flow_mass_conservation(tmp_path):
    """With complete rows every step's traced prompt share is 1, at any depth."""
    L = 4
    regs = [{"id": "user_0", "turn": 0, "token_start": 0, "token_end": L},
            {"id": "span:A", "turn": 0, "token_start": 0, "token_end": 2}]
    (tmp_path / "regions.jsonl").write_text(json.dumps({"turn": 0, "regions": regs}) + "\n")
    (tmp_path / "attn_received.jsonl").write_text(json.dumps({"turn": 0, "vec": [0.25] * L, "replayed": True}) + "\n")
    pd.DataFrame({"turn": [0] * L, "idx": range(L), "token_id": range(L), "token_str": ["x"] * L,
                  "region_id": ["user_0"] * L, "attn": [0.25] * L}).to_parquet(tmp_path / "attn_tokens.parquet")
    # step0: all prompt (A gets 0.6); step1: 0.5 prompt (A 0.2) + 0.5 on step0; step2 = </think>: 0.2 prompt (A 0.1) + 0.8 on step1;
    # step3 (answer): 0.1 prompt (A 0.05) + 0.9 on step2
    share = [[1.0, 0.6], [0.5, 0.2], [0.2, 0.1], [0.1, 0.05]]
    ids = [7, 7, END, 9]
    pd.DataFrame({"episode_id": ["e"] * 4, "turn": [0] * 4, "step": range(4), "token_id": ids, "ent": [1.0] * 4,
                  "fork": [False] * 4, "region_id": ["gen"] * 4, "attn_share": share}).to_parquet(tmp_path / "tokens-00000.parquet")
    rows = [dict(turn=0, step=0, pos=[0], w=[0.5], prompt_mass=1.0, gen_mass=0.0, gpos=[], gw=[]),
            dict(turn=0, step=1, pos=[0], w=[0.3], prompt_mass=0.5, gen_mass=0.5, gpos=[0], gw=[0.5]),
            dict(turn=0, step=2, pos=[0], w=[0.1], prompt_mass=0.2, gen_mass=0.8, gpos=[1], gw=[0.8]),
            dict(turn=0, step=3, pos=[0], w=[0.05], prompt_mass=0.1, gen_mass=0.9, gpos=[2], gw=[0.9])]
    pd.DataFrame(rows).to_parquet(tmp_path / "attn_sparse.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({"meta": {}}))
    f = phases.answer_flow(tmp_path)
    assert f["answer_row"]["traced"] == pytest.approx(1.0)
    # A through three hops: F0=0.6; F1=0.2+0.5*0.6=0.5; F2=0.1+0.8*0.5=0.5; F3=0.05+0.9*0.5=0.5 -> lift 0.5/0.5
    assert f["sections"]["A"]["reach"] == pytest.approx(0.5 / 0.5) and f["sections"]["A"]["reach_captured"] == pytest.approx(1.0)
    assert f["sections"]["A"]["direct"] == pytest.approx(0.05 / 0.5)


def test_answer_flow_without_joint(ep):
    sp = pd.read_parquet(ep / "attn_sparse.parquet").drop(columns=["gpos", "gw"])
    sp.to_parquet(ep / "attn_sparse.parquet")
    f = phases.answer_flow(ep)
    assert not f["available"] and f["sections"]["A"]["indirect"] is None and f["sections"]["A"]["direct"] == pytest.approx(0.25)


def test_traced_consult_ratio_identity(tmp_path):
    """Consult ratio on the resolved reading = reach(entropy-weighted) / reach(uniform)."""
    L = 4
    regs = [{"id": "user_0", "turn": 0, "token_start": 0, "token_end": L},
            {"id": "span:A", "turn": 0, "token_start": 0, "token_end": 2}]
    (tmp_path / "regions.jsonl").write_text(json.dumps({"turn": 0, "regions": regs}) + "\n")
    (tmp_path / "attn_received.jsonl").write_text(json.dumps({"turn": 0, "vec": [0.25] * L, "replayed": True}) + "\n")
    pd.DataFrame({"turn": [0] * L, "idx": range(L), "token_id": range(L), "token_str": ["x"] * L,
                  "region_id": ["user_0"] * L, "attn": [0.25] * L}).to_parquet(tmp_path / "attn_tokens.parquet")
    T = 12
    ids = [7] * 3 + [END] + [9] * (T - 4)
    rng = random.Random(0)
    share = [[0.5, rng.uniform(0.05, 0.4)] for _ in range(T)]
    ent = [rng.uniform(0.1, 2.0) for _ in range(T)]
    pd.DataFrame({"episode_id": ["e"] * T, "turn": [0] * T, "step": range(T), "token_id": ids, "ent": ent,
                  "fork": [False] * T, "region_id": ["gen"] * T, "attn_share": share}).to_parquet(tmp_path / "tokens-00000.parquet")
    rows = []
    for t in range(T):
        gp = list(range(max(0, t - 2), t))
        gw = [0.25] * len(gp)
        rows.append(dict(turn=0, step=t, pos=[0], w=[0.1], prompt_mass=0.5, gen_mass=0.5, gpos=gp, gw=gw))
    pd.DataFrame(rows).to_parquet(tmp_path / "attn_sparse.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({"meta": {}}))
    u = phases.answer_flow(tmp_path, min_tokens=2)["sections"]["A"]
    e = phases.answer_flow(tmp_path, answer_weights="entropy", min_tokens=2)["sections"]["A"]
    assert u["consult_answer_traced"] == pytest.approx(e["reach"] / u["reach"])
    assert u["consult_think_traced"] is not None


def test_reasoning_steps_qwen_and_harmony():
    assert phases.reasoning_steps(["Let", " me", "</think>", "\n\n", "A"]) == 3
    harmony = [" to", "=self", "<|message|>", "R", "1", "<|eom|>", "<|start|>", "assistant", " to", "=user",
               "<|message|>", "ANS", "<|eot|>"]
    assert phases.reasoning_steps(harmony) == 6                       # reasoning through its <|eom|>
    assert phases.reasoning_steps(harmony[6:]) == 0                   # a turn that starts with the answer
    call_first = [" to", "=ls", "<|message|>", "<atem:function_calls>", "<|eom|>"]
    assert phases.reasoning_steps(call_first) == 0                    # a tool call is not reasoning
    assert phases.reasoning_steps(harmony[:4]) == 4                   # stopped while still reasoning
    assert phases.reasoning_steps(["plain", " answer"]) == 0


def test_recorded_boundary_overrides_the_qwen_rule(ep):
    # a boundary recorded by replay wins over </think> detection
    (ep / "reasoning_steps.jsonl").write_text(json.dumps({"turn": 0, "n_reasoning": 2, "n_gen": 6}) + "\n")
    rows = {r["section"]: r for r in phases.section_phase_rows(ep)}
    assert rows["A"]["n_think"] == 2 and rows["A"]["n_answer"] == 4
    assert phases.read_reasoning_steps(ep) == {0: 2}
