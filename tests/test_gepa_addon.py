"""GEPA optimize_anything add-on.

Unit tests run without a model. The end-to-end test runs a minimal GEPA loop with a stub
reflection LM on the 0.8B GGUF and checks that the readings reach the reflection prompt;
it is skipped when the engine, the model or gepa is unavailable.
"""

from __future__ import annotations

import inspect
import json
import os
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import pytest

from vitals.addons.gepa import (
    FIELD,
    MAP_FIELD,
    STR_CANDIDATE_KEY,
    GepaVitals,
    aggregate,
    describe,
    message_fields,
)
from vitals.engine.template import ChatTemplate
from vitals.replay.record import forced_text, strip_end_of_turn

QWEN_LIKE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' %}<think>\n{{ m.reasoning_content or '' }}\n</think>\n\n{% endif %}"
    "{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}"
    "{% endif %}"
)


# -- parsing -----------------------------------------------------------------------------

def test_message_fields_from_dict_and_object():
    d = {"choices": [{"message": {"content": "hi", "reasoning": "because",
                                  "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {"x": 1}}}]}}]}
    content, reasoning, calls = message_fields(d)
    assert (content, reasoning) == ("hi", "because")
    assert calls == [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]
    obj = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="a", reasoning_content="r", tool_calls=None))])
    assert message_fields(obj) == ("a", "r", [])
    with pytest.raises(ValueError):
        message_fields({"choices": []})


def test_forced_text_is_the_template_continuation():
    tpl = ChatTemplate(QWEN_LIKE)
    msgs = [{"role": "user", "content": "Q?"}]
    out = forced_text(tpl, msgs, {"role": "assistant", "content": "A", "reasoning_content": "R"})
    assert out == "R\n</think>\n\nA<|im_end|>\n"
    # no reasoning under a template that opens a think block: the empty block is forced too
    assert forced_text(tpl, msgs, {"role": "assistant", "content": "A"}) == "\n</think>\n\nA<|im_end|>\n"
    # thinking disabled: the generation prompt pre-fills the empty block
    assert forced_text(tpl, msgs, {"role": "assistant", "content": "A"},
                       chat_kwargs={"enable_thinking": False}) == "A<|im_end|>\n"
    with pytest.raises(ValueError, match="enable_thinking"):
        forced_text(tpl, msgs, {"role": "assistant", "content": "A", "reasoning_content": "R"},
                    chat_kwargs={"enable_thinking": False})


def test_strip_end_of_turn():
    vocab = {1: "A", 2: "<|im_end|>", 3: "\n", 4: "B"}
    eng = SimpleNamespace(is_eog=lambda t: t == 2, detokenize=lambda ts: "".join(vocab[t] for t in ts))
    assert strip_end_of_turn(eng, [1, 4, 2, 3]) == [1, 4]
    assert strip_end_of_turn(eng, [1, 4]) == [1, 4]          # no end-of-turn token: unchanged
    assert strip_end_of_turn(eng, [1, 2, 4]) == [1, 2, 4]    # an inner end token is content


# -- aggregation and text ------------------------------------------------------------------

def test_aggregate_and_describe():
    rows = [
        {"turn": 0, "section": "s", "tokens": 40, "lift": 1.0, "peak": 3.0, "hot_line": "line one",
         "consult_ratio": 1.5, "consult_contrast": None},
        {"turn": 1, "section": "s", "tokens": 40, "lift": 3.0, "peak": 5.0, "hot_line": "line two",
         "consult_ratio": 1.0 / 1.5, "consult_contrast": 2.0},
    ]
    a = aggregate(rows)["s"]
    assert a["read_lift"] == pytest.approx(2.0)
    assert (a["peak_lift"], a["hot_line"]) == (5.0, "line two")
    assert a["consult_ratio"] == pytest.approx(1.0)
    assert a["consult_contrast"] == pytest.approx(2.0)
    text = describe(a)
    assert "read lift 2.00" in text and "peak lift 5.0 (hottest line: 'line two')" in text
    assert "2 model calls" in text


def test_aggregate_section_specific_consult():
    rows = [{"turn": 0, "section": "s", "tokens": 10, "lift": 1.0, "peak": None, "hot_line": None,
             "consult_ratio": 0.6, "consult_contrast": 0.5,
             "diagnostics": {"relative_consult_ratio": 1.2, "relative_consult_contrast": 1.5}}]
    assert aggregate(rows)["s"]["consult_ratio"] == pytest.approx(0.6)
    sec = aggregate(rows, consult="section")["s"]
    assert (sec["consult_ratio"], sec["consult_contrast"]) == (pytest.approx(1.2), pytest.approx(1.5))
    with pytest.raises(ValueError):
        aggregate(rows, consult="other")
    assert GepaVitals(None, ".").background != GepaVitals(None, ".", consult="raw").background


# -- the wrapper ---------------------------------------------------------------------------

def _fake(v: GepaVitals, measured: dict | Exception):
    calls = []

    def measure(components, recorded):
        calls.append((dict(components), list(recorded)))
        if isinstance(measured, Exception):
            raise measured
        return measured

    v.measure = measure
    return calls


MEASURED = {"declared": ["system", "rule", "question"], "missing": ["rule"], "match": 0.5,
            "sections": {"system": {"tokens": 20, "turns": 1, "read_lift": 0.4, "peak_lift": None,
                                    "hot_line": None, "consult_ratio": 1.2, "consult_contrast": None},
                         "question": {"tokens": 8, "turns": 1, "read_lift": 3.0, "peak_lift": None,
                                      "hot_line": None, "consult_ratio": 0.9, "consult_contrast": None}}}


def test_wrap_preserves_signature_and_annotates(tmp_path):
    v = GepaVitals(None, tmp_path)
    seen = _fake(v, MEASURED)

    def evaluate(candidate, example, opt_state=None):
        v.record([{"role": "user", "content": example}], content="ok", sections=[{"name": "question", "text": example}])
        return 1.0, {"Output": "ok", "system_specific_info": {"Analysis": "kept"}}

    w = v.wrap(evaluate)
    assert list(inspect.signature(w).parameters) == ["candidate", "example", "opt_state"]
    score, info = w({"system": "S", "rule": "R"}, "Q", opt_state="state")
    assert score == 1.0 and info["Output"] == "ok"
    assert info["system_specific_info"] == {"Analysis": "kept", FIELD: describe(MEASURED["sections"]["system"])}
    assert "not found verbatim" in info["rule_specific_info"][FIELD]
    assert "question: read lift 3.00" in info[MAP_FIELD] and "matches only 50%" in info[MAP_FIELD]
    assert "scores" not in info
    (components, recorded), = seen
    assert components == {"system": "S", "rule": "R"}
    assert recorded[0].sections == [{"name": "question", "text": "Q"}]


def test_wrap_handles_bare_scores_strings_and_failures(tmp_path):
    v = GepaVitals(None, tmp_path)
    seen = _fake(v, {"declared": [STR_CANDIDATE_KEY], "missing": [], "match": 0.95,
                     "sections": {STR_CANDIDATE_KEY: MEASURED["sections"]["system"]}})

    def evaluate(candidate):
        v.record([{"role": "user", "content": candidate}], content="x")
        return 0.25

    score, info = v.wrap(evaluate)("a plain prompt")
    assert score == 0.25 and FIELD in info[f"{STR_CANDIDATE_KEY}_specific_info"]
    assert MAP_FIELD not in info            # one section and good agreement: nothing to add
    assert seen[0][0] == {STR_CANDIDATE_KEY: "a plain prompt"}

    v2 = GepaVitals(None, tmp_path)
    _fake(v2, RuntimeError("boom"))
    score, info = v2.wrap(lambda c: (v2.record([{"role": "user", "content": "q"}], content="x"), 0.5)[1])("p")
    assert score == 0.5 and info[MAP_FIELD].startswith("unavailable (RuntimeError: boom)")


def test_no_record_no_measure_and_sampling(tmp_path):
    v = GepaVitals(None, tmp_path, sample=0.0)
    seen = _fake(v, MEASURED)
    w = v.wrap(lambda c: (v.record([{"role": "user", "content": "q"}], content="x"), 1.0)[1])
    assert w({"system": "S"}) == (1.0, {})
    v.sample = 1.0
    assert v.wrap(lambda c: 1.0)({"system": "S"}) == (1.0, {})   # nothing recorded: nothing measured
    assert not seen


def test_record_outside_wrap_raises(tmp_path):
    with pytest.raises(RuntimeError, match="outside"):
        GepaVitals(None, tmp_path).record([{"role": "user", "content": "q"}], content="x")


# -- end to end ----------------------------------------------------------------------------

VENDOR = Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin"
STR_COL = "token_str"
GGUFS = glob(os.environ.get("VITALS_TEST_GGUF") or os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/*/Qwen3.5-0.8B-BF16.gguf"))


@pytest.mark.skipif(not Path(os.environ.get("VITALS_LLAMA_LIB") or VENDOR / "libllama.so").exists() or not GGUFS,
                    reason="pinned engine build or 0.8B GGUF not available")
def test_gepa_reflection_prompt_carries_readings(tmp_path):
    gepa_oa = pytest.importorskip("gepa.optimize_anything")
    v = GepaVitals(GGUFS[0], tmp_path / "vitals", n_ctx=8192, chat_kwargs={"enable_thinking": True})
    prompts: list[str] = []

    def reflection_lm(prompt):
        prompts.append(prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False))
        return "```\nAnswer with the capital city only, as one JSON object.\n```"

    def evaluate(candidate, example):
        messages = [{"role": "system", "content": candidate["instructions"] + "\n\n" + candidate["format"]},
                    {"role": "user", "content": example["q"] + " \r\n"}]
        v.record(messages, content=json.dumps({"answer": example["a"]}),
                 reasoning="Recall the capital of the country named in the question.",
                 sections=[{"name": "question", "text": example["q"] + " \r\n"}])   # template trims the tail
        return 0.5, {"Output": example["a"]}

    data = [{"q": "What is the capital of France?", "a": "Paris"},
            {"q": "What is the capital of Japan?", "a": "Tokyo"}]
    seed = {"instructions": "You answer geography questions about capital cities. Think briefly first.",
            "format": 'Reply with exactly one JSON object of the form {"answer": "<city>"}.'}
    try:
        gepa_oa.optimize_anything(
            seed, evaluator=v.wrap(evaluate), dataset=data, valset=data, objective="Answer correctly.",
            background=v.background,
            config=gepa_oa.OptimizeAnythingConfig(
                engine="gepa", max_evals=8, output_dir=str(tmp_path / "oa"),
                engine_config={"reflection": {"reflection_lm": reflection_lm, "reflection_minibatch_size": 2},
                               "engine": {"parallel": False}}))
    finally:
        v.close()
    assert prompts, "GEPA never reflected"
    text = "\n".join(prompts)
    assert FIELD in text and "read lift" in text
    assert MAP_FIELD in text and "question: read lift" in text
    assert "Reading diagnostics (agent-vitals)" in text
    records = [json.loads(line) for line in (tmp_path / "vitals" / "evaluations.jsonl").read_text().splitlines()]
    assert records and all(not r["missing"] for r in records)
    assert {"instructions", "format", "question"} <= set(records[0]["sections"])
    # replay rendered with the recorded chat template options: the prompt ends inside an open
    # reasoning block (enable_thinking=True), not with the 0.8B template's default empty block
    import pandas as pd

    ep = tmp_path / "vitals" / "store" / records[0]["episode"]
    assert '"chat_kwargs": {"enable_thinking": true}' in (ep / "messages.jsonl").read_text()
    at = pd.read_parquet(ep / "attn_tokens.parquet").sort_values("idx")
    tail = "".join(at[STR_COL].tolist()[-4:])
    assert tail.endswith("<think>\n") and "</think>" not in tail
