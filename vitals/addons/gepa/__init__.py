"""GEPA ``optimize_anything`` integration: per-component reading diagnostics as side information.

    from vitals.addons.gepa import GepaVitals

    vitals = GepaVitals("Qwen3.6-27B-BF16.gguf", "runs/gepa-vitals")

    def evaluate(candidate, example):
        messages = [{"role": "system", "content": candidate["system_prompt"]},
                    {"role": "user", "content": example["input"]}]
        response = client.chat.completions.create(model=MODEL, messages=messages)
        vitals.record(messages, response)
        return score(response, example), {"Output": response.choices[0].message.content}

    optimize_anything(seed, evaluator=vitals.wrap(evaluate), background=BACKGROUND + "\\n\\n" + vitals.background, ...)

See README.md in this directory.
"""

from vitals.addons.gepa.adapter import (
    BACKGROUND,
    FIELD,
    MAP_FIELD,
    STR_CANDIDATE_KEY,
    Call,
    GepaVitals,
    aggregate,
    describe,
    message_fields,
)

__all__ = ["BACKGROUND", "FIELD", "MAP_FIELD", "STR_CANDIDATE_KEY", "Call", "GepaVitals", "aggregate",
           "describe", "message_fields"]
