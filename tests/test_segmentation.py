from vitals.core import segmentation as S
from vitals.core.events import events_for_new_regions
from vitals.core.schema import EventKind, RegionKind


def _msgs():
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "Let me run the tests."},
        {"role": "tool", "content": "Traceback (most recent call last):\n  ...\nAssertionError"},
        {"role": "assistant", "content": "The test fails because..."},
    ]


def test_roles_map_to_kinds_and_ids():
    regions = S.segment_messages(_msgs())
    kinds = [r.kind for r in regions]
    assert kinds == [
        RegionKind.SYSTEM,
        RegionKind.USER,
        RegionKind.REASONING,
        RegionKind.OBSERVATION,
        RegionKind.REASONING,
    ]
    assert regions[0].id == "sys_0"
    assert regions[3].id == "obs_0"
    assert regions[4].id == "gen_1"


def test_turn_counting_and_char_spans():
    regions = S.segment_messages(_msgs())
    assert regions[3].turn == 1  # tool result arrives after the first assistant turn
    assert regions[0].char_start == 0
    assert all(r.char_end >= r.char_start for r in regions)


def test_error_observation_detected_and_evented():
    regions = S.segment_messages(_msgs())
    obs = regions[3]
    assert obs.is_error is True
    ev = events_for_new_regions([obs], turn=1)
    assert [e.kind for e in ev] == [EventKind.OBSERVATION_INJECTED, EventKind.ERROR_BLOCK]


def test_explicit_is_error_overrides_heuristic():
    msgs = [{"role": "tool", "content": "All 12 tests passed.", "is_error": True}]
    assert S.segment_messages(msgs)[0].is_error is True
    msgs = [{"role": "tool", "content": "Error: nope", "is_error": False}]
    assert S.segment_messages(msgs)[0].is_error is False


def test_tools_param_yields_schema_region():
    regions = S.segment_messages(_msgs(), tools=[{"name": "bash"}])
    assert regions[0].kind == RegionKind.TOOL_SCHEMA


def test_extends_streaming_last_message():
    prev = _msgs()
    prev[-1] = {"role": "assistant", "content": "The test fails because"}
    new = _msgs()
    new[-1] = {"role": "assistant", "content": "The test fails because the fixture is stale."}
    assert S.extends(prev, new) is True
    assert S.extends(new, prev) is False


def test_compaction_detected_on_summarized_history():
    prev = _msgs() + [
        {"role": "tool", "content": "long output " * 50},
        {"role": "assistant", "content": "done step"},
    ]
    # realistic compaction: system head + summary carrying prior content + the
    # last turns carried over (compactions are >= 3 messages; 2-message openers
    # are always new episodes)
    new = [
        prev[0],
        {"role": "user", "content": "Summary of progress: Fix the failing test. AssertionError seen."},
        {"role": "assistant", "content": "done step"},
    ]
    assert S.detect_compaction(prev, new) is True


def test_marker_tagged_user_messages_become_observations():
    """mini-SWE-agent-style flat history: tool output arrives as user role with
    <returncode>/<output> markup; the JSON template variant likewise."""
    ok = {"role": "user", "content": "<returncode>0</returncode>\n<output>\nall tests pass\n</output>"}
    bad = {"role": "user", "content": "<returncode>2</returncode>\n<output>\nboom\n</output>"}
    exc = {"role": "user", "content": "<exception>TimeoutError</exception>\n<returncode>1</returncode>"}
    js = {"role": "user", "content": '{\n  "returncode": 1,\n  "output": "nope"\n}'}
    human = {"role": "user", "content": "Please fix the failing test."}
    regions = S.segment_messages([ok, bad, exc, js, human])
    kinds = [r.kind for r in regions]
    assert kinds[:4] == [RegionKind.OBSERVATION] * 4 and kinds[4] == RegionKind.USER
    assert [r.is_error for r in regions[:4]] == [False, True, True, True]


def test_tool_role_error_phrasings_from_real_harnesses():
    """pi: 'Command exited with code N'; Hermes: 'Script exited with code N';
    Python exception class names (SyntaxError:) — all must flag; exit code 0
    and prose 'errors' must not."""
    err_texts = [
        "ls: cannot access\nCommand exited with code 2",
        "Script exited with code 1",
        "quick command failed with exit code 5",
        'File "buggy.py", line 1\n    def main(:\nSyntaxError: invalid syntax',
    ]
    ok_texts = ["process exited with code 0", "0 errors found", "all 12 tests passed"]
    for t in err_texts:
        assert S.segment_messages([{"role": "tool", "content": t}])[0].is_error, t
    for t in ok_texts:
        assert not S.segment_messages([{"role": "tool", "content": t}])[0].is_error, t


def test_hermes_json_tool_result_exit_code():
    """Hermes wraps tool output as JSON with an exit_code field."""
    err = {"role": "tool", "content": '{"output": "pytest: command not found", "exit_code": 1, "error": null}'}
    ok = {"role": "tool", "content": '{"output": "total 12\\ndrwx", "exit_code": 0, "error": null}'}
    written = {"role": "tool", "content": '{"bytes_written": 99, "dirs_created": true}'}  # no exit_code
    assert S.segment_messages([err])[0].is_error is True
    assert S.segment_messages([ok])[0].is_error is False
    r = S.segment_messages([written])[0]
    assert r.kind == RegionKind.OBSERVATION and r.is_error is False


def test_fresh_two_message_opener_is_never_compaction():
    """Same system prompt + similarly-worded new task must start a NEW
    episode, not read as compaction of the old one."""
    prev = _msgs() + [{"role": "user", "content": "<returncode>0</returncode>\n<output>\nok\n</output>"}]
    new = [prev[0], {"role": "user", "content": "Fix the failing test in the directory src/t02."}]
    assert S.detect_compaction(prev, new) is False


def test_marker_error_beats_generic_heuristic():
    # returncode 0 but output mentions 'Error:' — exit code wins
    msg = {"role": "user", "content": "<returncode>0</returncode>\n<output>\ngrep: Error: pattern\n</output>"}
    assert S.segment_messages([msg])[0].is_error is False


def test_unrelated_short_history_is_not_compaction():
    prev = _msgs()
    new = [
        {"role": "system", "content": "You are a poet."},
        {"role": "user", "content": "Write a haiku about frogs."},
    ]
    assert S.detect_compaction(prev, new) is False
