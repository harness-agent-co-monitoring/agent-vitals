"""Atomic writes, tolerant JSON lines, cache locations, and episode ids that cannot leave the store."""

import json

import pytest

from vitals.core.io import atomic_savez, atomic_write_text, read_jsonl, write_jsonl
from vitals.core.paths import cache_dir, cache_root, store_cache, store_key
from vitals.core.store import TraceStore, check_episode_id


def test_atomic_writes_leave_no_temporaries(tmp_path):
    p = tmp_path / "a.txt"
    atomic_write_text(p, "one")
    atomic_write_text(p, "two")
    assert p.read_text() == "two"
    atomic_savez(tmp_path / "d.npz", x=[1, 2])
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.txt", "d.npz"]


def test_read_jsonl_skips_a_partial_last_line(tmp_path):
    p = tmp_path / "rows.jsonl"
    write_jsonl(p, [{"a": 1}, {"a": 2}])
    with p.open("a") as f:
        f.write('{"a": 3')                       # a writer interrupted mid-line
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_cache_locations(tmp_path, monkeypatch):
    for v in ("VITALS_CACHE_DIR", "XDG_CACHE_HOME", "VITALS_ALARM_DIR", "VITALS_LEDGER_DIR"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "xdg" / "vitals"
    monkeypatch.setenv("VITALS_CACHE_DIR", str(tmp_path / "root"))
    assert cache_dir("ledger") == tmp_path / "root" / "ledger"
    monkeypatch.setenv("VITALS_ALARM_DIR", str(tmp_path / "alarms"))
    assert cache_dir("alarms") == tmp_path / "alarms"
    assert cache_dir("alarms", tmp_path / "given") == tmp_path / "given"
    assert store_cache("ledger", tmp_path) == tmp_path / "root" / "ledger" / store_key(tmp_path)
    with pytest.raises(ValueError):
        cache_dir("nope")


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "/abs/path", "a\\b", "x\x00y"])
def test_episode_ids_stay_inside_the_store(tmp_path, bad):
    with pytest.raises(ValueError):
        check_episode_id(bad)
    store = TraceStore(tmp_path / "store")
    with pytest.raises(ValueError):
        store.write_messages(bad, 0, [{"role": "user", "content": "hi"}])
    assert not (tmp_path / "x").exists()


def test_store_readers_create_nothing(tmp_path):
    store = TraceStore(tmp_path / "store")
    assert store.read_tokens("ep-missing").num_rows == 0
    assert store.read_regions("ep-missing") == [] and store.read_messages("ep-missing") == []
    assert not (tmp_path / "store" / "ep-missing").exists()
    store.write_messages("session 7:a", 0, [{"role": "user", "content": "hi"}])     # odd but safe ids still work
    assert json.loads((tmp_path / "store" / "session 7:a" / "messages.jsonl").read_text())["turn"] == 0
