import numpy as np

from vitals import SCHEMA_VERSION
from vitals.core.schema import CaptureSpec, EngineInfo, EpisodeManifest, ModelInfo, TokenMetrics
from vitals.core.store import TraceStore


def _tm(step: int, ep="ep0") -> TokenMetrics:
    return TokenMetrics(
        episode_id=ep, turn=0, step=step, token_id=step + 100,
        ent=1.5, m2=0.2, js=0.05, fork=step % 5 == 0,
        probes={"commit:tool_call": 0.3} if step == 2 else {},
    )


def test_token_roundtrip(tmp_path):
    store = TraceStore(tmp_path, flush_every=4)
    for i in range(10):
        store.append(_tm(i))
    table = store.read_tokens("ep0")
    assert table.num_rows == 10
    assert table.column("step").to_pylist() == list(range(10))
    assert (table.schema.metadata or {})[b"schema_version"].decode() == SCHEMA_VERSION
    probes = table.column("probes").to_pylist()
    assert probes[2] == [("commit:tool_call", np.float32(0.3))]


def test_manifest_roundtrip_and_source_fields(tmp_path):
    store = TraceStore(tmp_path)
    m = EpisodeManifest(
        episode_id="ep1",
        engine=EngineInfo(kind="llamacpp", build="b9910 (f5525f7)", flags={"fa": "on", "ngl": 99}),
        model=ModelInfo(name="Qwen3.5-27B", quant="Q4_K_M", arch="qwen3_5", n_layers=64),
        sampler={"temp": 0.7, "seed": 42},
        capture=CaptureSpec(top_logprobs=20),
    )
    store.write_manifest(m)
    back = store.read_manifest("ep1")
    assert back.engine.build == "b9910 (f5525f7)"
    assert back.model.quant == "Q4_K_M"
    assert back.schema_version == SCHEMA_VERSION
    assert back.library_version  # auto-stamped


def test_events_roundtrip(tmp_path):
    from vitals.core.schema import EventKind, HarnessEvent

    store = TraceStore(tmp_path)
    evs = [
        HarnessEvent(kind=EventKind.OBSERVATION_INJECTED, turn=1, region_id="obs_0"),
        HarnessEvent(kind=EventKind.ERROR_BLOCK, turn=1, region_id="obs_0"),
    ]
    store.append_events("ep3", evs)
    store.append_events("ep3", [HarnessEvent(kind=EventKind.COMPACTION, turn=2)])
    back = store.read_events("ep3")
    assert [e.kind for e in back] == [EventKind.OBSERVATION_INJECTED, EventKind.ERROR_BLOCK, EventKind.COMPACTION]
    assert back[0].region_id == "obs_0"
    assert store.read_events("nope") == []


def test_activations_zarr_append(tmp_path):
    store = TraceStore(tmp_path)
    store.save_activations("ep2", layer=16, turn=0, arr=np.ones((4, 8), dtype=np.float32))
    store.save_activations("ep2", layer=16, turn=0, arr=np.zeros((2, 8), dtype=np.float32))
    import zarr

    grp = zarr.open_group(str(tmp_path / "ep2" / "activations.zarr"), mode="r")
    arr = grp["layer16/turn0"][:]
    assert arr.shape == (6, 8)
    assert arr.dtype == np.float16


def test_attn_received_roundtrip(tmp_path):
    store = TraceStore(tmp_path)
    store.append_attn_received("epA", 0, [0.1, 0.2, 0.7])
    store.append_attn_received("epA", 1, np.asarray([0.5, 0.5]))
    back = store.read_attn_received("epA")
    assert back[0] == [0.1, 0.2, 0.7]
    assert back[1] == [0.5, 0.5]
    assert store.read_attn_received("missing") == {}


def test_reopened_store_appends_new_token_parts(tmp_path):
    """A second TraceStore on the same directory continues the episode's part numbering instead of overwriting."""
    from vitals.core.schema import TokenMetrics
    from vitals.core.store import TraceStore

    for turn in (0, 1):
        st = TraceStore(tmp_path)
        st.append(TokenMetrics(episode_id="ep", turn=turn, step=0, token_id=7 + turn, ent=0.0, m2=0.0, js=0.0,
                               fork=False, region_id="gen"))
        st.flush("ep")
    t = TraceStore(tmp_path).read_tokens("ep")
    assert sorted(t.column("turn").to_pylist()) == [0, 1]
    assert len(list((tmp_path / "ep").glob("tokens-*.parquet"))) == 2
