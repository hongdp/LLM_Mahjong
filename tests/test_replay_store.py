"""Round-trip test for the exp60 game store: shards must decode bit-identically
(planes take only k/20 values, so the uint8 x20 encoding is lossless)."""
import numpy as np

from src.agents.dnn.replay_store import StoreReader, read_shard, write_shard


def _episode(rng, T, C=56, S=20, A=46, seed=1):
    planes = rng.integers(0, 21, size=(T, C, 34)).astype(np.float32) / 20
    planes[:, :50] = (planes[:, :50] > 0.5).astype(np.float32)   # mostly binary, like v3r
    mask = rng.random((T, A)) < 0.3
    mask[:, 0] = True
    return {"planes": planes.astype(np.float16), "scalars": rng.standard_normal((T, S)).astype(np.float32),
            "mask": mask, "actions": rng.integers(0, A, size=T).astype(np.int64),
            "rewards": rng.standard_normal(T).astype(np.float32) * 3000,
            "returns": rng.standard_normal(T).astype(np.float32) * 3000, "key": (seed, 2)}


def test_shard_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    eps = [_episode(rng, T, seed=100 + i) for i, T in enumerate((7, 30, 1))]
    tags = ["gen_0003", "bc49", "exp46I"]
    path = write_shard(str(tmp_path), eps, tags, {"games": 1})
    back, meta = read_shard(path)
    assert meta["n_episodes"] == 3 and meta["n_steps"] == 38
    for e, b, t in zip(eps, back, tags):
        assert b["tag"] == t and b["key"][0] == e["key"][0]
        # fp16 stores 0.05 as 0.0499878; the uint8 grid is exact, so compare on the grid
        np.testing.assert_array_equal(np.rint(b["planes"] * 20), np.rint(e["planes"].astype(np.float32) * 20))
        np.testing.assert_allclose(b["planes"], e["planes"].astype(np.float32), atol=1e-3)
        np.testing.assert_array_equal(b["mask"], e["mask"])
        np.testing.assert_array_equal(b["actions"], e["actions"])
        np.testing.assert_array_equal(b["rewards"], e["rewards"])
        np.testing.assert_array_equal(b["returns"], e["returns"])
        np.testing.assert_allclose(b["scalars"], e["scalars"], rtol=2e-3, atol=2e-3)  # fp16
    # a mutated shard must NOT pass (anti-idle guard for the test itself)
    back[0]["actions"][0] += 1
    assert not np.array_equal(back[0]["actions"], eps[0]["actions"])


def test_reader_sees_only_complete_shards(tmp_path):
    rng = np.random.default_rng(1)
    r = StoreReader(str(tmp_path))
    assert r.new_shards() == []
    write_shard(str(tmp_path), [_episode(rng, 5)], ["bc49"], {"games": 1}, name="shard_a")
    assert [p.split("/")[-1] for p in r.new_shards()] == ["shard_a.npz"]
    assert r.new_shards() == []                     # incremental
    (tmp_path / "shard_b.npz").write_bytes(b"partial")   # no .json yet
    assert r.new_shards() == []
