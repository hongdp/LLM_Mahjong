"""Style metrics: shared aggregator, rollout payload facts, style vs anchors."""
from src.agents.dnn.style_stats import new_agg, add_game, summarize, style_vs_anchors
from src.agents.dnn.arch_zoo import ZOO


def test_aggregator_counts_per_seat():
    a = new_agg()
    add_game(a, "玩家2 荣和(放铳:玩家0) | 3番30符 | 点数: [..]", [False, True, False, False], [0, 0, 2, 0], 40)
    add_game(a, "流局 | 听牌: [1] | 点数: [..]", [False] * 4, [0] * 4, 72)
    s = summarize(a)
    assert s["agari_rate"] == 1 / 8 and s["houjuu_rate"] == 1 / 8
    assert s["riichi_rate"] == 1 / 8 and s["call_rate"] == 1 / 8 and s["draw_rate"] == 0.5
    assert s["win_turn"] == 10.0 and s["dealin_turn"] == 10.0
    # candidate-seat-only view
    b = new_agg()
    add_game(b, "玩家2 自摸 | 点数", [False, False, True, False], [0] * 4, 48, seats=[2])
    s2 = summarize(b)
    assert s2["agari_rate"] == 1.0 and s2["tsumo_share"] == 1.0 and s2["riichi_rate"] == 1.0


def test_rollout_payload_carries_style_facts():
    from src.agents.dnn.parallel_rollout import collect_parallel
    net = ZOO["cnn_m_r"][0]().eval()
    cfg = dict(temperature=1.0, gamma=0.995, shaping=False, seed=1, critic_feats="none", arch="cnn_m_r")
    eps, res = collect_parallel(net, 4, cfg, 2, [5, 6, 7, 8])
    sty = summarize(collect_parallel.last_style)
    assert sty["games"] == 4 and 0.0 <= sty["agari_rate"] <= 1.0


def test_style_vs_anchors_runs():
    net = ZOO["cnn_m_r"][0]().eval()
    opp = [ZOO["cnn_m_r"][0]().eval() for _ in range(2)]
    s = style_vs_anchors(net, opp, games=4, seed0=100)
    assert s["games"] == 4 and set(s) >= {"agari_rate", "houjuu_rate", "win_turn"}
