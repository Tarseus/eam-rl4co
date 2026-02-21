from __future__ import annotations

import random
from pathlib import Path


def test_build_coverage_plus_bandit_pairs_dedupes_pairs(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    rng = random.Random(0)
    pairs, _ = loop._build_coverage_plus_bandit_pairs(
        cfg_yaml={
            "coverage_top_e_elites": 1,
            "coverage_anneal_generations": 1,
            "coverage_k_elite_start": 3,
            "coverage_k_elite_end": 3,
            "coverage_k_hof_start": 0,
            "coverage_k_hof_end": 0,
            "coverage_k_random_start": 0,
            "coverage_k_random_end": 0,
            "bandit_prior_mu": 1.0,
            "bandit_c_start": 0.0,
            "bandit_c_end": 0.0,
        },
        gen=0,
        generations=1,
        pairing_budget=10,
        rng=rng,
        new_g_ids=["g0"],
        new_f_ids=[],
        elite_g_ids=["g0"],
        elite_f_ids=["f0"],
        hof_g_ids=[],
        hof_f_ids=[],
        g_id_pool=["g0"],
        f_id_pool=["f0", "f1"],
        caches=loop.PrefLossEvalCaches(),
        eval_sig="sig",
    )

    assert len(pairs) == len(set(pairs))
    assert set(pairs) == {("g0", "f0"), ("g0", "f1")}
