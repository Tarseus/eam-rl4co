from __future__ import annotations

from pathlib import Path


def test_alternating_prefers_incumbent_loss_over_elite_loss(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    compiled_g = {"g_ref": object()}
    compiled_f = {"f_best": object(), "f_elite": object(), "f_ref": object()}

    best_builder_cost = None
    best_so_far = {"builder_id": "g_ref", "loss_id": "f_best"}
    resident_pop_g = [{"id": "g_ref"}]
    resident_pop_f = [{"id": "f_elite"}]

    fixed_builder_id = None
    fixed_loss_id = None

    # Mirror the resolution logic in run_pref_loss_coevo()'s alternating branch.
    if (not fixed_builder_id) and isinstance(best_builder_cost, dict):
        cand_g = str(best_builder_cost.get("builder_id") or "")
        if cand_g and cand_g in compiled_g:
            fixed_builder_id = cand_g
    if (not fixed_builder_id) and resident_pop_g:
        cand_g = str(resident_pop_g[0].get("id") or "")
        if cand_g and cand_g in compiled_g:
            fixed_builder_id = cand_g
    if (not fixed_builder_id) and isinstance(best_so_far, dict):
        cand_g = str(best_so_far.get("builder_id") or "")
        if cand_g and cand_g in compiled_g:
            fixed_builder_id = cand_g
    if not fixed_builder_id and loop.G_REF_ID in compiled_g:
        fixed_builder_id = str(loop.G_REF_ID)

    if (not fixed_loss_id) and isinstance(best_so_far, dict):
        cand_f = str(best_so_far.get("loss_id") or "")
        if cand_f and cand_f in compiled_f:
            fixed_loss_id = cand_f
    if (not fixed_loss_id) and resident_pop_f:
        cand_f = str(resident_pop_f[0].get("id") or "")
        if cand_f and cand_f in compiled_f:
            fixed_loss_id = cand_f
    if not fixed_loss_id and loop.F_REF_ID in compiled_f:
        fixed_loss_id = str(loop.F_REF_ID)

    assert fixed_builder_id == "g_ref"
    assert fixed_loss_id == "f_best"


def test_alternating_prefers_best_builder_cost_for_loss_phase(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    compiled_g = {"g_best_cost": object(), "g_elite": object(), "g_best_perf": object()}
    compiled_f = {"f_best": object(), "f_ref": object()}

    best_builder_cost = {"builder_id": "g_best_cost", "cost": 32}
    best_so_far = {"builder_id": "g_best_perf", "loss_id": "f_best"}
    resident_pop_g = [{"id": "g_elite"}]

    fixed_builder_id = None

    if (not fixed_builder_id) and isinstance(best_builder_cost, dict):
        cand_g = str(best_builder_cost.get("builder_id") or "")
        if cand_g and cand_g in compiled_g:
            fixed_builder_id = cand_g
    if (not fixed_builder_id) and resident_pop_g:
        cand_g = str(resident_pop_g[0].get("id") or "")
        if cand_g and cand_g in compiled_g:
            fixed_builder_id = cand_g
    if (not fixed_builder_id) and isinstance(best_so_far, dict):
        cand_g = str(best_so_far.get("builder_id") or "")
        if cand_g and cand_g in compiled_g:
            fixed_builder_id = cand_g
    if not fixed_builder_id and loop.G_REF_ID in compiled_g:
        fixed_builder_id = str(loop.G_REF_ID)

    assert fixed_builder_id == "g_best_cost"


def test_alternating_schedule_supports_final_loss_tail(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    phase0 = loop._resolve_alternating_phase_and_budgets(
        search_mode="alternating",
        generation=0,
        pairing_budget=8,
        pairing_budget_loss=4,
        pairing_budget_builder=4,
        alternating_schedule_enabled=True,
        alternating_loss_generations=2,
        alternating_builder_generations=3,
        alternating_rounds=1,
        alternating_final_loss_generations=2,
        alternating_start_phase="loss",
    )
    phase3 = loop._resolve_alternating_phase_and_budgets(
        search_mode="alternating",
        generation=3,
        pairing_budget=8,
        pairing_budget_loss=4,
        pairing_budget_builder=4,
        alternating_schedule_enabled=True,
        alternating_loss_generations=2,
        alternating_builder_generations=3,
        alternating_rounds=1,
        alternating_final_loss_generations=2,
        alternating_start_phase="loss",
    )
    phase5 = loop._resolve_alternating_phase_and_budgets(
        search_mode="alternating",
        generation=5,
        pairing_budget=8,
        pairing_budget_loss=4,
        pairing_budget_builder=4,
        alternating_schedule_enabled=True,
        alternating_loss_generations=2,
        alternating_builder_generations=3,
        alternating_rounds=1,
        alternating_final_loss_generations=2,
        alternating_start_phase="loss",
    )
    phase7 = loop._resolve_alternating_phase_and_budgets(
        search_mode="alternating",
        generation=7,
        pairing_budget=8,
        pairing_budget_loss=4,
        pairing_budget_builder=4,
        alternating_schedule_enabled=True,
        alternating_loss_generations=2,
        alternating_builder_generations=3,
        alternating_rounds=1,
        alternating_final_loss_generations=2,
        alternating_start_phase="loss",
    )

    assert phase0[0] == "loss"
    assert phase3[0] == "builder"
    assert phase5[0] == "loss"
    assert phase5[1] == 8 and phase5[2] == 0
    assert phase7[0] == "none"


def test_alternating_schedule_supports_builder_first(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    phase0 = loop._resolve_alternating_phase_and_budgets(
        search_mode="alternating",
        generation=0,
        pairing_budget=8,
        pairing_budget_loss=4,
        pairing_budget_builder=4,
        alternating_schedule_enabled=True,
        alternating_loss_generations=2,
        alternating_builder_generations=3,
        alternating_rounds=1,
        alternating_final_loss_generations=2,
        alternating_start_phase="builder",
    )
    phase2 = loop._resolve_alternating_phase_and_budgets(
        search_mode="alternating",
        generation=2,
        pairing_budget=8,
        pairing_budget_loss=4,
        pairing_budget_builder=4,
        alternating_schedule_enabled=True,
        alternating_loss_generations=2,
        alternating_builder_generations=3,
        alternating_rounds=1,
        alternating_final_loss_generations=2,
        alternating_start_phase="builder",
    )
    phase5 = loop._resolve_alternating_phase_and_budgets(
        search_mode="alternating",
        generation=5,
        pairing_budget=8,
        pairing_budget_loss=4,
        pairing_budget_builder=4,
        alternating_schedule_enabled=True,
        alternating_loss_generations=2,
        alternating_builder_generations=3,
        alternating_rounds=1,
        alternating_final_loss_generations=2,
        alternating_start_phase="builder",
    )

    assert phase0[0] == "builder"
    assert phase2[0] == "builder"
    assert phase5[0] == "loss"
    assert phase5[1] == 8 and phase5[2] == 0


def test_loss_only_search_mode_uses_full_budget_for_loss(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    phase = loop._resolve_alternating_phase_and_budgets(
        search_mode="loss_only",
        generation=7,
        pairing_budget=8,
        pairing_budget_loss=4,
        pairing_budget_builder=4,
        alternating_schedule_enabled=True,
        alternating_loss_generations=2,
        alternating_builder_generations=3,
        alternating_rounds=1,
        alternating_final_loss_generations=2,
        alternating_start_phase="builder",
    )

    assert phase == ("loss", 8, 0, -1, -1, 0)


def test_resolve_runtime_config_accepts_loss_only_search_mode(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg, _meta = loop._resolve_runtime_config(
        {
            "preset": "simple",
            "search_mode": "loss_only",
            "budgets": {"generations": 3, "pairing_budget_per_gen": 8},
            "population": {"n_candidates_loss": 4, "n_candidates_builder": 4, "keep_top_k": 2},
        }
    )

    assert cfg["search_mode"] == "loss_only"
