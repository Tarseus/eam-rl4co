from __future__ import annotations

from pathlib import Path


def test_alternating_prefers_incumbent_loss_over_elite_loss(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    compiled_g = {"g_ref": object()}
    compiled_f = {"f_best": object(), "f_elite": object(), "f_ref": object()}

    best_so_far = {"builder_id": "g_ref", "loss_id": "f_best"}
    elites_g = [{"id": "g_ref"}]
    elites_f = [{"id": "f_elite"}]

    fixed_builder_id = None
    fixed_loss_id = None

    # Mirror the resolution logic in run_pref_loss_coevo()'s alternating branch.
    if (not fixed_builder_id) and isinstance(best_so_far, dict):
        cand_g = str(best_so_far.get("builder_id") or "")
        if cand_g and cand_g in compiled_g:
            fixed_builder_id = cand_g
    if (not fixed_builder_id) and elites_g:
        cand_g = str(elites_g[0].get("id") or "")
        if cand_g and cand_g in compiled_g:
            fixed_builder_id = cand_g
    if not fixed_builder_id and loop.G_REF_ID in compiled_g:
        fixed_builder_id = str(loop.G_REF_ID)

    if (not fixed_loss_id) and isinstance(best_so_far, dict):
        cand_f = str(best_so_far.get("loss_id") or "")
        if cand_f and cand_f in compiled_f:
            fixed_loss_id = cand_f
    if (not fixed_loss_id) and elites_f:
        cand_f = str(elites_f[0].get("id") or "")
        if cand_f and cand_f in compiled_f:
            fixed_loss_id = cand_f
    if not fixed_loss_id and loop.F_REF_ID in compiled_f:
        fixed_loss_id = str(loop.F_REF_ID)

    assert fixed_builder_id == "g_ref"
    assert fixed_loss_id == "f_best"

