from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_seed_reserve_keeps_seed_builders_and_losses(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    # Patch LLM ops so propose() can run without network.
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint
    from ptp_discovery.free_loss_ir import FreeLossIR, FreeLossImplementationHint

    good_builder = PreferenceBuilderIR(
        name="llm_builder",
        intuition="build all objective-ordered pairs",
        implementation_hint=PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise"),
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, w_idx, l_idx = mask.nonzero(as_tuple=True)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, w_idx, l_idx), weight=None, meta={})\n"
        ),
    )

    def _b(*args, **kwargs):  # noqa: ANN001
        return good_builder, {"llm_op": "E1", "prompt_path": "x", "prompt_sha1": "x"}

    monkeypatch.setattr(loop.builder_llm_ops, "generate_pref_builder_candidate_with_meta", _b)
    monkeypatch.setattr(loop.builder_llm_ops, "crossover_pref_builder_with_meta", _b)
    monkeypatch.setattr(loop.builder_llm_ops, "e2_pref_builder_with_meta", _b)
    monkeypatch.setattr(loop.builder_llm_ops, "mutate_pref_builder_with_meta", _b)
    monkeypatch.setattr(loop.builder_llm_ops, "m2_tune_builder_with_meta", _b)

    good_loss = FreeLossIR(
        name="llm_loss",
        intuition="negative logsigmoid on alpha-scaled log-prob gap",
        pseudocode="loss = -logsigmoid(alpha*(lpw-lpl))",
        hyperparams={"scale": 1.0},
        operators_used=["logsigmoid"],
        implementation_hint=FreeLossImplementationHint(expects=["log_prob_w", "log_prob_l", "delta_z", "weight"], returns="scalar", mode="pairwise"),
        code=(
            "def generated_loss(batch, model_output, extra):\n"
            "    alpha = float(extra.get('alpha', 1.0))\n"
            "    x = alpha * (batch['log_prob_w'] - batch['log_prob_l'])\n"
            "    return (-ops.logsigmoid(x)).mean()\n"
        ),
        theoretical_basis="",
    )

    monkeypatch.setattr(loop.loss_llm_ops, "generate_free_loss_candidate", lambda *a, **k: good_loss)
    monkeypatch.setattr(loop.loss_llm_ops, "crossover_free_loss", lambda *a, **k: good_loss)
    monkeypatch.setattr(loop.loss_llm_ops, "e2_free_loss", lambda *a, **k: good_loss)
    monkeypatch.setattr(loop.loss_llm_ops, "mutate_free_loss", lambda *a, **k: good_loss)
    monkeypatch.setattr(loop.loss_llm_ops, "m2_tune_hparams", lambda *a, **k: good_loss)

    llm_cfg = {
        "builder": {"enabled": True, "seed_reserve": 2, "parent_p": 2, "num_E1": 100, "num_E2": 0, "num_M1": 0, "num_M2": 0, "repair": {"enabled": False}},
        "loss": {"enabled": True, "seed_reserve": 2, "parent_p": 2, "num_E1": 100, "num_E2": 0, "num_M1": 0, "num_M2": 0, "repair": {"enabled": False}},
        "prompts": {},
        "builder_gate": {"min_pairs": 1, "min_coverage": 1.0, "max_pairs_per_instance": 4096, "weight_nonneg": True, "semantic_tolerance": 0.0, "semantic_min_pass_rate": 1.0},
    }

    builders = loop._propose_builders_for_generation(
        generation=0,
        pop_g=6,
        elites_g=[],
        diverse_elites_g=[],
        rng=__import__("random").Random(0),
        llm_cfg=llm_cfg,
        operator_whitelist=[],
        global_feedback={},
    )
    assert len(builders) == 6
    seed_builders = [c for c in builders if str(c.get("op_type", "")).startswith("SEED")]
    assert len(seed_builders) >= 2

    losses = loop._propose_losses_for_generation(
        generation=0,
        pop_f=6,
        elites_f=[],
        diverse_elites_f=[],
        rng=__import__("random").Random(0),
        llm_cfg=llm_cfg,
        operator_whitelist=[],
        global_feedback={},
    )
    assert len(losses) == 6
    seed_losses = [c for c in losses if str(c.get("op_type", "")).startswith("SEED")]
    assert len(seed_losses) >= 2

