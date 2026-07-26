from __future__ import annotations

import json

from fitness.ptp_high_fidelity import HighFidelityConfig
from ptp_discovery.analyze_matched_branch_probe import analyze
from ptp_discovery.matched_branch_probe import (
    MatchedBranchProbe,
    _collect_policy_state,
    save_matched_branch_probes,
)


def _cpu_tsp_config() -> HighFidelityConfig:
    return HighFidelityConfig(
        problem="tsp",
        backend="rl4co",
        env_name="tsp",
        generator_params={"num_loc": 8},
        policy_name="pomo",
        policy_kwargs={
            "po4cops_compat": True,
            "embed_dim": 32,
            "num_encoder_layers": 2,
            "decoder_layer_num": 1,
            "qkv_dim": 8,
            "num_heads": 4,
            "feedforward_hidden": 64,
            "tanh_clipping": 10,
            "eval_type": "argmax",
        },
        rollout_strategy="auto",
        objective_sign="neg_reward",
        train_problem_size=8,
        valid_problem_sizes=(8,),
        train_batch_size=2,
        pomo_size=8,
        device="cpu",
        seed=1234,
    )


def test_collect_policy_state_builds_same_prefix_probe():
    probes = _collect_policy_state(
        _cpu_tsp_config(),
        policy_state="early",
        checkpoint_path=None,
        instances=2,
        branch_count=3,
        depths=(2,),
        device="cpu",
    )
    assert len(probes) == 1
    probe = probes[0]
    probe.validate()
    assert probe.local_logp.shape == (2, 3)
    assert probe.terminal_logp.shape == (2, 3)
    assert probe.objective.shape == (2, 3)


def test_save_matched_branch_probes_requires_both_policy_states(tmp_path):
    early = _collect_policy_state(
        _cpu_tsp_config(),
        policy_state="early",
        checkpoint_path=None,
        instances=2,
        branch_count=3,
        depths=(2,),
        device="cpu",
    )[0]
    late = MatchedBranchProbe(
        policy_state="late",
        depth=early.depth,
        local_logp=early.local_logp,
        terminal_logp=early.terminal_logp,
        objective=early.objective,
    )
    destination = save_matched_branch_probes(
        tmp_path / "branch.json",
        (early, late),
        config={"problem": "tsp"},
        late_checkpoint="checkpoint.ckpt",
        problem_size=8,
        branch_count=3,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["source"] == "real_same_prefix_forced_action_rollout"
    assert payload["policy_states"] == ["early", "late"]
    assert len(payload["probes"]) == 2

    analysis = analyze(
        destination,
        step_size=1.0,
        repeats=4,
        seed=9,
    )
    assert analysis["schema"] == "nco-matched-branch-analysis-v2"
    assert analysis["probe_count"] == 4
    assert analysis["program_count"] == 32
