import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _dummy_openai_client(*, responses: list[str]):
    """Minimal OpenAI client stub for `client.chat.completions.create(...)`."""

    def create(*, model, messages, temperature=0.7, **kwargs):  # noqa: ARG001
        content = messages[0]["content"]
        # Mutation prompts include PARENT_JSON; generation prompts do not.
        if "PARENT_JSON:" in content:
            payload = responses[1]
        else:
            payload = responses[0]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_eoh_smoke_generates_m1(monkeypatch, tmp_path):
    # Ensure local `PTP/` modules are importable under pytest.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "PTP"))

    # Ensure the LLM layer doesn't fail early due to env checks.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")

    # Good, gate-friendly pairwise loss that depends on objective signals.
    gen_json = json.dumps(
        {
            "name": "smoke_gen",
            "intuition": "Stable logsigmoid with a cost-gap term for objective sensitivity.",
            "pseudocode": "loss = -logsigmoid(alpha*(lpw-lpl) - beta*clamp(cost_gap))",
            "hyperparams": {"alpha": 1.0, "beta": 0.1, "gap_clip": 10.0},
            "operators_used": ["logsigmoid", "clamp"],
            "implementation_hint": {
                "expects": ["log_prob_w", "log_prob_l", "cost_a", "cost_b", "weight"],
                "returns": "scalar",
                "mode": "pairwise",
            },
            "code": (
                "def generated_loss(batch, model_output, extra):\n"
                "    lpw = batch['log_prob_w']\n"
                "    lpl = batch['log_prob_l']\n"
                "    cost_a = batch['cost_a']\n"
                "    cost_b = batch['cost_b']\n"
                "    w = batch.get('weight', None)\n"
                "    hp = extra.get('hyperparams', {}) if isinstance(extra, dict) else {}\n"
                "    alpha = float(extra.get('alpha', hp.get('alpha', 1.0)))\n"
                "    beta = float(hp.get('beta', 0.1))\n"
                "    gap_clip = float(hp.get('gap_clip', 10.0))\n"
                "    gap = cost_b - cost_a\n"
                "    gap = ops.clamp(gap, min=0.0, max=gap_clip)\n"
                "    x = ops.clamp(alpha * (lpw - lpl) - beta * gap, min=-20.0, max=20.0)\n"
                "    loss = -ops.logsigmoid(x)\n"
                "    if w is not None:\n"
                "        loss = loss * w\n"
                "    return loss.mean()\n"
            ),
        }
    )
    mut_json = json.dumps(
        {
            "name": "smoke_mut",
            "intuition": "Mutation: slightly higher beta for stronger objective conditioning.",
            "pseudocode": "same as parent, beta tuned",
            "hyperparams": {"alpha": 1.0, "beta": 0.12, "gap_clip": 10.0},
            "operators_used": ["logsigmoid", "clamp"],
            "implementation_hint": {
                "expects": ["log_prob_w", "log_prob_l", "cost_a", "cost_b", "weight"],
                "returns": "scalar",
                "mode": "pairwise",
            },
            "code": json.loads(gen_json)["code"],
        }
    )

    # Monkeypatch the OpenAI client acquisition so no network calls occur.
    from ptp_discovery import free_loss_llm_ops as llm_ops

    monkeypatch.setattr(
        llm_ops,
        "_get_openai_client",
        lambda: _dummy_openai_client(responses=[gen_json, mut_json]),
    )

    # Keep the smoke test fast: disable mp workers and stub the expensive fidelity eval.
    from ptp_discovery import free_loss_eoh_loop as eoh_loop

    def _fast_eval(*args, **kwargs):  # noqa: ARG001
        objectives = [1.0] * 20
        return {
            "hf_like_score": 1.0,
            "validation_objective": 1.0,
            "generalization_penalty": 0.0,
            "generalization_objectives": {},
            "epoch_objective_mean": None,
            "epoch_baseline_violations": None,
            "epoch_better_than_baseline": True,
            "epoch_tail_better_than_baseline": True,
            "epoch_eval": {"enabled": True, "epochs_total": len(objectives), "objectives": objectives},
            "epoch_window_eval": {"k": 10, "early_mean": 1.0, "late_mean": 1.0, "objectives": objectives},
            "baseline_epoch_window_eval": {"early_mean": None, "late_mean": None},
            "epoch_window_margins": None,
            "epoch_window_violations": 1,
            "epoch_window_better_than_baseline": False,
            "train_score_mean": 0.0,
            "train_loss_mean": 0.0,
            "pair_count": 1,
            "early_eval": {
                "enabled": False,
                "steps": 0,
                "baseline_validation_objective": None,
                "candidate_validation_objective": None,
                "early_stopped": False,
            },
            "size_objectives": {},
            "size_aggregation": "cvar",
        }

    monkeypatch.setattr(eoh_loop, "evaluate_free_loss_candidate", _fast_eval)

    metrics_path = tmp_path / "baseline_metrics.csv"
    metrics_lines = ["epoch,val/reward"]
    for epoch in range(1, 21):
        metrics_lines.append(f"{epoch},-1.0")
    metrics_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    # Minimal config (generations=2 triggers gen=1 which must choose M1 when only mutation prompt exists).
    out_root = tmp_path / "runs"
    config_path = tmp_path / "eoh_smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 123",
                "backend: rl4co",
                "env_name: tsp",
                "objective_sign: neg_reward",
                "rollout_strategy: auto",
                "generations: 2",
                "population_size: 4",
                "elite_size: 2",
                "init_llm: 4",
                "max_repair_rounds: 1",
                "train_problem_size: 20",
                "valid_problem_sizes: [20]",
                "train_batch_size: 64",
                "pomo_size: 20",
                "hf_epochs: 20",
                "hf_instances_per_epoch: 64",
                "f1_steps: 1",
                "baseline_epoch_window_k: 10",
                "device: cpu",
                "eval_mp_enabled: false",
                "num_validation_episodes: 16",
                "validation_batch_size: 16",
                "pref_semantic_gate_enabled: false",
                "co_gate_enabled: false",
                "hidden_dynamic_gates_enabled: false",
                "operator_whitelist: [logsigmoid, clamp, softplus, sigmoid, exp, log, tanh, relu, normalize, zscore, rank_gap]",
                "prompts:",
                "  generation: PTP/prompts/free_loss_generation.txt",
                "  mutation: PTP/prompts/free_loss_mutation.txt",
                "  repair: PTP/prompts/free_loss_repair.txt",
                "  expects_repair: PTP/prompts/free_loss_expects_repair.txt",
                "baseline:",
                f"  metrics_csv: {metrics_path.as_posix()}",
                "  checkpoint_epoch: 0",
                "  val_column: val/reward",
                f"output_root: {out_root.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    from ptp_discovery.free_loss_eoh_loop import run_free_loss_eoh

    run_free_loss_eoh(str(config_path))

    # Find the created run directory under output_root.
    run_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    assert run_dirs, "Expected at least one run directory under output_root"
    run_dir = sorted(run_dirs)[-1]

    assert (run_dir / "checkpoint.json").is_file()

    gate_path = run_dir / "gate_reports.jsonl"
    assert gate_path.is_file()
    found = False
    for line in gate_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("llm_op") in {"E1", "M1"}:
            found = True
            break
    assert found, "Expected at least one gate record with llm_op in {'E1','M1'}"

    fitness_path = run_dir / "fitness_scores.jsonl"
    assert fitness_path.is_file()
    recs = [
        json.loads(line)
        for line in fitness_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert recs, "Expected at least one fitness record"
    assert any(r.get("epoch_window_better_than_baseline") is False for r in recs)
    assert any(r.get("better_than_baseline") is False for r in recs)
