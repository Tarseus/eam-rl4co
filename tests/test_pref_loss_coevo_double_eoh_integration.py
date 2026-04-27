from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dummy_openai_client(state: dict):
    def _builder_json(*, good: bool) -> str:
        if not good:
            code = (
                "def generated_builder(feature_cache, extra):\n"
                "    return PrefBatch(mode='pairwise', pair_idx=None, weight=None, meta={'bad': True})\n"
            )
        else:
            code = (
                "def generated_builder(feature_cache, extra):\n"
                "    objective = feature_cache['objective']\n"
                "    mask = objective[:, :, None] < objective[:, None, :]\n"
                "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
                "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'all_pairs'})\n"
            )
        return json.dumps(
            {
                "name": "llm_builder",
                "intuition": "dummy",
                "implementation_hint": {"expects": ["objective", "log_prob"], "returns": "PrefBatch", "mode": "pairwise"},
                "code": code,
            }
        )

    def _loss_json(*, good: bool) -> str:
        if not good:
            # Contains a syntax error but still includes the required function header for parsing.
            code = "def generated_loss(batch, model_output, extra):\n    return (\n"
        else:
            code = (
                "def generated_loss(batch, model_output, extra):\n"
                "    lpw = batch['log_prob_w']\n"
                "    lpl = batch['log_prob_l']\n"
                "    w = batch.get('weight')\n"
                "    x = (lpw - lpl)\n"
                "    loss = -ops.logsigmoid(x)\n"
                "    if w is not None:\n"
                "        loss = loss * w\n"
                "    return loss.mean()\n"
            )
        return json.dumps(
            {
                "name": "llm_loss",
                "intuition": "dummy",
                "pseudocode": "loss = -logsigmoid(lpw-lpl)",
                "hyperparams": {"scale": 1.0},
                "operators_used": ["logsigmoid"],
                "implementation_hint": {"expects": ["log_prob_w", "log_prob_l", "weight"], "returns": "scalar", "mode": "pairwise"},
                "code": code,
                "theoretical_basis": "",
            }
        )

    def _content(prompt: str) -> str:
        p = prompt or ""
        is_builder = ("generated_builder" in p) or ("preference builder" in p.lower()) or ("PrefBatch" in p)
        is_loss = ("generated_loss" in p) or ("free-form preference loss" in p.lower())

        if '"side": "builder"' in p and '"op_type": "M1"' in p and not state.get("builder_failed_once", False):
            state["builder_failed_once"] = True
            return _builder_json(good=False)
        if '"side": "loss"' in p and '"op_type": "M2"' in p and not state.get("loss_failed_once", False):
            state["loss_failed_once"] = True
            return _loss_json(good=False)

        if "CANDIDATE_AND_FAILURE_JSON" in p:
            # Treat as repair/simplify: return a good candidate.
            if is_builder and not is_loss:
                return _builder_json(good=True)
            return _loss_json(good=True)

        if is_builder and not is_loss:
            return _builder_json(good=True)
        return _loss_json(good=True)

    class _ChatCompletions:
        @staticmethod
        def create(model, messages, temperature):  # noqa: ANN001
            prompt = ""
            if messages and isinstance(messages, list) and isinstance(messages[0], dict):
                prompt = str(messages[0].get("content", ""))
            content = _content(prompt)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class _Chat:
        completions = _ChatCompletions()

    class _Client:
        chat = _Chat()

    return _Client()


def _patch_fast_proxy(monkeypatch, loop_module):
    import torch

    def _fast_rollout_feature_cache(*args, **kwargs):  # noqa: ANN001
        batch_size = int(kwargs.get("batch_size", 8) or 8)
        cfg = kwargs.get("cfg")
        K = 16
        try:
            pomo_size = getattr(cfg, "pomo_size", None)
            if pomo_size is not None:
                K = int(pomo_size)
        except Exception:  # noqa: BLE001
            K = 16
        return loop_module._dummy_feature_cache(batch_size=batch_size, k=K, variant="visible")

    def _fast_proxy_metrics_for_pair_on_batch(*args, **kwargs):  # noqa: ANN001
        pref_batch = kwargs.get("pref_batch")
        pair_count = int(getattr(pref_batch, "num_examples", lambda: 0)())
        return {
            "loss": 1.0,
            "loss_swap": None,
            "effective_grad_ratio": 0.5,
            "grad_w_pass_rate": 1.0,
            "grad_l_pass_rate": 1.0,
            "swap_ok": True,
            "ess_ratio": 0.5,
            "pair_count": pair_count,
            "joint_ok": True,
            "joint_reason": "ok",
            "joint_trace": {"failed_gate": None, "observed": {"loss": 1.0, "effective_grad_ratio": 0.5}},
        }

    monkeypatch.setattr(loop_module, "build_or_get_rollout_feature_cache", _fast_rollout_feature_cache)
    monkeypatch.setattr(loop_module, "proxy_metrics_for_pair_on_batch", _fast_proxy_metrics_for_pair_on_batch)


def test_pref_loss_coevo_double_eoh_smoke(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.free_loss_llm_ops as llm_ops
    import ptp_discovery.pref_loss_coevo_loop as loop

    _patch_fast_proxy(monkeypatch, loop)

    state: dict = {}
    monkeypatch.setattr(llm_ops, "_get_openai_client", lambda: _dummy_openai_client(state))

    cfg = {
        "seed": 0,
        "output_root": str(tmp_path / "runs"),
        "generations": 2,
        "pop_g": 6,
        "pop_f": 6,
        "elite_g": 2,
        "elite_f": 2,
        "pairing_budget_per_gen": 12,
        "cheap_gate_on": True,
        "high_fidelity_on": False,
        # Keep HF signature stable but avoid baseline loading.
        "backend": "rl4co",
        "env_name": "tsp",
        "policy_name": "pomo",
        "generator_params": {"num_loc": 20},
        "hf_epochs": 0,
        "hf_instances_per_epoch": 0,
        "train_problem_size": 20,
        "valid_problem_sizes": [20],
        "train_batch_size": 8,
        "pomo_size": 16,
        "device": "cpu",
        # Proxy
        "proxy_problem_size": 20,
        "proxy_batch_size": 8,
        "proxy_batches": 1,
        # Gates
        "builder_max_pairs_per_instance": 4096,
        # LLM ops: force E1/E2/M1/M2 for both sides
        "builder_llm": {"enabled": True, "offline_mode": False, "init_num_E1": 1, "init_num_E2": 1, "init_num_M1": 1, "init_num_M2": 1, "num_E1": 1, "num_E2": 1, "num_M1": 1, "num_M2": 1, "repair": {"enabled": True, "max_attempts": 1, "simplify_first": True}},
        "loss_llm": {"enabled": True, "offline_mode": False, "init_num_E1": 1, "init_num_E2": 1, "init_num_M1": 1, "init_num_M2": 1, "num_E1": 1, "num_E2": 1, "num_M1": 1, "num_M2": 1, "repair": {"enabled": True, "max_attempts": 1, "simplify_first": True}},
    }
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loop.run_pref_loss_coevo(str(cfg_path))

    out_root = tmp_path / "runs"
    run_dirs = sorted([p for p in out_root.iterdir() if p.is_dir()])
    assert run_dirs, "No run directory created under output_root"
    run_dir = run_dirs[-1]

    for name in ("builders.jsonl", "losses.jsonl", "pairs.jsonl", "gate_reports.jsonl", "checkpoint.json"):
        assert (run_dir / name).is_file()

    cache_path = run_dir / "llm_cache.jsonl"
    assert cache_path.is_file()

    builder_ops = set()
    loss_ops = set()
    repair_seen = False
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        op = str(rec.get("llm_op", ""))
        prompt_path = str(rec.get("prompt_path", "") or "")
        if "pref_builder_" in prompt_path:
            builder_ops.add(op)
        if "free_loss_" in prompt_path:
            loss_ops.add(op)
        if op == "REPAIR":
            repair_seen = True

    assert {"E1", "E2", "M1", "M2"}.issubset(builder_ops)
    assert {"E1", "E2", "M1", "M2"}.issubset(loss_ops)
    assert repair_seen is True


def test_pref_loss_coevo_llm_disabled_fallback(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.free_loss_llm_ops as llm_ops
    import ptp_discovery.pref_loss_coevo_loop as loop

    _patch_fast_proxy(monkeypatch, loop)

    monkeypatch.setattr(llm_ops, "_get_openai_client", lambda: (_ for _ in ()).throw(AssertionError("_get_openai_client called")))

    cfg = {
        "seed": 0,
        "output_root": str(tmp_path / "runs"),
        "generations": 2,
        "pop_g": 6,
        "pop_f": 6,
        "elite_g": 2,
        "elite_f": 2,
        "pairing_budget_per_gen": 12,
        "cheap_gate_on": True,
        "high_fidelity_on": False,
        "backend": "rl4co",
        "env_name": "tsp",
        "policy_name": "pomo",
        "generator_params": {"num_loc": 20},
        "hf_epochs": 0,
        "hf_instances_per_epoch": 0,
        "train_problem_size": 20,
        "valid_problem_sizes": [20],
        "train_batch_size": 8,
        "pomo_size": 16,
        "device": "cpu",
        "proxy_problem_size": 20,
        "proxy_batch_size": 8,
        "proxy_batches": 1,
        "builder_llm": {"enabled": False},
        "loss_llm": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loop.run_pref_loss_coevo(str(cfg_path))

