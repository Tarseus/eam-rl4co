import os
import sys
import json

import pytest
import torch

from rl4co.envs import (
    ATSPEnv,
    CVRPMVCEnv,
    FJSPEnv,
    JSSPEnv,
    PDPEnv,
    PDPRuinRepairEnv,
    TSPEnv,
    TSPkoptEnv,
)
from rl4co.models.rl import A2C, PPO, REINFORCE
from rl4co.models.zoo import (
    DACT,
    GLOP,
    MDAM,
    N2S,
    POMO,
    ActiveSearch,
    AttentionModelPolicy,
    DeepACO,
    EASEmb,
    EASLay,
    HeterogeneousAttentionModel,
    L2DPPOModel,
    MatNet,
    NARGNNPolicy,
    NeuOpt,
    PolyNet,
    SymNCO,
)
from rl4co.models.zoo.pomo.po4cops_tsp_policy import (
    PO4COPsTSPPolicy,
    _select_actions_from_probs,
)
from rl4co.utils import RL4COTrainer
from rl4co.utils.meta_trainer import ReptileCallback
from rl4co.utils.test_utils import generate_env_data

# Get env variable MAC_OS_GITHUB_RUNNER
if "MAC_OS_GITHUB_RUNNER" in os.environ:
    accelerator = "cpu"
else:
    accelerator = "auto"


# Test out simple training loop and test with multiple baselines
@pytest.mark.parametrize("baseline", ["rollout", "exponential", "mean", "no", "critic"])
def test_reinforce(baseline):
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = REINFORCE(
        env,
        policy,
        baseline=baseline,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_a2c():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = A2C(env, policy, train_data_size=10, val_data_size=10, test_data_size=10)
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_trainer_max_epoch_alias():
    trainer = RL4COTrainer(max_epoch=1, devices=1, accelerator=accelerator)
    assert trainer.max_epochs == 1

    trainer = RL4COTrainer(
        max_epochs=1, max_epoch=1, devices=1, accelerator=accelerator
    )
    assert trainer.max_epochs == 1

    with pytest.raises(ValueError, match="max_epoch"):
        RL4COTrainer(max_epochs=2, max_epoch=1, devices=1, accelerator=accelerator)


def test_ppo():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)
    model = PPO(env, policy, train_data_size=10, val_data_size=10, test_data_size=10)
    trainer = RL4COTrainer(
        max_epochs=1, gradient_clip_val=None, devices=1, accelerator=accelerator
    )
    trainer.fit(model)
    trainer.test(model)


def test_symnco():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = SymNCO(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        num_augment=2,
        num_starts=20,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_ham():
    env = PDPEnv(generator_params=dict(num_loc=20))
    model = HeterogeneousAttentionModel(
        env, train_data_size=10, val_data_size=10, test_data_size=10
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_matnet():
    env = ATSPEnv(generator_params=dict(num_loc=20))
    model = MatNet(
        env,
        baseline="shared",
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_mdam():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = MDAM(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


def test_pomo_reptile():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=6,
        num_heads=8,
        normalization="instance",
        use_graph_context=False,
    )
    model = POMO(
        env,
        policy,
        batch_size=5,
        train_data_size=5 * 3,
        val_data_size=10,
        test_data_size=10,
    )
    meta_callback = ReptileCallback(
        data_type="size",
        sch_bar=0.9,
        num_tasks=2,
        alpha=0.99,
        alpha_decay=0.999,
        min_size=20,
        max_size=50,
    )
    trainer = RL4COTrainer(
        max_epochs=2,
        callbacks=[meta_callback],
        devices=1,
        accelerator=accelerator,
        limit_train_batches=3,
    )
    trainer.fit(model)
    trainer.test(model)


def test_pomo_po_loss_smoke():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = POMO(
        env,
        loss_type="po_loss",
        alpha=1.0,
        num_augment=1,
        batch_size=4,
        train_data_size=8,
        val_data_size=8,
        test_data_size=8,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        devices=1,
        accelerator=accelerator,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
    )
    trainer.fit(model)
    trainer.test(model)


def test_pomo_exponential_po_loss_smoke():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = POMO(
        env,
        loss_type="po_loss",
        po_impl="exponential",
        alpha=1.0,
        num_augment=1,
        batch_size=4,
        train_data_size=8,
        val_data_size=8,
        test_data_size=8,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        devices=1,
        accelerator=accelerator,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
    )
    trainer.fit(model)
    trainer.test(model)


def test_pomo_po4cops_compat_po_loss_smoke():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = POMO(
        env,
        loss_type="po_loss",
        alpha=0.05,
        num_augment=1,
        num_starts=20,
        batch_size=4,
        train_data_size=8,
        val_data_size=8,
        test_data_size=8,
        policy_kwargs={
            "po4cops_compat": True,
            "embed_dim": 128,
            "num_encoder_layers": 6,
            "decoder_layer_num": 1,
            "qkv_dim": 16,
            "num_heads": 8,
            "feedforward_hidden": 512,
            "tanh_clipping": 50,
            "eval_type": "argmax",
        },
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        devices=1,
        accelerator=accelerator,
        precision="32-true",
        gradient_clip_val=None,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
    )
    trainer.fit(model)
    trainer.test(model)


def test_po4cops_tsp_policy_supports_official_bopo_same_start():
    env, x = generate_env_data("tsp", size=20, batch_size=2)
    td = env.reset(x)
    policy = PO4COPsTSPPolicy(
        env_name=env.name,
        start_node="same",
        logit_clipping=10,
        eval_type="hybrid",
    )

    out = policy(td, env, phase="train", num_starts=32, return_actions=True)

    assert out["reward"].shape == (64,)
    assert out["actions"].shape[0] == 64
    assert (out["actions"][:, 0] == 0).all()


def test_po4cops_tsp_hybrid_keeps_one_greedy_line():
    probs = torch.tensor(
        [
            [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.3, 0.2, 0.5]],
            [[0.4, 0.1, 0.5], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]],
        ]
    )

    selected = _select_actions_from_probs(probs, use_sampling=True, use_hybrid=True)

    assert selected.shape == probs.shape[:2]
    assert selected[:, 0].equal(probs[:, 0].argmax(dim=-1))


def test_pomo_pref_pair_artifact_smoke(tmp_path):
    builder_payload = {
        "id": "g_unit",
        "ir": {
            "name": "all_pairs_builder",
            "intuition": "use all improving pairs",
            "implementation_hint": {
                "expects": ["objective", "log_prob"],
                "returns": "PrefBatch",
                "mode": "pairwise",
            },
            "code": (
                "def generated_builder(feature_cache, extra):\n"
                "    objective = feature_cache['objective']\n"
                "    mask = objective[:, :, None] < objective[:, None, :]\n"
                "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
                "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'all_pairs'})\n"
            ),
        },
    }
    loss_payload = {
        "id": "f_unit",
        "ir": {
            "name": "bt_loss",
            "intuition": "stable Bradley-Terry pairwise loss",
            "pseudocode": "loss = -logsigmoid(alpha * (log_prob_w - log_prob_l))",
            "hyperparams": {},
            "operators_used": ["logsigmoid"],
            "implementation_hint": {
                "expects": ["log_prob_w", "log_prob_l", "cost_a", "cost_b", "weight"],
                "returns": "scalar",
                "mode": "pairwise",
            },
            "code": (
                "def generated_loss(batch, model_output, extra):\n"
                "    alpha = float(extra.get('alpha', 1.0)) if isinstance(extra, dict) else 1.0\n"
                "    margin = alpha * (batch['log_prob_w'] - batch['log_prob_l'])\n"
                "    loss = -ops.logsigmoid(margin)\n"
                "    weight = batch.get('weight', None)\n"
                "    if weight is not None:\n"
                "        loss = loss * weight\n"
                "    return loss.mean()\n"
            ),
            "theoretical_basis": "Pairwise logistic preference loss.",
        },
    }
    pair_payload = {"g_id": "g_unit", "f_id": "f_unit"}

    builder_path = tmp_path / "best_builder.json"
    loss_path = tmp_path / "best_loss.json"
    pair_path = tmp_path / "best_pair.json"
    builder_path.write_text(json.dumps(builder_payload), encoding="utf-8")
    loss_path.write_text(json.dumps(loss_payload), encoding="utf-8")
    pair_path.write_text(json.dumps(pair_payload), encoding="utf-8")

    env = TSPEnv(generator_params=dict(num_loc=20))
    model = POMO(
        env,
        loss_type="free_loss",
        pref_pair_json_path=str(pair_path),
        num_augment=1,
        batch_size=4,
        train_data_size=8,
        val_data_size=8,
        test_data_size=8,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        devices=1,
        accelerator=accelerator,
        precision="32-true",
        gradient_clip_val=None,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
    )
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.parametrize("SearchMethod", [ActiveSearch, EASEmb, EASLay])
def test_search_methods(SearchMethod):
    env = TSPEnv(generator_params=dict(num_loc=20))
    batch_size = 2 if SearchMethod not in [ActiveSearch] else 1
    dataset = env.dataset(2)
    policy = AttentionModelPolicy(env_name=env.name)
    model = SearchMethod(env, policy, dataset, max_iters=2, batch_size=batch_size)
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.skipif(
    "torch_geometric" not in sys.modules, reason="PyTorch Geometric not installed"
)
def test_nargnn():
    env = TSPEnv(generator_params=dict(num_loc=20))
    policy = NARGNNPolicy(env_name=env.name)
    model = REINFORCE(
        env, policy=policy, train_data_size=10, val_data_size=10, test_data_size=10
    )
    trainer = RL4COTrainer(
        max_epochs=1, gradient_clip_val=None, devices=1, accelerator=accelerator
    )
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.skipif(
    "torch_geometric" not in sys.modules, reason="PyTorch Geometric not installed"
)
@pytest.mark.skipif("numba" not in sys.modules, reason="Numba not installed")
@pytest.mark.parametrize("use_local_search", [False])
def test_deepaco(use_local_search):
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = DeepACO(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        train_with_local_search=use_local_search,
        policy_kwargs={"n_ants": 5, "aco_kwargs": {"use_local_search": use_local_search}},
    )
    trainer = RL4COTrainer(
        max_epochs=1, gradient_clip_val=1, devices=1, accelerator=accelerator
    )
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.skipif(
    "torch_geometric" not in sys.modules, reason="PyTorch Geometric not installed"
)
@pytest.mark.parametrize(
    "Environment", [TSPEnv] if "numba" not in sys.modules else [TSPEnv, CVRPMVCEnv]
)
def test_glop(Environment):
    import torch

    def dummy_solver(c):
        return torch.arange(c.shape[1] - 1, -1, -1).unsqueeze(0).expand(c.shape[0], -1)

    env = Environment(generator_params=dict(num_loc=50))
    model = GLOP(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        policy_kwargs={
            "subprob_solver": dummy_solver,
        },
    )
    trainer = RL4COTrainer(
        max_epochs=1, gradient_clip_val=1, devices=1, accelerator=accelerator
    )
    trainer.fit(model)
    trainer.test(model)


def test_n2s():
    env = PDPRuinRepairEnv(generator_params=dict(num_loc=20))
    model = N2S(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        n_step=2,
        T_train=4,
        T_test=4,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=0.05,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)


def test_dact():
    env = TSPkoptEnv(generator_params=dict(num_loc=20), k_max=2)
    model = DACT(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        n_step=2,
        T_train=4,
        T_test=4,
        CL_best=True,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=0.05,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)


def test_neuopt():
    env = TSPkoptEnv(generator_params=dict(num_loc=20), k_max=4)
    model = NeuOpt(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        n_step=2,
        T_train=4,
        T_test=4,
        CL_best=True,
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=0.05,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.parametrize("env_cls", [FJSPEnv, JSSPEnv])
def test_l2d_ppo(env_cls):
    env = env_cls(stepwise_reward=True, _torchrl_mode=True)
    model = L2DPPOModel(
        env, train_data_size=10, val_data_size=10, test_data_size=10, buffer_size=1000
    )
    trainer = RL4COTrainer(
        max_epochs=1,
        gradient_clip_val=0.05,
        devices=1,
        accelerator=accelerator,
    )
    trainer.fit(model)
    trainer.test(model)


def test_polynet():
    env = TSPEnv(generator_params=dict(num_loc=20))
    model = PolyNet(
        env,
        k=10,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model)
    trainer.test(model)
