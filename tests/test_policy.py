import pytest

from rl4co.models import AttentionModelPolicy, N2SPolicy, PointerNetworkPolicy
from rl4co.envs import FFSPEnv
from rl4co.models.zoo.matnet.policy import MultiStageFFSPPolicy
from rl4co.utils.ops import select_start_nodes
from rl4co.utils.test_utils import generate_env_data


# Main autorergressive policy: rollout over multiple envs since it is the base
@pytest.mark.parametrize(
    "env_name",
    [
        "tsp",
        "cvrp",
        "cvrptw",
        "sdvrp",
        "mtsp",
        "op",
        "pctsp",
        "spctsp",
        "dpp",
        "mdpp",
        "smtwtp",
        "flp",
        "mcp",
    ],
)
def test_am_policy(env_name, size=20, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    policy = AttentionModelPolicy(env_name=env.name)
    out = policy(td, env, decode_type="greedy")
    assert out["reward"].shape == (batch_size,)


@pytest.mark.parametrize(
    "env_name", ["tsp", "cvrp", "cvrptw", "pctsp", "spctsp", "sdvrp", "op", "pdp"]
)
@pytest.mark.parametrize("policy_cls", [AttentionModelPolicy])
def test_policy_multistart(env_name, policy_cls, size=20, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    policy = policy_cls(env_name=env.name)
    num_starts = size // 2 if env.name in ["pdp"] else size
    out = policy(
        td,
        env,
        decode_type="multistart_greedy",
        num_starts=num_starts,
        select_start_nodes_fn=select_start_nodes,
    )
    assert out["reward"].shape == (
        batch_size * num_starts,
    )  # to evaluate, we could just unbatchify


@pytest.mark.parametrize(
    "env_name",
    ["tsp", "cvrp", "cvrptw", "pctsp", "spctsp", "sdvrp", "op", "pdp"],
)
@pytest.mark.parametrize("select_best", [True, False])
def test_beam_search(env_name, select_best, size=20, batch_size=2):
    env, x = generate_env_data(env_name, size, batch_size)
    td = env.reset(x)
    policy = AttentionModelPolicy(env_name=env.name)
    beam_width = size // 2 if env.name in ["pdp"] else size
    out = policy(
        td, env, decode_type="beam_search", beam_width=beam_width, select_best=select_best
    )

    if select_best:
        expected_shape = (batch_size,)
    else:
        expected_shape = (batch_size * beam_width,)
    assert out["reward"].shape == expected_shape


def test_pointer_network(size=20, batch_size=2):
    env, x = generate_env_data("tsp", size, batch_size)
    td = env.reset(x)
    policy = PointerNetworkPolicy(env_name=env.name)
    out = policy(td, env, decode_type="greedy")
    assert out["reward"].shape == (batch_size,)


def test_N2S(size=20, batch_size=2):
    env, x = generate_env_data("pdp_ruin_repair", size, batch_size)
    td = env.reset(x)
    policy = N2SPolicy(env_name=env.name)
    out = policy(td, env, decode_type="greedy")
    assert out["cost_bsf"].shape == (batch_size,)


def test_multistage_ffsp_policy_clears_decoder_cache():
    env = FFSPEnv(
        generator_params={
            "num_stage": 2,
            "num_machine": 2,
            "num_job": 4,
            "flatten_stages": False,
        }
    )
    td = env.reset(batch_size=[2])
    policy = MultiStageFFSPPolicy(
        stage_cnt=env.num_stage,
        embed_dim=16,
        num_heads=2,
        num_encoder_layers=1,
        feedforward_hidden=16,
        train_decode_type="greedy",
    )

    out = policy(td, env, phase="train", num_starts=2, return_actions=False)
    loss = out["log_likelihood"].sum()
    loss.backward()

    assert all(decoder.cached_embs is None for decoder in policy.decoders)
