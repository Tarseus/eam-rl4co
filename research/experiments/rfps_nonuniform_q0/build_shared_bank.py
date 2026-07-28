from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "rfps_feature_pilot"
REPO_ROOT = HERE.parents[2]
RUNTIME_POMO = PILOT / "find_pref_runtime" / "rl4co" / "models" / "zoo" / "pomo"

SOURCE_FILES = {
    1: {
        "scratch": PILOT / "tsp100_scratch1234_rollout_probes_seed1.npz",
        "warm": PILOT / "tsp100_epoch135_rollout_probes.npz",
    },
    2: {
        "scratch": PILOT / "tsp100_scratch1234_rollout_probes_seed2.npz",
        "warm": PILOT / "tsp100_epoch135_rollout_probes_seed2.npz",
    },
}


def install_runtime() -> tuple[type, type]:
    sys.path.insert(0, str(REPO_ROOT))
    import rl4co.models.zoo.pomo as pomo_package

    runtime_path = str(RUNTIME_POMO.resolve())
    if runtime_path not in pomo_package.__path__:
        pomo_package.__path__.append(runtime_path)
    from rl4co.envs import TSPEnv
    from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy

    return TSPEnv, PO4COPsTSPPolicy


def make_policy(policy_class: type, anchor: str) -> torch.nn.Module:
    torch.manual_seed(1234)
    policy = policy_class(
        env_name="tsp",
        start_node="pomo",
        eval_type="argmax",
        train_decode_type="sampling",
        val_decode_type="greedy",
        test_decode_type="greedy",
    )
    if anchor == "warm":
        payload = torch.load(
            REPO_ROOT / "tsp100_epoch_135.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        state_dict = {
            key.removeprefix("policy."): value
            for key, value in payload["state_dict"].items()
            if key.startswith("policy.")
        }
        policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def score_actions(
    policy: torch.nn.Module,
    env_class: type,
    locations: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_starts, _ = actions.shape
    env = env_class(
        generator_params={"num_loc": locations.shape[-2]},
        device="cpu",
        seed=0,
    )
    reset_td = env.reset(
        TensorDict({"locs": locations}, batch_size=[batch_size])
    )
    with torch.inference_mode():
        output = policy(
            reset_td,
            env,
            phase="train",
            num_starts=num_starts,
            forced_actions=actions,
            return_actions=False,
            return_entropy=False,
            return_sum_log_likelihood=True,
        )
    return output["log_likelihood"].reshape(
        num_starts, batch_size
    ).transpose(0, 1).cpu()


def load_sources(seed_index: int) -> dict[str, np.ndarray]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for anchor, path in SOURCE_FILES[seed_index].items():
        with np.load(path) as archive:
            loaded[anchor] = {
                key: np.asarray(archive[key]).copy() for key in archive.files
            }
    if not np.array_equal(
        loaded["scratch"]["locations"], loaded["warm"]["locations"]
    ):
        raise RuntimeError(f"seed {seed_index}: scratch/warm locations differ")
    return loaded


def build_one(
    seed_index: int,
    output: Path,
    *,
    env_class: type,
    policy_class: type,
) -> dict[str, float | int | str]:
    sources = load_sources(seed_index)
    locations = torch.as_tensor(sources["scratch"]["locations"])
    action_groups = {
        anchor: torch.as_tensor(sources[anchor]["actions"])
        for anchor in ("scratch", "warm")
    }
    actions = torch.cat(
        [action_groups["scratch"], action_groups["warm"]], dim=1
    )
    objective = np.concatenate(
        [sources["scratch"]["objective"], sources["warm"]["objective"]], axis=1
    )
    entropy = np.concatenate(
        [sources["scratch"]["entropy"], sources["warm"]["entropy"]], axis=1
    )
    p_anchor: dict[str, np.ndarray] = {}
    own_errors: dict[str, float] = {}
    for anchor in ("scratch", "warm"):
        policy = make_policy(policy_class, anchor)
        # One POMO multistart batch has one start per node. Score each source
        # group separately and concatenate only after teacher forcing.
        scored = np.concatenate(
            [
                score_actions(
                    policy, env_class, locations, action_groups[source]
                ).numpy()
                for source in ("scratch", "warm")
            ],
            axis=1,
        )
        p_anchor[anchor] = scored
        own_slice = slice(0, 100) if anchor == "scratch" else slice(100, 200)
        own_errors[anchor] = float(
            np.max(
                np.abs(
                    scored[:, own_slice]
                    - sources[anchor]["p0"]
                )
            )
        )
        print(
            f"[shared-bank seed{seed_index}] scored {anchor}; "
            f"own max error={own_errors[anchor]:.3e}",
            flush=True,
        )
        if own_errors[anchor] > 2e-3:
            raise RuntimeError(
                f"seed {seed_index} {anchor}: forced-action score does not "
                f"reproduce stored log likelihoods ({own_errors[anchor]:.3e})"
            )

    log_mu = np.logaddexp(p_anchor["scratch"], p_anchor["warm"]) - math.log(2.0)
    reward = -objective
    advantage = reward - reward.mean(axis=1, keepdims=True)
    seq_len = np.full_like(objective, actions.shape[-1], dtype=np.float32)
    source_anchor = np.concatenate(
        [np.zeros(100, dtype=np.int64), np.ones(100, dtype=np.int64)]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        objective=objective,
        p_scratch=p_anchor["scratch"],
        p_warm=p_anchor["warm"],
        log_mu=log_mu,
        seq_len=seq_len,
        advantage=advantage,
        entropy=entropy,
        locations=locations.numpy(),
        actions=actions.numpy(),
        source_anchor=source_anchor,
    )
    metadata: dict[str, float | int | str] = {
        "seed_index": seed_index,
        "num_instances": int(objective.shape[0]),
        "num_trajectories": int(objective.shape[1]),
        "scratch_own_score_max_error": own_errors["scratch"],
        "warm_own_score_max_error": own_errors["warm"],
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1, 2))
    parser.add_argument("--output-dir", type=Path, default=HERE / "shared_banks")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    env_class, policy_class = install_runtime()
    seed_indices = (args.seed,) if args.seed else (1, 2)
    summaries = []
    for seed_index in seed_indices:
        summaries.append(
            build_one(
                seed_index,
                args.output_dir / f"shared_seed{seed_index}.npz",
                env_class=env_class,
                policy_class=policy_class,
            )
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
