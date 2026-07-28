from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from tensordict import TensorDict


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PILOT = HERE.parent / "rfps_feature_pilot"
RUNTIME_POMO = PILOT / "find_pref_runtime" / "rl4co" / "models" / "zoo" / "pomo"
sys.path.insert(0, str(ROOT))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    for path in (checkpoint, source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not RUNTIME_POMO.is_dir():
        raise FileNotFoundError(RUNTIME_POMO)

    import rl4co.models.zoo.pomo as pomo_package

    runtime_path = str(RUNTIME_POMO.resolve())
    if runtime_path not in pomo_package.__path__:
        pomo_package.__path__.append(runtime_path)

    from rl4co.envs import TSPEnv
    from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy

    with np.load(source) as archive:
        original = {name: np.asarray(archive[name]).copy() for name in archive.files}
    locations = torch.as_tensor(original["locations"], dtype=torch.float32)
    actions = torch.as_tensor(original["actions"], dtype=torch.long)
    p0 = torch.as_tensor(original["p0"], dtype=torch.float32)
    batch_size, num_starts, horizon = actions.shape

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = {
        key.removeprefix("policy."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("policy.")
    }
    policy = PO4COPsTSPPolicy(
        env_name="tsp",
        start_node="pomo",
        eval_type="argmax",
        train_decode_type="sampling",
        val_decode_type="greedy",
        test_decode_type="greedy",
    )
    incompatible = policy.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    policy.eval()

    captured_l2: list[torch.Tensor] = []
    original_forward = policy.decoder.forward_with_cache

    def capture_forward(
        self,
        encoded_last_node: torch.Tensor,
        ninf_mask: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = original_forward(
            encoded_last_node,
            ninf_mask,
            *cache_tensors,
        )
        captured_l2.append(probabilities.square().sum(dim=2).detach().cpu())
        return probabilities

    policy.decoder.forward_with_cache = MethodType(
        capture_forward, policy.decoder
    )

    env = TSPEnv(
        generator_params={"num_loc": locations.shape[1]},
        device="cpu",
        seed=0,
    )
    reset_input = TensorDict(
        {"locs": locations},
        batch_size=[batch_size],
    )
    reset_td = env.reset(reset_input)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    with torch.inference_mode():
        result = policy(
            reset_td,
            env,
            phase="train",
            num_starts=num_starts,
            return_actions=True,
            return_entropy=False,
            return_sum_log_likelihood=False,
            forced_actions=actions,
        )

    step_log_probs_flat = result["log_likelihood"].cpu()
    step_log_probs = (
        step_log_probs_flat.reshape(num_starts, batch_size, horizon)
        .permute(1, 0, 2)
        .contiguous()
    )
    if len(captured_l2) != horizon - 1:
        raise RuntimeError(
            f"expected {horizon - 1} probability captures, "
            f"found {len(captured_l2)}"
        )
    step_prob_l2 = torch.cat(
        [
            torch.ones(
                (batch_size, num_starts, 1),
                dtype=step_log_probs.dtype,
            ),
            torch.stack(captured_l2, dim=2),
        ],
        dim=2,
    )

    reconstructed = step_log_probs.sum(dim=2)
    max_log_likelihood_error = float((reconstructed - p0).abs().max())
    replay_actions = (
        result["actions"]
        .cpu()
        .reshape(num_starts, batch_size, horizon)
        .permute(1, 0, 2)
        .contiguous()
    )
    action_match = bool(torch.equal(replay_actions, actions))
    if max_log_likelihood_error > 1e-5:
        raise RuntimeError(
            "selected step log probabilities do not reproduce p0: "
            f"max error {max_log_likelihood_error}"
        )
    if not action_match:
        raise RuntimeError("forced-action replay did not reproduce actions")
    if not torch.isfinite(step_log_probs).all():
        raise RuntimeError("non-finite selected step log probabilities")
    if not torch.isfinite(step_prob_l2).all():
        raise RuntimeError("non-finite squared-probability sums")

    selected_probabilities = step_log_probs.exp()
    euclidean_terms = (
        1.0 - 2.0 * selected_probabilities + step_prob_l2
    )
    fisher_terms = selected_probabilities.reciprocal() - 1.0
    minimum_euclidean_term = float(euclidean_terms.min())
    minimum_fisher_term = float(fisher_terms.min())
    if minimum_euclidean_term < -1e-5 or minimum_fisher_term < -1e-5:
        raise RuntimeError(
            "negative norm term: "
            f"euclidean={minimum_euclidean_term}, "
            f"fisher={minimum_fisher_term}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        step_log_probs=step_log_probs.numpy(),
        step_prob_l2=step_prob_l2.numpy(),
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "source": str(source),
        "source_sha256": file_sha256(source),
        "batch_size": batch_size,
        "num_starts": num_starts,
        "horizon": horizon,
        "max_log_likelihood_error": max_log_likelihood_error,
        "action_match": action_match,
        "selected_probability_min": float(selected_probabilities.min()),
        "selected_probability_median": float(selected_probabilities.median()),
        "selected_probability_max": float(selected_probabilities.max()),
        "euclidean_weight_median": float(
            euclidean_terms.sum(dim=2).median()
        ),
        "fisher_weight_median": float(fisher_terms.sum(dim=2).median()),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
