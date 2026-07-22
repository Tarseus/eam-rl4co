from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.data.transforms import StateAugmentation
from rl4co.models import SymNCO
from rl4co.utils.ops import unbatchify
from scripts.eval_downloaded_routing_checkpoints import (
    build_routing_env,
    checkpoint_hparams,
)


TEST_FILE = REPO_ROOT / "data/vrp/vrp50_test_seed1234.npz"
EXPECTED_TEST_SHA256 = (
    "1af0d3502e13ce571048e5437593a8679e8ce79c25c33415be22062c2d9e59d7"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = payload.get("hyper_parameters", {}) or {}
    policy = hparams.get("policy")
    env = hparams.get("env")
    generator = getattr(env, "generator", None)
    state_dict = payload.get("state_dict", {})
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "policy_class": policy.__class__.__name__ if policy is not None else None,
        "env_class": env.__class__.__name__ if env is not None else None,
        "num_loc": getattr(generator, "num_loc", None),
        "checkpoint_num_starts": hparams.get("num_starts"),
        "checkpoint_num_augment": hparams.get("num_augment"),
        "has_projection_head": any("projection_head" in key for key in state_dict),
        "is_eam": hparams.get("ea_kwargs") is not None,
        "ea_kwargs": hparams.get("ea_kwargs"),
    }
    expected = {
        "policy_class": "SymNCOPolicy",
        "env_class": "CVRPEnv",
        "num_loc": 50,
        "has_projection_head": True,
    }
    failures = {
        key: (metadata[key], value)
        for key, value in expected.items()
        if metadata[key] != value
    }
    if failures:
        raise RuntimeError(f"Not a valid CVRP50 Sym-NCO checkpoint: {failures}")
    return metadata


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint: Path) -> tuple[SymNCO, str]:
    hparams = checkpoint_hparams(checkpoint)
    payload = hparams.pop("_checkpoint_payload")
    # EAM-Sym uses the same Sym-NCO network but stores its trainer-only EA
    # settings in the Lightning hparams.  They are not SymNCO constructor args.
    hparams.pop("ea_kwargs", None)
    # We only evaluate the policy; old checkpoints may serialize the training
    # baseline as ``rollout`` even though current SymNCO requires its own
    # baseline identifier at construction time.
    hparams["baseline"] = "symnco"
    env, resolved_test_file = build_routing_env(
        env_name="cvrp", size=50, hparams=hparams, repo_root=REPO_ROOT
    )
    hparams["env"] = env
    model = SymNCO(**hparams)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.data_cfg["generate_default_data"] = False
    model.data_cfg["data_dir"] = str((REPO_ROOT / "data/vrp").resolve())
    return model, str(resolved_test_file)


def evaluate_mode(
    model: SymNCO,
    *,
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    if mode == "none":
        num_augment = 1
        augment = None
    elif mode == "symmetric2":
        num_augment = 2
        augment = StateAugmentation(num_augment=2, augment_fn="symmetric")
    elif mode == "symmetric8":
        num_augment = 8
        augment = StateAugmentation(num_augment=8, augment_fn="symmetric")
    elif mode == "symmetric16":
        num_augment = 16
        augment = StateAugmentation(num_augment=16, augment_fn="symmetric")
    elif mode == "dihedral8":
        num_augment = 8
        augment = StateAugmentation(num_augment=8, augment_fn="dihedral8")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    seed_everything(seed)
    model.num_starts = 50
    model.num_augment = num_augment
    model.augment = augment
    model.set_decode_type_multistart("test")
    model.eval()
    model.setup(stage="test")

    costs: list[float] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in model.test_dataloader():
            batch = batch.to(device)
            td = model.env.reset(batch).to(device)
            if augment is not None:
                td = augment(td)
            out = model.policy(
                td,
                model.env,
                phase="test",
                num_starts=50,
                return_actions=False,
            )
            rewards = unbatchify(out["reward"], (num_augment, 50))
            best = rewards.amax(dim=(-1, -2))
            costs.extend((-best).detach().cpu().double().tolist())
    return np.asarray(costs, dtype=np.float64), time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument(
        "--modes", default="none,symmetric2,symmetric8,dihedral8"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    observed_test_sha = sha256(TEST_FILE)
    if observed_test_sha != EXPECTED_TEST_SHA256:
        raise RuntimeError(f"CVRP50 test set SHA mismatch: {observed_test_sha}")
    metadata = validate_checkpoint(checkpoint)
    model, resolved_test_file = load_model(checkpoint)
    model.data_cfg["test_data_size"] = int(args.num_instances)
    model.data_cfg["test_batch_size"] = int(args.test_batch_size)
    device = torch.device(args.device)
    model = model.to(device)

    result = {
        "label": args.label,
        "checkpoint": metadata,
        "protocol": {
            "test_file": resolved_test_file,
            "test_file_sha256": observed_test_sha,
            "requested_num_instances": int(args.num_instances),
            "test_batch_size": int(args.test_batch_size),
            "num_starts": 50,
            "seed": int(args.seed),
        },
        "modes": {},
    }
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    for mode in modes:
        print(f"[eval] {args.label} {mode}", flush=True)
        costs, elapsed = evaluate_mode(
            model, mode=mode, seed=int(args.seed), device=device
        )
        npy_path = output.with_name(f"{output.stem}_{mode}_costs.npy")
        np.save(npy_path, costs)
        result["modes"][mode] = {
            "mean_cost": float(costs.mean()),
            "std_cost": float(costs.std(ddof=1)),
            "num_instances": int(costs.size),
            "elapsed_sec": float(elapsed),
            "per_instance_npy": str(npy_path),
        }
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"[done] {args.label} {mode} mean={costs.mean():.9f} "
            f"n={costs.size} sec={elapsed:.1f}",
            flush=True,
        )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
