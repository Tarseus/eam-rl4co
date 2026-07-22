from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tsp-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--problem-size", type=int, required=True, choices=(50, 100))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--per-instance-output",
        type=Path,
        default=None,
        help="Optional CSV path for ordered per-instance costs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official_root = args.official_tsp_root.resolve()
    sys.path.insert(0, str(official_root))

    # TSPEnv imports the optional Gurobi helper at module import time even
    # though checkpoint evaluation never calls it.
    guribo_stub = types.ModuleType("utils.guribo")
    guribo_stub.guribo_tsp = lambda *_args, **_kwargs: None
    sys.modules.setdefault("utils.guribo", guribo_stub)

    from TSPEnv import TSPEnv  # noqa: PLC0415
    from TSPModel import TSPModel  # noqa: PLC0415
    from utils.TSProblemDef import augment_xy_data_by_8_fold  # noqa: PLC0415

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TSPModel(
        start_node="pomo",
        embedding_dim=128,
        sqrt_embedding_dim=128**0.5,
        encoder_layer_num=6,
        qkv_dim=16,
        head_num=8,
        logit_clipping=10,
        ff_hidden_dim=512,
        eval_type="argmax",
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()

    locs = np.load(args.data)["locs"]
    if args.num_instances is not None:
        locs = locs[: int(args.num_instances)]
    if locs.ndim != 3 or tuple(locs.shape[1:]) != (args.problem_size, 2):
        raise ValueError(f"Unexpected locs shape: {locs.shape}")

    torch.set_default_device(device)
    env = TSPEnv(problem_size=args.problem_size, B=args.problem_size)
    costs: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(locs), args.batch_size):
            base = torch.from_numpy(locs[offset : offset + args.batch_size]).to(device)
            batch_size = int(base.shape[0])
            augmented = augment_xy_data_by_8_fold(base)
            env.batch_size = int(augmented.shape[0])
            env.problems = augmented
            env.opts = None
            env.BATCH_IDX = torch.arange(env.batch_size)[:, None].expand(
                env.batch_size, env.sols_num
            )
            env.B_IDX = torch.arange(env.sols_num)[None, :].expand(
                env.batch_size, env.sols_num
            )
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)
            reward = reward.reshape(8, batch_size, env.sols_num)
            costs.append((-reward.amax(dim=(0, 2))).cpu())

    all_costs = torch.cat(costs)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "data": str(args.data.resolve()),
        "problem_size": int(args.problem_size),
        "num_instances": int(all_costs.numel()),
        "mean_cost": float(all_costs.double().mean()),
        "std_cost": float(all_costs.double().std(unbiased=True)),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    if args.per_instance_output is not None:
        per_instance_output = args.per_instance_output.resolve()
        per_instance_output.parent.mkdir(parents=True, exist_ok=True)
        with per_instance_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("problem", "method", "instance_id", "instance_index", "cost"),
            )
            writer.writeheader()
            for index, cost in enumerate(all_costs.double().tolist()):
                writer.writerow(
                    {
                        "problem": f"tsp{args.problem_size}",
                        "method": "BOPO-official",
                        "instance_id": f"tsp{args.problem_size}_{index:05d}",
                        "instance_index": index,
                        "cost": cost,
                    }
                )
        summary["per_instance_output"] = str(per_instance_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
