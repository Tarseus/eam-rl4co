from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_cvrp_instances(
    *,
    num_instances: int,
    num_loc: int,
    seed: int,
    capacity: int,
) -> dict[str, np.ndarray]:
    if num_instances < 1 or num_loc < 2 or capacity < 1:
        raise ValueError("num_instances, num_loc, and capacity must be positive")
    rng = np.random.default_rng(int(seed))
    coordinates = rng.random((num_instances, num_loc + 1, 2), dtype=np.float32)
    demand = rng.integers(1, 10, size=(num_instances, num_loc), dtype=np.int64)
    return {
        "depot": coordinates[:, 0],
        "locs": coordinates[:, 1:],
        "demand": demand,
        "capacity": np.full((num_instances,), capacity, dtype=np.float32),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-instances", type=int, default=100)
    parser.add_argument("--num-loc", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--capacity", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "vrp" / "vrp1000_test_seed1234.npz",
    )
    args = parser.parse_args()

    output = args.output.expanduser()
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = generate_cvrp_instances(
        num_instances=args.num_instances,
        num_loc=args.num_loc,
        seed=args.seed,
        capacity=args.capacity,
    )
    np.savez_compressed(output, **arrays)
    metadata = {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "num_instances": int(args.num_instances),
        "num_loc": int(args.num_loc),
        "seed": int(args.seed),
        "capacity": int(args.capacity),
        "demand_min": int(arrays["demand"].min()),
        "demand_max": int(arrays["demand"].max()),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
