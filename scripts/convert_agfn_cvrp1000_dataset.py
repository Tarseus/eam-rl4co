from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the official AGFN CVRP tensor to a compact RL4CO NPZ."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--expected-instances", type=int, default=128)
    parser.add_argument("--expected-locations", type=int, default=1000)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    tensor = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError(f"Expected a rank-3 tensor, got {type(tensor)} {getattr(tensor, 'shape', None)}")
    expected_shape = (
        args.expected_instances,
        args.expected_locations + 4,
        args.expected_locations + 1,
    )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected AGFN tensor shape {expected_shape}, got {tuple(tensor.shape)}")

    normalized_demand = tensor[:, 0, :]
    positions = tensor[:, 1:3, :].transpose(1, 2).contiguous()
    stored_distances = tensor[:, 3:, :]
    if not torch.allclose(normalized_demand[:, 0], torch.zeros_like(normalized_demand[:, 0])):
        raise ValueError("AGFN depot demand is not zero")
    integer_demand = torch.round(normalized_demand[:, 1:] * args.capacity)
    if not torch.allclose(
        normalized_demand[:, 1:] * args.capacity, integer_demand, atol=1e-8, rtol=0
    ):
        raise ValueError("Normalized AGFN demands do not recover exact integer demands")
    if int(integer_demand.min()) != 1 or int(integer_demand.max()) != 9:
        raise ValueError(
            f"Expected demands in [1, 9], got [{int(integer_demand.min())}, {int(integer_demand.max())}]"
        )

    # Validate every stored distance entry in bounded row chunks before discarding the matrix.
    max_distance_error = 0.0
    for index in range(args.expected_instances):
        recomputed = torch.cdist(positions[index], positions[index], p=2)
        recomputed.fill_diagonal_(1e-10)
        error = float((recomputed - stored_distances[index]).abs().max())
        max_distance_error = max(max_distance_error, error)
        if error > 1e-10:
            raise ValueError(f"Distance mismatch at instance {index}: max_abs_error={error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        depot=positions[:, 0].numpy().astype(np.float32),
        locs=positions[:, 1:].numpy().astype(np.float32),
        demand=integer_demand.numpy().astype(np.int64),
        capacity=np.full((args.expected_instances,), args.capacity, dtype=np.float32),
        source_index=np.arange(args.expected_instances, dtype=np.int64),
    )
    metadata = {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "num_instances": args.expected_instances,
        "num_loc": args.expected_locations,
        "capacity": args.capacity,
        "demand_min": int(integer_demand.min()),
        "demand_max": int(integer_demand.max()),
        "max_distance_error": max_distance_error,
        "source_indices": [0, args.expected_instances - 1],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
