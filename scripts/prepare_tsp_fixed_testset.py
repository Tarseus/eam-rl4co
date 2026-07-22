from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_locations(locations: np.ndarray) -> str:
    canonical = np.ascontiguousarray(locations, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def generate_locations(*, size: int, count: int, seed: int) -> np.ndarray:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(size) * 1_000_003)
    return torch.rand(
        (int(count), int(size), 2),
        generator=generator,
        dtype=torch.float32,
    ).numpy()


def load_locations(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if "locs" not in payload:
            raise KeyError(f"Dataset has no 'locs' array: {path}")
        locations = np.asarray(payload["locs"], dtype=np.float32)
    if locations.ndim != 3 or locations.shape[-1] != 2:
        raise ValueError(f"Expected locs with shape [B, N, 2], got {locations.shape}")
    if not np.isfinite(locations).all():
        raise ValueError("Dataset contains non-finite coordinates")
    if locations.min() < 0.0 or locations.max() >= 1.0:
        raise ValueError("Expected coordinates in [0, 1)")
    return locations


def prepare_dataset(
    *,
    output: Path,
    size: int,
    count: int,
    seed: int,
    source: Path | None,
    protocol: str,
) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        source = source.resolve()
        locations = load_locations(source)
        if locations.shape != (count, size, 2):
            raise ValueError(
                f"Source shape {locations.shape} does not match {(count, size, 2)}"
            )
        if source != output:
            shutil.copyfile(source, output)
        generation = {"mode": "copied", "source": str(source)}
    else:
        locations = generate_locations(size=size, count=count, seed=seed)
        np.savez_compressed(output, locs=locations)
        generation = {
            "mode": "generated",
            "algorithm": "torch.Generator(cpu)+torch.rand(float32)",
            "effective_seed": int(seed) + int(size) * 1_000_003,
        }

    manifest = {
        "protocol": protocol,
        "dataset": str(output),
        "num_instances": int(locations.shape[0]),
        "num_nodes": int(locations.shape[1]),
        "coordinate_dtype": "float32",
        "coordinate_range": "[0, 1)",
        "seed": int(seed),
        "generation": generation,
        "file_sha256": _sha256_file(output),
        "locations_sha256": _sha256_locations(locations),
        "split": "test-only; do not use for training, validation, or checkpoint selection",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a content-addressed fixed Euclidean TSP test set."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "tsp" / "tsp1000_test_seed1234.npz",
    )
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--protocol", default="tsp1000_fixed_test_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 2 or args.count < 1:
        raise ValueError("size must be >= 2 and count must be >= 1")
    manifest = prepare_dataset(
        output=args.output,
        size=args.size,
        count=args.count,
        seed=args.seed,
        source=args.source,
        protocol=args.protocol,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
