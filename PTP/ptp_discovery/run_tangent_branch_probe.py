from __future__ import annotations

import argparse
from pathlib import Path

from .tangent_branch_probe import freeze_tangent_branch_probe_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--late-checkpoint", required=True)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--depths", default="1,25,50,75")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depths = tuple(
        int(value.strip())
        for value in str(args.depths).split(",")
        if value.strip()
    )
    destination = freeze_tangent_branch_probe_suite(
        args.config,
        args.output,
        late_checkpoint=str(args.late_checkpoint),
        instances=int(args.instances),
        branch_count=int(args.branches),
        depths=depths,
        device=str(args.device),
    )
    print(destination)


if __name__ == "__main__":
    main()
