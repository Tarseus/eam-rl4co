import argparse
import subprocess
import sys
from pathlib import Path


VARIANT_TO_IMPROVE_MODE = {
    "resample": "resample",
    "random_only": "random_only",
    "ls_only": "ls_only",
    "eam": "eam",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four mechanism variants sequentially for a Hydra experiment."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Hydra experiment config, e.g. routing/cvrp100_pomo_mechanism",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["resample", "random_only", "ls_only", "eam"],
        choices=sorted(VARIANT_TO_IMPROVE_MODE.keys()),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides appended verbatim after '--'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    extra_overrides = [item for item in args.overrides if item != "--"]

    for variant in args.variants:
        improve_mode = VARIANT_TO_IMPROVE_MODE[variant]
        command = [
            sys.executable,
            "run.py",
            f"experiment={args.experiment}",
            f"model.mechanism.variant={variant}",
            f"model.ea_kwargs.improve_mode={improve_mode}",
            f"model.ea_kwargs.val_improve_mode={improve_mode}",
        ]
        command.extend(extra_overrides)
        print(" ".join(command))
        if not args.dry_run:
            completed = subprocess.run(command, cwd=repo_root)
            if completed.returncode != 0:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
