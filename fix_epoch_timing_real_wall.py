import argparse
from pathlib import Path

import pandas as pd


def _load_real_wall(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"model", "problem", "wall_time_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[list(required)].dropna()
    df = df.rename(columns={"wall_time_s": "wall_time_real_s"})
    df["model"] = df["model"].astype(str).str.strip()
    df["problem"] = df["problem"].astype(str).str.strip()
    return df


def _model_alias(model: str) -> str:
    model = str(model).strip()
    aliases = {
        # sampled suite
        "AttentionModel": "AM",
        # keep existing values
        "AM": "AM",
        "POMO": "POMO",
        "SymNCO": "SymNCO",
        # tevc suite
        "EAM-AM": "EAM-AM",
        "EAM-POMO": "EAM-POMO",
        "EAM-SymNCO": "EAM-SymNCO",
    }
    return aliases.get(model, model)


def _fix_one_csv(csv_path: Path, real_wall: pd.DataFrame, *, overwrite_wall: bool) -> Path:
    df = pd.read_csv(csv_path)
    if "model" not in df.columns or "problem" not in df.columns:
        raise ValueError(f"{csv_path} missing 'model'/'problem' columns")

    df["model_realwall_key"] = df["model"].map(_model_alias)
    merged = df.merge(
        real_wall,
        left_on=["model_realwall_key", "problem"],
        right_on=["model", "problem"],
        how="left",
        suffixes=("", "_realwall"),
    )
    merged = merged.drop(columns=["model_realwall_key", "model_realwall"])

    missing = merged["wall_time_real_s"].isna()
    if missing.any():
        rows = merged.loc[missing, ["model", "problem"]].drop_duplicates()
        raise ValueError(
            f"{csv_path}: missing real wall time for {len(rows)} tasks: "
            + ", ".join(f"{r.model}/{r.problem}" for r in rows.itertuples(index=False))
        )

    if "wall_time_s" in merged.columns:
        merged["wall_time_profiled_s"] = merged["wall_time_s"]
        merged["profile_wall_over_real"] = (
            merged["wall_time_profiled_s"] / merged["wall_time_real_s"]
        )
        merged["wall_profiler_overhead_s"] = (
            merged["wall_time_profiled_s"] - merged["wall_time_real_s"]
        )
        if overwrite_wall:
            merged["wall_time_s"] = merged["wall_time_real_s"]

    if "event_gpu_time_s" in merged.columns:
        merged["event_gpu_over_wall_real"] = (
            merged["event_gpu_time_s"] / merged["wall_time_real_s"]
        )
        merged["event_gpu_over_wall_real_capped"] = merged[
            "event_gpu_over_wall_real"
        ].clip(lower=0.0, upper=1.0)
        merged["other_time_real_s"] = merged["wall_time_real_s"] - merged[
            "event_gpu_time_s"
        ]
        merged["other_time_real_s_capped"] = merged["other_time_real_s"].clip(lower=0.0)

    if "event_cpu_time_s" in merged.columns:
        merged["event_cpu_over_wall_real"] = (
            merged["event_cpu_time_s"] / merged["wall_time_real_s"]
        )

    out_path = csv_path.with_name(csv_path.stem + ".realwall_fixed.csv")
    merged.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge real wall-time baselines into epoch timing CSVs and add corrected ratios."
    )
    parser.add_argument(
        "--real-wall-xlsx",
        type=Path,
        default=Path("results") / "real_wall_time.xlsx",
        help="Excel file containing (model, problem, wall_time_s) baselines.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing epoch_timing_*.csv files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="epoch_timing_*.csv",
        help="Glob for timing CSVs to fix (default: epoch_timing_*.csv).",
    )
    parser.add_argument(
        "--overwrite-wall",
        action="store_true",
        default=False,
        help="Overwrite wall_time_s with wall_time_real_s in the output (keeps wall_time_profiled_s).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="realwall_fixed",
        help="Suffix for output filenames (default: realwall_fixed).",
    )
    args = parser.parse_args()

    real_wall = _load_real_wall(args.real_wall_xlsx)
    csv_paths = sorted(args.results_dir.glob(args.pattern))
    csv_paths = [p for p in csv_paths if ".realwall" not in p.stem]
    if not csv_paths:
        raise SystemExit(f"No CSVs found under {args.results_dir} matching {args.pattern}")

    outputs: list[Path] = []
    for csv_path in csv_paths:
        out_path = csv_path.with_name(csv_path.stem + f".{args.output_suffix}.csv")
        fixed_path = _fix_one_csv(csv_path, real_wall, overwrite_wall=args.overwrite_wall)
        # Rename to the requested suffix unless it already matches.
        if fixed_path != out_path:
            fixed_path.replace(out_path)
        outputs.append(out_path)

    print("Wrote:")
    for p in outputs:
        print(" -", p.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
