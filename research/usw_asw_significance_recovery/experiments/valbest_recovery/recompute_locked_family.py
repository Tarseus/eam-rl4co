from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.analyze_paper_significance import compare, holm_adjust


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FAMILY_PATH = REPO / "paper_materials/statistical_analysis/final_results/paired_significance.json"
FFSP_CANDIDATE = RESULTS / "ffsp50_usw_epoch117/per_instance.csv"
FFSP_SUMMARY = RESULTS / "ffsp50_usw_epoch117/summary.json"
JSSP_CANDIDATE = RESULTS / "jssp15_usw_epoch013/generated.csv"
JSSP_SUMMARY = RESULTS / "jssp15_usw_epoch013/summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ffsp() -> pd.DataFrame:
    frames = [pd.read_csv(FFSP_CANDIDATE)]
    for relative in (
        "paper_materials/statistical_analysis/ffsp_eval_seeded/ffsp50_po_usw/per_instance.csv",
        "paper_materials/statistical_analysis/ffsp_eval_seeded/ffsp50_bopo_asw/per_instance.csv",
    ):
        frame = pd.read_csv(REPO / relative)
        frames.append(frame[frame["method"] != "USW"])
    return pd.concat(frames, ignore_index=True)[
        ["problem", "method", "instance_id", "cost"]
    ]


def load_jssp() -> pd.DataFrame:
    paths = {
        "USW": JSSP_CANDIDATE,
        "PO4COPs": REPO / "logs/eval_jssp_generated/jssp15x15_po_seed12345678_n100_fixedparser/generated.csv",
        "SLL": REPO / "logs/eval_jssp_generated/jssp15x15_sll_seed12345678_n100_fixedparser/generated.csv",
        "BOPO": REPO / "logs/eval_jssp_generated/jssp15x15_bopo_seed12345678_n100_fixedparser/generated.csv",
        "ASW": REPO / "logs/eval_jssp_generated/jssp15x15_weighting_seed12345678_n100_fixedparser/generated.csv",
    }
    frames = []
    for method, path in paths.items():
        source = pd.read_csv(path)
        frames.append(
            pd.DataFrame(
                {
                    "problem": "jssp15x15",
                    "method": method,
                    "instance_id": source["instance"].astype(str),
                    "cost": source["pred_makespan"].astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def assert_alignment(frame: pd.DataFrame, problem: str, methods: list[str], expected: int) -> list[str]:
    ordered = {}
    for method in methods:
        rows = frame[(frame["problem"] == problem) & (frame["method"] == method)]
        if len(rows) != expected or rows["instance_id"].duplicated().any():
            raise ValueError(f"{problem}/{method} does not contain {expected} unique instances")
        ordered[method] = rows["instance_id"].astype(str).tolist()
    reference = ordered[methods[0]]
    for method in methods[1:]:
        if ordered[method] != reference:
            raise ValueError(f"ordered instance mismatch: {problem}/{methods[0]} vs {method}")
    return reference


def main() -> int:
    family = json.loads(FAMILY_PATH.read_text(encoding="utf-8"))
    comparisons = family["comparisons"]
    if len(comparisons) != 52:
        raise ValueError(f"expected locked 52-comparison family, found {len(comparisons)}")

    ffsp = load_ffsp()
    jssp = load_jssp()
    ffsp_ids = assert_alignment(ffsp, "ffsp50", ["USW", "PO4COPs", "BOPO", "ASW"], 1000)
    jssp_ids = assert_alignment(jssp, "jssp15x15", ["USW", "PO4COPs", "SLL", "BOPO", "ASW"], 100)

    targets = {
        ("ffsp50", "USW", "PO4COPs"): ffsp,
        ("ffsp50", "USW", "BOPO"): ffsp,
        ("ffsp50", "ASW", "USW"): ffsp,
        ("jssp15x15", "USW", "PO4COPs"): jssp,
        ("jssp15x15", "USW", "SLL"): jssp,
        ("jssp15x15", "USW", "BOPO"): jssp,
        ("jssp15x15", "ASW", "USW"): jssp,
    }
    replaced = []
    for index, original in enumerate(comparisons):
        key = (original["problem"], original["left_method"], original["right_method"])
        if key not in targets:
            continue
        row = compare(
            targets[key],
            problem=key[0],
            left=key[1],
            right=key[2],
            seed=20260716 + index,
            bootstrap_samples=20000,
        )
        comparisons[index] = row
        replaced.append(index)
    if len(replaced) != len(targets):
        raise ValueError(f"replaced {len(replaced)} comparisons, expected {len(targets)}")

    adjusted = holm_adjust([float(row["wilcoxon_pvalue"]) for row in comparisons])
    for row, holm_p in zip(comparisons, adjusted, strict=True):
        row["wilcoxon_holm_pvalue"] = holm_p
        row["holm_outcome"] = (
            "better"
            if row["left_is_better"] and holm_p < 0.05
            else "worse"
            if not row["left_is_better"] and holm_p < 0.05
            else "not_significant"
        )

    ffsp_summary = json.loads(FFSP_SUMMARY.read_text(encoding="utf-8"))["evaluations"][0]
    jssp_summary = json.loads(JSSP_SUMMARY.read_text(encoding="utf-8"))
    target_rows = [comparisons[index] for index in replaced]

    def selected(problem: str, left: str, right: str) -> dict:
        return next(
            row
            for row in target_rows
            if (row["problem"], row["left_method"], row["right_method"])
            == (problem, left, right)
        )

    ffsp_po = selected("ffsp50", "USW", "PO4COPs")
    ffsp_bopo = selected("ffsp50", "USW", "BOPO")
    jssp_bopo = selected("jssp15x15", "USW", "BOPO")
    payload = {
        "protocol": {
            "primary_test": "two-sided paired Wilcoxon signed-rank",
            "correction": "Holm over the unchanged 52-comparison paper family",
            "family_size": len(comparisons),
            "bootstrap_samples": 20000,
            "bootstrap_seed": 20260716,
            "source_family": str(FAMILY_PATH.relative_to(REPO)).replace("\\", "/"),
            "source_family_sha256": sha256(FAMILY_PATH),
            "paper_registry_replaced": False,
        },
        "audit": {
            "ffsp50": {
                "ordered_instances_aligned": True,
                "instance_count": len(ffsp_ids),
                "checkpoint_sha256": ffsp_summary["checkpoint_sha256"],
                "data_sha256": ffsp_summary["test_file_sha256"],
                "num_starts": ffsp_summary["num_starts"],
                "num_augment": ffsp_summary["num_augment"],
                "seed": ffsp_summary["seed"],
                "raw_csv_sha256": sha256(FFSP_CANDIDATE),
            },
            "jssp15x15": {
                "ordered_instances_aligned": True,
                "instance_count": len(jssp_ids),
                "checkpoint_sha256": "93b79af77a42156753882f5d37c4be1b1e9b814d9890852b666de278be1ac3ff",
                "ordered_dataset_sha256": "cf30cda4099e756540c5fe15bd6a47849a6b74a3325bedb0bb7066db40c9294c",
                "B": jssp_summary["B"],
                "greedy": jssp_summary["greedy"],
                "sampling_seed": jssp_summary["sampling_seed"],
                "augmentation": "none" if jssp_summary["aug_factor"] == 1 else jssp_summary["aug_factor"],
                "raw_csv_sha256": sha256(JSSP_CANDIDATE),
            },
        },
        "success": {
            "ffsp50_usw_epoch117": bool(
                ffsp_po["left_mean"] < ffsp_po["right_mean"]
                and ffsp_bopo["left_mean"] < ffsp_bopo["right_mean"]
                and ffsp_po["wilcoxon_holm_pvalue"] < 0.05
                and ffsp_bopo["wilcoxon_holm_pvalue"] < 0.05
            ),
            "jssp15x15_usw_epoch13": bool(
                jssp_bopo["left_mean"] < jssp_bopo["right_mean"]
                and jssp_bopo["wilcoxon_holm_pvalue"] < 0.05
            ),
        },
        "replacement_comparisons": target_rows,
        "raw_preview": {
            "ffsp50_instance_ids": ffsp_ids[:3] + ffsp_ids[-2:],
            "jssp15x15_instance_ids": jssp_ids[:3] + jssp_ids[-2:],
        },
    }
    output = RESULTS / "locked_candidates_full_holm.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["success"], sort_keys=True))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
