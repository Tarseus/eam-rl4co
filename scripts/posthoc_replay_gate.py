#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _rec_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rec.get("generation"),
        rec.get("pair_index"),
        rec.get("g_id"),
        rec.get("f_id"),
    )


def _builder_family_from_ir(ir: dict[str, Any] | None) -> str:
    hyper = (ir or {}).get("hyperparams", {}) or {}
    return " | ".join(
        [
            str(hyper.get("geometry_family", "?")),
            str(hyper.get("cap_family", "?")),
            str(hyper.get("weight_family", "?")),
            str(hyper.get("constraint_family", "?")),
        ]
    )


def _loss_family_from_ir(ir: dict[str, Any] | None) -> str:
    hyper = (ir or {}).get("hyperparams", {}) or {}
    return " | ".join(
        [
            str(hyper.get("paradigm_family", "?")),
            str(hyper.get("signal_family", "?")),
            str(hyper.get("link_family", "?")),
            str(hyper.get("agg_family", "?")),
            str(hyper.get("constraint_family", "?")),
        ]
    )


def _entry_to_family(entry: dict[str, Any], kind: str) -> str:
    ir = entry.get("ir") if isinstance(entry, dict) else None
    if kind == "builder":
        return _builder_family_from_ir(ir)
    return _loss_family_from_ir(ir)


def _family_counts(entries: list[dict[str, Any]], kind: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for entry in entries:
        counter[_entry_to_family(entry, kind)] += 1
    return counter


def _format_top_counter(counter: Counter[str], top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family, count in counter.most_common(top_k):
        out.append({"family": family, "count": int(count)})
    return out


def _get_nested(dct: dict[str, Any] | None, *path: str) -> Any:
    cur: Any = dct
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _score_value(rec: dict[str, Any], metric_mode: str) -> float | None:
    final_score = rec.get("final_score")
    if _is_finite_number(final_score):
        return float(final_score)
    stage = str(rec.get("stage", ""))
    stage_final = str(rec.get("stage_final", ""))
    if stage == "high_fidelity" or stage_final == "high_fidelity":
        score = rec.get("score")
        if _is_finite_number(score):
            return float(score)
    return None


def _sort_best(records: list[dict[str, Any]], metric_mode: str) -> list[dict[str, Any]]:
    reverse = str(metric_mode).lower() == "maximize"
    return sorted(records, key=lambda rec: float(_score_value(rec, metric_mode)), reverse=reverse)


def _best_record(records: list[dict[str, Any]], metric_mode: str) -> dict[str, Any] | None:
    scored = [rec for rec in records if _score_value(rec, metric_mode) is not None]
    if not scored:
        return None
    return _sort_best(scored, metric_mode)[0]


def _record_family_summary(
    rec: dict[str, Any],
    pair_meta: dict[tuple[Any, ...], dict[str, Any]],
    pair_meta_by_ids: dict[tuple[Any, ...], dict[str, Any]],
    metric_mode: str,
) -> dict[str, Any]:
    meta = pair_meta.get(_rec_key(rec), {})
    if not meta:
        meta = pair_meta_by_ids.get((rec.get("g_id"), rec.get("f_id")), {})
    return {
        "generation": rec.get("generation"),
        "pair_index": rec.get("pair_index"),
        "g_id": rec.get("g_id"),
        "f_id": rec.get("f_id"),
        "score": _score_value(rec, metric_mode),
        "pair_reason": rec.get("pair_reason"),
        "builder_family": _builder_family_from_ir(meta.get("g_ir")),
        "loss_family": _loss_family_from_ir(meta.get("f_ir")),
    }


def _fails_threshold(value: Any, threshold: float, direction: str) -> bool:
    if not _is_finite_number(value):
        return True
    numeric = float(value)
    if direction == "min":
        return numeric + 1e-12 < threshold
    return numeric - 1e-12 > threshold


def _passes_replay(rec: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if not args.all_records and not bool(rec.get("pair_ok", False)):
        reasons.append("pair_ok=false")

    if args.require_builder_gate_ok and not bool(rec.get("builder_gate_ok", False)):
        reasons.append("builder_gate_ok=false")

    if args.require_joint_gate_ok and not bool(rec.get("joint_gate_ok", False)):
        reasons.append("joint_gate_ok=false")

    if args.require_co_ok:
        co_ok = rec.get("co_ok")
        if co_ok is not True:
            reasons.append("co_ok!=true")

    if args.require_high_fidelity and str(rec.get("stage_final")) != "high_fidelity":
        reasons.append("stage_final!=high_fidelity")

    builder_semantic = _get_nested(rec, "builder_gate_trace", "checks")
    if args.min_builder_semantic_pass_rate is not None:
        observed = None
        if isinstance(builder_semantic, list):
            for item in builder_semantic:
                if isinstance(item, dict) and item.get("metric_name") == "semantic_pass_rate":
                    observed = item.get("observed_value")
                    break
        if _fails_threshold(observed, args.min_builder_semantic_pass_rate, "min"):
            reasons.append("builder_semantic_pass_rate")

    if args.min_builder_instance_weight_cv_pass_rate is not None:
        observed = _get_nested(rec, "builder_gate_trace", "instance_weight_cv_gate", "observed_cv_pass_rate")
        if observed is None:
            observed_values = _get_nested(rec, "builder_gate_trace", "instance_weight_cv_gate", "observed_cv_values")
            cv_threshold = _get_nested(rec, "builder_gate_trace", "instance_weight_cv_gate", "cv_threshold")
            if isinstance(observed_values, list) and _is_finite_number(cv_threshold):
                thresh = float(cv_threshold)
                total = len(observed_values)
                passed = sum(float(v) + 1e-12 >= thresh for v in observed_values if _is_finite_number(v))
                observed = (passed / total) if total > 0 else None
        if _fails_threshold(observed, args.min_builder_instance_weight_cv_pass_rate, "min"):
            reasons.append("builder_instance_weight_cv_pass_rate")

    joint_observed = _get_nested(rec, "joint_gate_trace", "observed") or {}
    if args.min_grad_w_pass_rate is not None and _fails_threshold(
        joint_observed.get("grad_w_pass_rate"), args.min_grad_w_pass_rate, "min"
    ):
        reasons.append("grad_w_pass_rate")
    if args.min_grad_l_pass_rate is not None and _fails_threshold(
        joint_observed.get("grad_l_pass_rate"), args.min_grad_l_pass_rate, "min"
    ):
        reasons.append("grad_l_pass_rate")
    if args.min_effective_grad_ratio is not None and _fails_threshold(
        joint_observed.get("effective_grad_ratio"), args.min_effective_grad_ratio, "min"
    ):
        reasons.append("effective_grad_ratio")

    if args.min_co_sensitivity_rel_delta is not None and _fails_threshold(
        rec.get("co_sensitivity_visible_rel_delta"), args.min_co_sensitivity_rel_delta, "min"
    ):
        reasons.append("co_sensitivity_visible_rel_delta")

    if args.max_co_invariance_rel_delta is not None and _fails_threshold(
        rec.get("co_invariance_visible_rel_delta"), args.max_co_invariance_rel_delta, "max"
    ):
        reasons.append("co_invariance_visible_rel_delta")

    return (len(reasons) == 0, reasons)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay stricter gate rules on an existing pref-loss discovery run."
    )
    p.add_argument("run_dir", type=Path, help="Run directory containing gate_reports.jsonl and checkpoint.json")
    p.add_argument("--all-records", action="store_true", help="Start from all gate_reports rows, not only pair_ok=true rows")
    p.add_argument("--require-builder-gate-ok", action="store_true", help="Require builder_gate_ok=true")
    p.add_argument("--require-joint-gate-ok", action="store_true", help="Require joint_gate_ok=true")
    p.add_argument("--require-co-ok", action="store_true", help="Require co_ok=true")
    p.add_argument("--require-high-fidelity", action="store_true", help="Require stage_final=high_fidelity")
    p.add_argument("--min-grad-w-pass-rate", type=float)
    p.add_argument("--min-grad-l-pass-rate", type=float)
    p.add_argument("--min-effective-grad-ratio", type=float)
    p.add_argument("--min-builder-semantic-pass-rate", type=float)
    p.add_argument("--min-builder-instance-weight-cv-pass-rate", type=float)
    p.add_argument("--min-co-sensitivity-rel-delta", type=float)
    p.add_argument("--max-co-invariance-rel-delta", type=float)
    p.add_argument("--top-k", type=int, default=8, help="How many top families to print")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    run_dir = args.run_dir.resolve()

    gate_path = run_dir / "gate_reports.jsonl"
    pair_path = run_dir / "pairs.jsonl"
    ckpt_path = run_dir / "checkpoint.json"
    summary_path = run_dir / "summary.json"

    for path in (gate_path, pair_path, ckpt_path):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    gate_rows = _load_jsonl(gate_path)
    pair_rows = _load_jsonl(pair_path)
    ckpt = _load_json(ckpt_path)
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    metric_mode = str(summary.get("metric_mode", ckpt.get("metric_mode", "minimize"))).lower()

    pair_meta = {_rec_key(rec): rec for rec in pair_rows}
    pair_meta_by_ids = {(rec.get("g_id"), rec.get("f_id")): rec for rec in pair_rows}

    original_pair_ok = [rec for rec in gate_rows if bool(rec.get("pair_ok", False))]
    original_scored = [rec for rec in original_pair_ok if _score_value(rec, metric_mode) is not None]
    replay_survivors: list[dict[str, Any]] = []
    replay_fail_reasons: Counter[str] = Counter()

    for rec in gate_rows:
        ok, reasons = _passes_replay(rec, args)
        if ok:
            replay_survivors.append(rec)
        else:
            replay_fail_reasons.update(reasons)

    replay_scored = [rec for rec in replay_survivors if _score_value(rec, metric_mode) is not None]
    original_best = _best_record(original_scored, metric_mode)
    replay_best = _best_record(replay_scored, metric_mode)

    by_generation: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "original_pair_ok": 0, "replay_survivors": 0, "replay_scored": 0})
    for rec in gate_rows:
        gen = int(rec.get("generation", -999999))
        by_generation[gen]["total"] += 1
        if bool(rec.get("pair_ok", False)):
            by_generation[gen]["original_pair_ok"] += 1
    for rec in replay_survivors:
        gen = int(rec.get("generation", -999999))
        by_generation[gen]["replay_survivors"] += 1
    for rec in replay_scored:
        gen = int(rec.get("generation", -999999))
        by_generation[gen]["replay_scored"] += 1

    replay_builder_family_counts: Counter[str] = Counter()
    replay_loss_family_counts: Counter[str] = Counter()
    replay_pair_family_counts: Counter[str] = Counter()
    replay_gen_winners: list[dict[str, Any]] = []

    for rec in replay_scored:
        meta = pair_meta.get(_rec_key(rec), {})
        if not meta:
            meta = pair_meta_by_ids.get((rec.get("g_id"), rec.get("f_id")), {})
        b_family = _builder_family_from_ir(meta.get("g_ir"))
        l_family = _loss_family_from_ir(meta.get("f_ir"))
        replay_builder_family_counts[b_family] += 1
        replay_loss_family_counts[l_family] += 1
        replay_pair_family_counts[f"{b_family} || {l_family}"] += 1

    grouped_scored: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in replay_scored:
        grouped_scored[int(rec.get("generation", -999999))].append(rec)
    for gen in sorted(grouped_scored):
        winner = _best_record(grouped_scored[gen], metric_mode)
        if winner is not None:
            replay_gen_winners.append(_record_family_summary(winner, pair_meta, pair_meta_by_ids, metric_mode))

    output = {
        "run_dir": str(run_dir),
        "metric_mode": metric_mode,
        "filters": {
            "all_records": bool(args.all_records),
            "require_builder_gate_ok": bool(args.require_builder_gate_ok),
            "require_joint_gate_ok": bool(args.require_joint_gate_ok),
            "require_co_ok": bool(args.require_co_ok),
            "require_high_fidelity": bool(args.require_high_fidelity),
            "min_grad_w_pass_rate": args.min_grad_w_pass_rate,
            "min_grad_l_pass_rate": args.min_grad_l_pass_rate,
            "min_effective_grad_ratio": args.min_effective_grad_ratio,
            "min_builder_semantic_pass_rate": args.min_builder_semantic_pass_rate,
            "min_builder_instance_weight_cv_pass_rate": args.min_builder_instance_weight_cv_pass_rate,
            "min_co_sensitivity_rel_delta": args.min_co_sensitivity_rel_delta,
            "max_co_invariance_rel_delta": args.max_co_invariance_rel_delta,
        },
        "counts": {
            "gate_rows_total": len(gate_rows),
            "current_pair_ok": len(original_pair_ok),
            "current_pair_ok_scored": len(original_scored),
            "replay_survivors": len(replay_survivors),
            "replay_scored": len(replay_scored),
        },
        "original_best": _record_family_summary(original_best, pair_meta, pair_meta_by_ids, metric_mode) if original_best else None,
        "replay_best": _record_family_summary(replay_best, pair_meta, pair_meta_by_ids, metric_mode) if replay_best else None,
        "replay_removed_reason_counts": dict(replay_fail_reasons.most_common()),
        "per_generation": [
            {"generation": gen, **stats}
            for gen, stats in sorted(by_generation.items(), key=lambda item: item[0])
        ],
        "original_resident_families": {
            "resident_g": _format_top_counter(_family_counts(ckpt.get("resident_pop_g", []) or [], "builder"), args.top_k),
            "resident_f": _format_top_counter(_family_counts(ckpt.get("resident_pop_f", []) or [], "loss"), args.top_k),
            "elites_g": _format_top_counter(_family_counts(ckpt.get("elites_g", []) or [], "builder"), args.top_k),
            "elites_f": _format_top_counter(_family_counts(ckpt.get("elites_f", []) or [], "loss"), args.top_k),
        },
        "replay_survivor_families": {
            "builder": _format_top_counter(replay_builder_family_counts, args.top_k),
            "loss": _format_top_counter(replay_loss_family_counts, args.top_k),
            "pair": _format_top_counter(replay_pair_family_counts, args.top_k),
        },
        "replay_generation_winners": replay_gen_winners,
        "note": (
            "This is a post-hoc replay on already logged candidates. It can show which evaluated pairs "
            "would survive stricter gates, but it cannot exactly reconstruct the search trajectory after gating."
        ),
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    print(f"run_dir: {output['run_dir']}")
    print(f"metric_mode: {metric_mode}")
    print(
        "counts: "
        f"total={output['counts']['gate_rows_total']} "
        f"current_pair_ok={output['counts']['current_pair_ok']} "
        f"current_scored={output['counts']['current_pair_ok_scored']} "
        f"replay_survivors={output['counts']['replay_survivors']} "
        f"replay_scored={output['counts']['replay_scored']}"
    )
    print()

    print("original_best:")
    print(json.dumps(output["original_best"], ensure_ascii=False, indent=2))
    print("replay_best:")
    print(json.dumps(output["replay_best"], ensure_ascii=False, indent=2))
    print()

    print("replay_removed_reason_counts:")
    print(json.dumps(output["replay_removed_reason_counts"], ensure_ascii=False, indent=2))
    print()

    print("original_resident_families:")
    print(json.dumps(output["original_resident_families"], ensure_ascii=False, indent=2))
    print("replay_survivor_families:")
    print(json.dumps(output["replay_survivor_families"], ensure_ascii=False, indent=2))
    print("replay_generation_winners:")
    print(json.dumps(output["replay_generation_winners"], ensure_ascii=False, indent=2))
    print()
    print(output["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
