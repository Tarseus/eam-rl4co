from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _repo_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_import_path() -> None:
    repo_root = _repo_root_dir()
    ptp_root = repo_root / "PTP"
    for path in (repo_root, ptp_root):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


_configure_import_path()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return dict(payload)


def _pair_key(rec: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(rec.get("generation", -1) or -1),
        int(rec.get("pair_index", -1) or -1),
        str(rec.get("g_id") or ""),
        str(rec.get("f_id") or ""),
    )


def _dedupe_keep_last(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for rec in records:
        out[_pair_key(rec)] = dict(rec)
    return list(out.values())


def _load_pairs_index(run_dir: Path) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    pairs_path = run_dir / "pairs.jsonl"
    if not pairs_path.is_file():
        return {}
    out: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for rec in _iter_jsonl(pairs_path):
        out[_pair_key(rec)] = dict(rec)
    return out


def _hydrate_candidate_irs(
    rows: Sequence[Mapping[str, Any]],
    *,
    pairs_index: Mapping[Tuple[Any, ...], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for rec in rows:
        merged = dict(rec)
        cached = pairs_index.get(_pair_key(rec))
        if isinstance(cached, Mapping):
            if not isinstance(merged.get("g_ir"), dict) and isinstance(cached.get("g_ir"), dict):
                merged["g_ir"] = dict(cached.get("g_ir") or {})
            if not isinstance(merged.get("f_ir"), dict) and isinstance(cached.get("f_ir"), dict):
                merged["f_ir"] = dict(cached.get("f_ir") or {})
        hydrated.append(merged)
    return hydrated


def _build_no_gate_cfg(base_cfg: Mapping[str, Any], *, pure_no_gate: bool) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    eval_stages = dict(cfg.get("eval_stages", {}) or {})
    eval_stages["stage0_gate"] = False
    cfg["eval_stages"] = eval_stages
    cfg["cheap_gate_on"] = False
    cfg["stage0_sandbox_gate_enabled"] = False
    cfg["stage0_sandbox_gate_hard_block_hf"] = False
    if pure_no_gate:
        cfg["joint_gate_repair_enabled"] = False
        cfg["builder_gate_repair_enabled"] = False
        proxy_gate_repair = dict(cfg.get("proxy_gate_repair", {}) or {})
        proxy_gate_repair["enabled"] = False
        cfg["proxy_gate_repair"] = proxy_gate_repair
    return cfg


def _resolve_run_dir(run_dir_raw: str) -> Path:
    run_dir = Path(run_dir_raw)
    if not run_dir.is_absolute():
        run_dir = (_repo_root_dir() / run_dir).resolve()
    return run_dir


def _select_candidates(
    *,
    run_dir: Path,
    pair_reasons: Sequence[str],
    generation: str | None,
    max_pairs: int | None,
    sample_size: int | None,
    sample_seed: int,
    min_per_generation: int,
) -> List[Dict[str, Any]]:
    gate_path = run_dir / "gate_reports.jsonl"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Missing gate_reports.jsonl: {gate_path}")

    rows = list(_iter_jsonl(gate_path))
    wanted = {str(v).strip() for v in pair_reasons if str(v).strip()}
    rows = [rec for rec in rows if str(rec.get("pair_reason") or "") in wanted]

    if generation is not None:
        token = str(generation).strip().lower()
        if token == "latest":
            gens = [int(rec.get("generation", -1) or -1) for rec in rows]
            if gens:
                target = max(gens)
                rows = [rec for rec in rows if int(rec.get("generation", -1) or -1) == int(target)]
        else:
            target = int(token)
            rows = [rec for rec in rows if int(rec.get("generation", -1) or -1) == int(target)]

    rows = _dedupe_keep_last(rows)
    if sample_size is not None and sample_size > 0 and len(rows) > int(sample_size):
        rng = random.Random(int(sample_seed))
        if int(min_per_generation) > 0:
            grouped: Dict[int, List[Dict[str, Any]]] = {}
            for rec in rows:
                gen = int(rec.get("generation", -1) or -1)
                grouped.setdefault(gen, []).append(rec)

            selected_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
            for gen in sorted(grouped):
                group = list(grouped[gen])
                k = min(len(group), int(min_per_generation))
                if k > 0:
                    for picked in rng.sample(group, k):
                        selected_map[_pair_key(picked)] = picked

            min_required = len(selected_map)
            target_size = max(int(sample_size), int(min_required))
            target_size = min(target_size, len(rows))
            remaining = [rec for rec in rows if _pair_key(rec) not in selected_map]
            need_more = max(0, target_size - len(selected_map))
            if need_more > 0 and remaining:
                for picked in rng.sample(remaining, min(need_more, len(remaining))):
                    selected_map[_pair_key(picked)] = picked
            rows = list(selected_map.values())
        else:
            rows = rng.sample(rows, int(sample_size))
    rows = sorted(rows, key=lambda rec: (int(rec.get("generation", -1) or -1), int(rec.get("pair_index", -1) or -1)))
    if max_pairs is not None:
        rows = rows[: int(max_pairs)]
    return rows


def _apply_shard(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    shard_count = max(1, int(num_shards))
    shard_id = int(shard_index)
    if shard_id < 0 or shard_id >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_id}")
    if shard_count == 1:
        return [dict(rec) for rec in rows]
    out: List[Dict[str, Any]] = []
    for idx, rec in enumerate(rows):
        if idx % shard_count == shard_id:
            out.append(dict(rec))
    return out


def _worker_main(args: argparse.Namespace) -> int:
    from PTP.ptp_discovery.pref_loss_coevo_loop import _evaluate_pair_worker  # noqa: E402

    payload_path = Path(args.payload).resolve()
    payload = _load_json(payload_path)
    result = _evaluate_pair_worker(payload)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def _coordinator_main(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args.run_dir)
    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint.json: {checkpoint_path}")

    base_cfg = _load_json(checkpoint_path)
    replay_cfg = _build_no_gate_cfg(base_cfg, pure_no_gate=bool(args.pure_no_gate))
    all_candidates = _select_candidates(
        run_dir=run_dir,
        pair_reasons=[v.strip() for v in str(args.pair_reasons).split(",") if v.strip()],
        generation=args.generation,
        max_pairs=args.max_pairs,
        sample_size=args.sample_size,
        sample_seed=int(args.sample_seed),
        min_per_generation=int(args.min_per_generation),
    )
    if not all_candidates:
        raise SystemExit("No matching gate-rejected pairs found.")
    pairs_index = _load_pairs_index(run_dir)
    all_candidates = _hydrate_candidate_irs(all_candidates, pairs_index=pairs_index)
    candidates = _apply_shard(
        all_candidates,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (run_dir / f"replay_gate_rejected_{str(args.device).replace(':', '_')}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "replay_gate_rejected_payloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "replay_gate_rejected_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    python_exe = str(Path(args.python or sys.executable).resolve())
    script_path = Path(__file__).resolve()
    repo_root = _repo_root_dir()
    env = dict(os.environ)
    py_paths = [str(repo_root), str(repo_root / "PTP")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    selected_summary = [
        {
            "generation": int(rec.get("generation", -1) or -1),
            "pair_index": int(rec.get("pair_index", -1) or -1),
            "g_id": str(rec.get("g_id") or ""),
            "f_id": str(rec.get("f_id") or ""),
            "pair_reason": str(rec.get("pair_reason") or ""),
            "joint_gate_reason": rec.get("joint_gate_reason"),
            "builder_gate_reason": rec.get("builder_gate_reason"),
        }
        for rec in candidates
    ]

    if not candidates:
        empty_payload = {
            "run_dir": str(run_dir),
            "device": str(args.device),
            "pure_no_gate": bool(args.pure_no_gate),
            "pair_reasons": [v.strip() for v in str(args.pair_reasons).split(",") if v.strip()],
            "selected_count_before_shard": len(all_candidates),
            "selected_count": 0,
            "sample_size": (int(args.sample_size) if args.sample_size is not None else None),
            "sample_seed": int(args.sample_seed),
            "min_per_generation": int(args.min_per_generation),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "selected_pairs": [],
            "results": [],
            "summary": {"selected": 0, "hf_ok": 0, "non_hf_or_failed": 0, "best_score": None},
        }
        if bool(args.dry_run):
            print(json.dumps(empty_payload, ensure_ascii=False, indent=2))
            return 0
        output_path.write_text(json.dumps(empty_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(empty_payload["summary"], ensure_ascii=False, indent=2), flush=True)
        print(f"report_path={output_path}", flush=True)
        return 0

    if bool(args.dry_run):
        dry_payload = {
            "run_dir": str(run_dir),
            "device": str(args.device),
            "pure_no_gate": bool(args.pure_no_gate),
            "pair_reasons": [v.strip() for v in str(args.pair_reasons).split(",") if v.strip()],
            "selected_count_before_shard": len(all_candidates),
            "selected_count": len(selected_summary),
            "sample_size": (int(args.sample_size) if args.sample_size is not None else None),
            "sample_seed": int(args.sample_seed),
            "min_per_generation": int(args.min_per_generation),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "selected_pairs": selected_summary,
        }
        print(json.dumps(dry_payload, ensure_ascii=False, indent=2))
        return 0

    results: List[Dict[str, Any]] = []
    for idx, rec in enumerate(candidates, start=1):
        payload = {
            "generation": int(rec.get("generation", -1) or -1),
            "pair_index": int(rec.get("pair_index", -1) or -1),
            "g_entry": {"id": str(rec.get("g_id") or ""), "ir": dict(rec.get("g_ir") or {})},
            "f_entry": {"id": str(rec.get("f_id") or ""), "ir": dict(rec.get("f_ir") or {})},
            "cfg_yaml": replay_cfg,
            "device_str": str(args.device),
            "operator_whitelist": [],
            "run_dir": str(run_dir),
            "cheap_gate_on": False,
            "high_fidelity_on": True,
            "eval_budget_signature": str(rec.get("eval_budget_signature") or "posthoc_no_gate"),
            "proxy_record": None,
            "baseline_epoch_objectives": None,
            "baseline_early_valid": None,
            "early_eval_steps": int(replay_cfg.get("early_eval_steps", 0) or 0),
        }

        payload_path = tmp_dir / f"{idx:03d}_{payload['generation']:03d}_{payload['pair_index']:03d}_{payload['g_entry']['id']}_{payload['f_entry']['id']}.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            python_exe,
            str(script_path),
            "--worker",
            "--payload",
            str(payload_path),
        ]
        started = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            env=env,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        log_path = log_dir / f"{idx:03d}_{payload['generation']:03d}_{payload['pair_index']:03d}_{payload['g_entry']['id']}_{payload['f_entry']['id']}.log"
        log_path.write_text(
            "\n".join(
                [
                    f"command: {' '.join(cmd)}",
                    f"returncode: {proc.returncode}",
                    "",
                    "[stdout]",
                    stdout,
                    "",
                    "[stderr]",
                    stderr,
                    "",
                ]
            ),
            encoding="utf-8",
        )

        if stdout:
            try:
                result = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError:
                result = {
                    "generation": payload["generation"],
                    "pair_index": payload["pair_index"],
                    "g_id": payload["g_entry"]["id"],
                    "f_id": payload["f_entry"]["id"],
                    "pair_ok": False,
                    "pair_reason": "worker_output_parse_error",
                    "score": float("inf"),
                    "error": stdout[-4000:],
                }
        else:
            result = {
                "generation": payload["generation"],
                "pair_index": payload["pair_index"],
                "g_id": payload["g_entry"]["id"],
                "f_id": payload["f_entry"]["id"],
                "pair_ok": False,
                "pair_reason": "worker_no_output",
                "score": float("inf"),
                "error": stderr[-4000:],
            }

        result["worker_returncode"] = int(proc.returncode)
        result["worker_elapsed_s"] = float(time.time() - started)
        result["worker_log"] = os.path.relpath(str(log_path), start=str(run_dir))
        results.append(result)

        checkpoint_payload = {
            "run_dir": str(run_dir),
            "device": str(args.device),
            "pure_no_gate": bool(args.pure_no_gate),
            "pair_reasons": [v.strip() for v in str(args.pair_reasons).split(",") if v.strip()],
            "selected_count_before_shard": len(all_candidates),
            "selected_pairs": selected_summary,
            "sample_size": (int(args.sample_size) if args.sample_size is not None else None),
            "sample_seed": int(args.sample_seed),
            "min_per_generation": int(args.min_per_generation),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "results": results,
        }
        output_path.write_text(json.dumps(checkpoint_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{idx}/{len(candidates)}] gen={payload['generation']} pair={payload['pair_index']} "
            f"g={payload['g_entry']['id']} f={payload['f_entry']['id']} "
            f"pair_ok={bool(result.get('pair_ok'))} stage={result.get('stage')} reason={result.get('pair_reason')}",
            flush=True,
        )

    summary = {
        "selected": len(selected_summary),
        "hf_ok": sum(1 for rec in results if bool(rec.get("pair_ok")) and str(rec.get("stage")) == "high_fidelity"),
        "non_hf_or_failed": sum(1 for rec in results if not (bool(rec.get("pair_ok")) and str(rec.get("stage")) == "high_fidelity")),
        "best_score": min(
            [
                float(rec.get("score"))
                for rec in results
                if isinstance(rec.get("score"), (int, float))
            ],
            default=None,
        ),
    }
    final_payload = {
        "run_dir": str(run_dir),
        "device": str(args.device),
        "pure_no_gate": bool(args.pure_no_gate),
        "pair_reasons": [v.strip() for v in str(args.pair_reasons).split(",") if v.strip()],
        "selected_count_before_shard": len(all_candidates),
        "selected_pairs": selected_summary,
        "sample_size": (int(args.sample_size) if args.sample_size is not None else None),
        "sample_seed": int(args.sample_seed),
        "min_per_generation": int(args.min_per_generation),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "results": results,
        "summary": summary,
    }
    output_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"report_path={output_path}", flush=True)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay previously gate-rejected (g,f) pairs at high fidelity with cheap gates disabled."
    )
    p.add_argument("--run-dir", default=None, type=str, help="Run directory containing checkpoint.json and gate_reports.jsonl")
    p.add_argument("--device", default="cuda:0", type=str, help="Replay device, e.g. cuda:0")
    p.add_argument("--pair-reasons", default="cheap_gate_failed", type=str, help="CSV of pair_reason values to replay")
    p.add_argument("--generation", default=None, type=str, help='Optional generation filter, integer or "latest"')
    p.add_argument("--max-pairs", default=None, type=int, help="Optional cap on number of pairs to replay")
    p.add_argument("--sample-size", default=None, type=int, help="Randomly sample this many pairs before replay")
    p.add_argument("--sample-seed", default=1234, type=int, help="Random seed for --sample-size")
    p.add_argument("--min-per-generation", default=0, type=int, help="When sampling, keep at least this many pairs per generation if available")
    p.add_argument("--num-shards", default=1, type=int, help="Split selected pairs into this many shards")
    p.add_argument("--shard-index", default=0, type=int, help="0-based shard index to run")
    p.add_argument("--output", default=None, type=str, help="Optional JSON report path")
    p.add_argument("--python", default=None, type=str, help="Python executable for worker subprocesses")
    p.add_argument("--pure-no-gate", action="store_true", help="Also disable sandbox/repair-related gate helpers")
    p.add_argument("--dry-run", action="store_true", help="Only print selected pairs, do not evaluate")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--payload", default=None, type=str, help=argparse.SUPPRESS)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.worker:
        if not args.payload:
            raise SystemExit("--worker requires --payload")
        return _worker_main(args)
    if not args.run_dir:
        raise SystemExit("--run-dir is required unless --worker is set")
    return _coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
