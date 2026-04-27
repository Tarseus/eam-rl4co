from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml


def _repo_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_import_path() -> None:
    repo_root = _repo_root_dir()
    ptp_root = repo_root / "PTP"
    for path in (repo_root, ptp_root):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict YAML at {path}")
    return dict(payload)


def _find_latest_run_dir(output_root: Path) -> Path:
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories under {output_root}")
    return sorted(candidates)[-1]


def _resolve_run_dir(config_path: Path, run_dir_raw: str | None) -> Path:
    if run_dir_raw:
        run_dir = Path(run_dir_raw)
        if not run_dir.is_absolute():
            run_dir = (_repo_root_dir() / run_dir).resolve()
        return run_dir
    cfg = _read_yaml(config_path)
    output_root = cfg.get("output_root", "runs/pref_loss_coevo")
    out_dir = Path(output_root)
    if not out_dir.is_absolute():
        out_dir = (_repo_root_dir() / out_dir).resolve()
    return _find_latest_run_dir(out_dir)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _parse_explicit_fids(explicit_fids_raw: str | None) -> List[str]:
    if not explicit_fids_raw:
        return []
    out: List[str] = []
    for token in str(explicit_fids_raw).split(","):
        fid = str(token).strip()
        if fid:
            out.append(fid)
    return out


def _pick_target_fids(
    run_dir: Path,
    *,
    explicit_fids_raw: str | None,
    failed_only: bool,
    max_targets: int | None,
) -> List[str]:
    explicit_fids = _parse_explicit_fids(explicit_fids_raw)
    if explicit_fids:
        uniq: List[str] = []
        seen: set[str] = set()
        for fid in explicit_fids:
            if fid not in seen:
                seen.add(fid)
                uniq.append(fid)
        return uniq[: int(max_targets)] if max_targets is not None else uniq

    pairs_path = run_dir / "pairs.jsonl"
    if pairs_path.is_file():
        records = list(_iter_jsonl(pairs_path))
        if failed_only:
            preferred = [
                rec
                for rec in records
                if str(rec.get("pair_reason", "")) in {"stage3_early_pruned", "stage3_runtime_error", "stage3_fatal"}
            ]
        else:
            preferred = [
                rec
                for rec in records
                if str(rec.get("pair_reason", "")) in {"ok_stage3_offline_minitrain", "stage3_early_pruned", "stage3_runtime_error"}
            ]
        if not preferred:
            preferred = [rec for rec in records if bool(rec.get("joint_gate_ok")) and bool(rec.get("co_ok"))]
        if preferred:
            preferred = sorted(
                preferred,
                key=lambda rec: (int(rec.get("generation", -1) or -1), int(rec.get("pair_index", -1) or -1)),
            )
            out: List[str] = []
            seen: set[str] = set()
            for rec in reversed(preferred):
                fid = str(rec.get("f_id") or "").strip()
                if not fid or fid in seen:
                    continue
                seen.add(fid)
                out.append(fid)
                if max_targets is not None and len(out) >= int(max_targets):
                    break
            if out:
                return out

    losses_path = run_dir / "losses.jsonl"
    if losses_path.is_file():
        out = []
        seen: set[str] = set()
        for rec in _iter_jsonl(losses_path):
            fid = str(rec.get("id") or rec.get("f_id") or "").strip()
            if fid and fid != "f_ref" and bool(rec.get("compile_ok", True)) and fid not in seen:
                seen.add(fid)
                out.append(fid)
                if max_targets is not None and len(out) >= int(max_targets):
                    break
        if out:
            return out

    raise FileNotFoundError(f"Could not determine target loss ids from run_dir={run_dir}")


def _effective_fixed_pomo(cfg_yaml: Mapping[str, Any], explicit_pomo: int | None) -> int:
    if explicit_pomo is not None:
        return int(explicit_pomo)
    pomo = cfg_yaml.get("pomo_size", None)
    if pomo is None:
        problem_size = int(cfg_yaml.get("train_problem_size", 1) or 1)
        return max(problem_size, 1)
    return max(int(pomo), 1)


def _parse_batch_sizes(raw: str | None, *, min_batch_size: int, max_batch_size: int) -> List[int]:
    if raw:
        values = []
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            values.append(int(token))
        uniq = sorted({int(v) for v in values if int(v) > 0})
        if not uniq:
            raise ValueError("No positive batch sizes parsed from --batch-sizes")
        return uniq
    if min_batch_size <= 0 or max_batch_size <= 0:
        raise ValueError("min/max batch size must be positive")
    if min_batch_size > max_batch_size:
        raise ValueError("min_batch_size must be <= max_batch_size")
    return list(range(int(min_batch_size), int(max_batch_size) + 1))


def _write_override_config(
    base_cfg: Mapping[str, Any],
    *,
    batch_size: int,
    pomo_size: int,
) -> Path:
    payload = dict(base_cfg)
    payload["train_batch_size"] = int(batch_size)
    payload["pomo_size"] = int(pomo_size)
    tmp = tempfile.NamedTemporaryFile(prefix="ffsp_stage3_probe_", suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    with tmp:
        yaml.safe_dump(payload, tmp, sort_keys=False, allow_unicode=True)
    return Path(tmp.name)


def _probe_one_baseline_batch_size(
    *,
    base_cfg: Mapping[str, Any],
    device: str,
    batch_size: int,
    pomo_size: int,
) -> Dict[str, Any]:
    _configure_import_path()

    import gc
    import traceback

    import torch

    from fitness.free_loss_fidelity import evaluate_po_baseline_rl4co
    from ptp_discovery.pref_loss_coevo_loop import _build_hf_cfg, _resolve_training_seed, _stage3_init_specs_from_baseline_cfg

    cfg_yaml = dict(base_cfg)
    cfg_yaml["train_batch_size"] = int(batch_size)
    cfg_yaml["pomo_size"] = int(pomo_size)
    scratch_seed = int(_resolve_training_seed(cfg_yaml))
    hf_cfg = _build_hf_cfg(cfg_yaml, seed=scratch_seed, device_str=str(device))
    init_specs = _stage3_init_specs_from_baseline_cfg(cfg_yaml)

    result: Dict[str, Any] = {
        "attempt_batch_size": int(batch_size),
        "attempt_pomo_size": int(pomo_size),
        "device": str(device),
        "scratch_init_seed": int(scratch_seed),
        "per_init": [],
    }

    def _reset_cuda_peak() -> None:
        if not torch.cuda.is_available():
            return
        dev = torch.device(str(device))
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        except Exception:
            pass

    def _peak_memory_mb() -> Dict[str, float | None]:
        if not torch.cuda.is_available():
            return {"peak_allocated_mb": None, "peak_reserved_mb": None}
        dev = torch.device(str(device))
        try:
            torch.cuda.synchronize(dev)
            return {
                "peak_allocated_mb": float(torch.cuda.max_memory_allocated(dev)) / (1024.0**2),
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved(dev)) / (1024.0**2),
            }
        except Exception:
            return {"peak_allocated_mb": None, "peak_reserved_mb": None}

    for init_name, init_ckpt in init_specs:
        gc.collect()
        _reset_cuda_peak()
        started = time.time()
        per_init: Dict[str, Any] = {
            "name": str(init_name),
            "init_checkpoint": str(init_ckpt) if init_ckpt else None,
        }
        try:
            payload = evaluate_po_baseline_rl4co(
                hf_cfg,
                init_checkpoint_path=(str((_repo_root_dir() / str(init_ckpt)).resolve()) if init_ckpt else None),
                init_checkpoint_epoch=None,
                scratch_hf_epochs=int(cfg_yaml.get("scratch_hf_epochs", 0) or 0),
                warmstart_hf_epochs=int(cfg_yaml.get("warmstart_hf_epochs", 0) or 0),
                baseline_epoch_compare_offset=int(cfg_yaml.get("baseline_epoch_compare_offset", 0) or 0),
                baseline_epoch_violation_weight=float(cfg_yaml.get("baseline_epoch_violation_weight", 1.0) or 1.0),
                baseline_epoch_tail_frac=float(cfg_yaml.get("baseline_epoch_tail_frac", 1.0) or 1.0),
                baseline_epoch_window_k=int(cfg_yaml.get("baseline_epoch_window_k", 10) or 10),
                baseline_epoch_window_violation_weight=float(
                    cfg_yaml.get("baseline_epoch_window_violation_weight", 1.0) or 1.0
                ),
            )
            per_init["ok"] = True
            per_init["oom"] = False
            if isinstance(payload, Mapping):
                metadata = payload.get("metadata")
                if isinstance(metadata, Mapping):
                    fit = metadata.get("fitness")
                    if isinstance(fit, Mapping):
                        per_init["fitness_score"] = fit.get("score")
                        per_init["final_validation_objective"] = fit.get("final_validation_objective")
        except torch.cuda.OutOfMemoryError as exc:
            per_init["ok"] = False
            per_init["oom"] = True
            per_init["error_type"] = "cuda_oom"
            per_init["error"] = str(exc)
        except RuntimeError as exc:
            msg = str(exc)
            per_init["ok"] = False
            per_init["oom"] = "out of memory" in msg.lower()
            per_init["error_type"] = "runtime_error"
            per_init["error"] = msg
        except Exception as exc:  # noqa: BLE001
            per_init["ok"] = False
            per_init["oom"] = False
            per_init["error_type"] = type(exc).__name__
            per_init["error"] = str(exc)
            per_init["traceback"] = traceback.format_exc(limit=10)
        finally:
            per_init.update(_peak_memory_mb())
            per_init["elapsed_s"] = float(time.time() - started)
            result["per_init"].append(per_init)
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    result["any_oom"] = bool(any(bool(item.get("oom")) for item in result["per_init"]))
    result["all_ok"] = bool(all(bool(item.get("ok")) for item in result["per_init"]))
    result["success"] = bool(result["all_ok"])
    return result


def _probe_one_batch_size(
    *,
    config_path: Path,
    base_cfg: Mapping[str, Any],
    run_dir: Path | None,
    device: str,
    python_exe: str,
    target_fid: str,
    batch_size: int,
    pomo_size: int,
    probe_target: str,
) -> Dict[str, Any]:
    if str(probe_target) == "baseline":
        return _probe_one_baseline_batch_size(
            base_cfg=base_cfg,
            device=device,
            batch_size=batch_size,
            pomo_size=pomo_size,
        )
    override_path = _write_override_config(base_cfg, batch_size=batch_size, pomo_size=pomo_size)
    try:
        cmd = [
            python_exe,
            str((_repo_root_dir() / "scripts" / "check_pref_loss_stage3_oom.py").resolve()),
            "--config",
            str(override_path),
            "--run-dir",
            str(run_dir),
            "--device",
            str(device),
            "--worker",
            "--f-id",
            str(target_fid),
        ]
        env = dict(os.environ)
        py_paths = [str(_repo_root_dir()), str(_repo_root_dir() / "PTP")]
        if env.get("PYTHONPATH"):
            py_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(py_paths)

        started = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_repo_root_dir()),
            env=env,
        )
        elapsed_s = float(time.time() - started)

        payload: Dict[str, Any]
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if stdout:
            try:
                payload = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError:
                payload = {
                    "all_ok": False,
                    "any_oom": False,
                    "per_init": [],
                    "error_type": "worker_output_parse_error",
                    "error": stdout[-4000:],
                }
        else:
            payload = {
                "all_ok": False,
                "any_oom": False,
                "per_init": [],
                "error_type": "worker_no_output",
                "error": stderr[-4000:],
            }

        payload["attempt_batch_size"] = int(batch_size)
        payload["attempt_pomo_size"] = int(pomo_size)
        payload["worker_returncode"] = int(proc.returncode)
        payload["worker_elapsed_s"] = float(elapsed_s)
        payload["worker_stderr_tail"] = stderr[-4000:] if stderr else ""
        payload["success"] = bool(payload.get("all_ok"))
        return payload
    finally:
        try:
            override_path.unlink(missing_ok=True)
        except Exception:
            pass


def _binary_search_batch_capacity(
    *,
    config_path: Path,
    base_cfg: Mapping[str, Any],
    run_dir: Path | None,
    device: str,
    python_exe: str,
    target_fid: str,
    batch_sizes: Sequence[int],
    pomo_size: int,
    probe_target: str,
) -> Dict[str, Any]:
    sorted_sizes = sorted({int(v) for v in batch_sizes if int(v) > 0})
    attempts: List[Dict[str, Any]] = []
    lo = 0
    hi = len(sorted_sizes) - 1
    best: Dict[str, Any] | None = None

    while lo <= hi:
        mid = (lo + hi) // 2
        batch_size = int(sorted_sizes[mid])
        payload = _probe_one_batch_size(
            config_path=config_path,
            base_cfg=base_cfg,
            run_dir=run_dir,
            device=device,
            python_exe=python_exe,
            target_fid=target_fid,
            batch_size=batch_size,
            pomo_size=pomo_size,
            probe_target=probe_target,
        )
        attempts.append(payload)
        if bool(payload.get("success")):
            best = payload
            lo = mid + 1
        else:
            hi = mid - 1

    return {
        "mode": "binary",
        "attempts": attempts,
        "max_supported_batch_size": (int(best["attempt_batch_size"]) if best is not None else None),
        "best_attempt": best,
    }


def _list_probe_batch_capacity(
    *,
    config_path: Path,
    base_cfg: Mapping[str, Any],
    run_dir: Path | None,
    device: str,
    python_exe: str,
    target_fid: str,
    batch_sizes: Sequence[int],
    pomo_size: int,
    probe_target: str,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for batch_size in sorted({int(v) for v in batch_sizes if int(v) > 0}):
        payload = _probe_one_batch_size(
            config_path=config_path,
            base_cfg=base_cfg,
            run_dir=run_dir,
            device=device,
            python_exe=python_exe,
            target_fid=target_fid,
            batch_size=batch_size,
            pomo_size=pomo_size,
            probe_target=probe_target,
        )
        attempts.append(payload)
        if bool(payload.get("success")):
            best = payload
    return {
        "mode": "list",
        "attempts": attempts,
        "max_supported_batch_size": (int(best["attempt_batch_size"]) if best is not None else None),
        "best_attempt": best,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe the largest FFSP stage3 train_batch_size supported by the current machine while keeping pomo_size fixed."
    )
    p.add_argument(
        "--config",
        default="PTP/configs/experiment/pref_loss_coevo/loss_only_ffsp100_discovery.yaml",
        type=str,
        help="Path to the pref-loss YAML config.",
    )
    p.add_argument("--run-dir", default=None, type=str, help="Run dir used to select a representative loss in stage3 mode.")
    p.add_argument("--device", default="cuda:0", type=str, help="Target device, e.g. cuda:0.")
    p.add_argument("--python", default=sys.executable, type=str, help="Python executable for worker subprocesses.")
    p.add_argument("--f-id", default=None, type=str, help="Explicit loss id to probe in stage3 mode.")
    p.add_argument("--f-ids", default=None, type=str, help="Comma-separated loss ids to probe in stage3 mode.")
    p.add_argument("--failed-only", action="store_true", help="In stage3 mode, prefer only previously stage3-failed individuals.")
    p.add_argument("--max-targets", default=None, type=int, help="Optional cap on number of target losses in stage3 mode.")
    p.add_argument(
        "--probe-target",
        default="stage3",
        choices=("stage3", "baseline"),
        help="stage3: probe a representative candidate loss; baseline: probe the common FFSP100 baseline mini-train path.",
    )
    p.add_argument("--fixed-pomo-size", default=None, type=int, help="Fixed pomo_size to use for all attempts.")
    p.add_argument(
        "--batch-sizes",
        default=None,
        type=str,
        help="Comma-separated explicit batch sizes to probe, e.g. 4,8,12,16,20,24.",
    )
    p.add_argument("--min-batch-size", default=1, type=int, help="Minimum batch size for auto-generated probe list.")
    p.add_argument("--max-batch-size", default=64, type=int, help="Maximum batch size for auto-generated probe list.")
    p.add_argument(
        "--mode",
        default="binary",
        choices=("binary", "list"),
        help="binary: search max supported batch; list: probe every candidate batch size.",
    )
    p.add_argument("--output", default=None, type=str, help="Optional JSON report path.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    config_path = Path(args.config).resolve()
    base_cfg = _read_yaml(config_path)
    if str(args.probe_target) == "stage3":
        run_dir = _resolve_run_dir(config_path, args.run_dir)
        explicit_fids_raw = args.f_ids if args.f_ids is not None else args.f_id
        target_fids = _pick_target_fids(
            run_dir,
            explicit_fids_raw=explicit_fids_raw,
            failed_only=bool(args.failed_only),
            max_targets=(int(args.max_targets) if args.max_targets is not None else None),
        )
    else:
        run_dir = None
        target_fids = []
    fixed_pomo_size = _effective_fixed_pomo(base_cfg, args.fixed_pomo_size)
    batch_sizes = _parse_batch_sizes(
        args.batch_sizes,
        min_batch_size=int(args.min_batch_size),
        max_batch_size=int(args.max_batch_size),
    )
    python_exe = str(Path(args.python).resolve())

    def _run_for_one_target(target_fid: str) -> Dict[str, Any]:
        if args.mode == "binary":
            return _binary_search_batch_capacity(
                config_path=config_path,
                base_cfg=base_cfg,
                run_dir=run_dir,
                device=str(args.device),
                python_exe=python_exe,
                target_fid=str(target_fid),
                batch_sizes=batch_sizes,
                pomo_size=fixed_pomo_size,
                probe_target=str(args.probe_target),
            )
        return _list_probe_batch_capacity(
            config_path=config_path,
            base_cfg=base_cfg,
            run_dir=run_dir,
            device=str(args.device),
            python_exe=python_exe,
            target_fid=str(target_fid),
            batch_sizes=batch_sizes,
            pomo_size=fixed_pomo_size,
            probe_target=str(args.probe_target),
        )

    if str(args.probe_target) == "baseline":
        payload = _run_for_one_target("")
        report = {
            "config_path": str(config_path),
            "probe_target": str(args.probe_target),
            "run_dir": None,
            "device": str(args.device),
            "python": python_exe,
            "target_fid": None,
            "fixed_pomo_size": int(fixed_pomo_size),
            "batch_sizes": [int(v) for v in batch_sizes],
            **payload,
        }
    else:
        per_target_reports: List[Dict[str, Any]] = []
        for target_fid in target_fids:
            payload = _run_for_one_target(str(target_fid))
            per_target_reports.append(
                {
                    "target_fid": str(target_fid),
                    **payload,
                }
            )

        supported = [
            int(item["max_supported_batch_size"])
            for item in per_target_reports
            if item.get("max_supported_batch_size") is not None
        ]
        report = {
            "config_path": str(config_path),
            "probe_target": str(args.probe_target),
            "run_dir": (str(run_dir) if run_dir is not None else None),
            "device": str(args.device),
            "python": python_exe,
            "target_fids": [str(fid) for fid in target_fids],
            "fixed_pomo_size": int(fixed_pomo_size),
            "batch_sizes": [int(v) for v in batch_sizes],
            "per_target": per_target_reports,
            "safe_batch_size_for_all_targets": (min(supported) if supported else None),
            "max_supported_batch_size": (min(supported) if supported else None),
        }

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (
            ((run_dir if run_dir is not None else config_path.parent) / (
                f"{str(args.probe_target)}_batch_probe_{str(args.device).replace(':', '_')}_pomo{int(fixed_pomo_size)}.json"
            ))
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(
        {
            "target_fid": report.get("target_fid"),
            "target_fids": report.get("target_fids"),
            "fixed_pomo_size": report["fixed_pomo_size"],
            "max_supported_batch_size": report["max_supported_batch_size"],
            "attempt_count": (
                len(report["attempts"])
                if "attempts" in report
                else sum(len(item.get("attempts", [])) for item in report.get("per_target", []))
            ),
            "report_path": str(output_path),
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
