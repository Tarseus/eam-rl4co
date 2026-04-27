from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
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


_configure_import_path()

from fitness.free_loss_fidelity import FreeLossFidelityConfig, evaluate_free_loss_candidate  # noqa: E402
from ptp_discovery.free_loss_compiler import compile_free_loss  # noqa: E402
from ptp_discovery.free_loss_ir import ir_from_json as free_loss_ir_from_json  # noqa: E402
from ptp_discovery.pref_builder_compiler import compile_preference_builder  # noqa: E402
from ptp_discovery.pref_loss_coevo_loop import (  # noqa: E402
    _CompiledBuilderAdapter,
    _abs_from_repo_root,
    _build_hf_cfg,
    _ref_builder_ir,
    _resolve_training_seed,
    _stage3_init_specs_from_baseline_cfg,
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict YAML at {path}")
    return dict(payload)


def _find_latest_run_dir(output_root: Path) -> Path:
    candidates = [p for p in output_root.iterdir() if p.is_dir() and (p / "checkpoint.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No run directories with checkpoint.json under {output_root}")
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


def _load_loss_entries(
    run_dir: Path,
    *,
    generation: str | None,
    include_ref: bool,
    include_compile_failures: bool,
    max_losses: int | None,
) -> List[Dict[str, Any]]:
    losses_path = run_dir / "losses.jsonl"
    if not losses_path.is_file():
        raise FileNotFoundError(f"Missing losses.jsonl: {losses_path}")

    records = list(_iter_jsonl(losses_path))
    if generation:
        gen_token = str(generation).strip().lower()
        if gen_token == "latest":
            gens = [int(rec.get("generation", -1)) for rec in records if rec.get("generation") is not None]
            if gens:
                target = max(gens)
                records = [rec for rec in records if int(rec.get("generation", -1)) == int(target)]
        else:
            target = int(gen_token)
            records = [rec for rec in records if int(rec.get("generation", -1)) == int(target)]

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rec in records:
        f_id = str(rec.get("id") or rec.get("f_id") or "").strip()
        if not f_id or f_id in seen:
            continue
        if not include_ref and f_id == "f_ref":
            continue
        if not include_compile_failures and not bool(rec.get("compile_ok", True)):
            continue
        if not isinstance(rec.get("ir"), Mapping):
            continue
        seen.add(f_id)
        out.append(dict(rec))
        if max_losses is not None and len(out) >= int(max_losses):
            break
    return out


def _cuda_device_for_stats(device_str: str) -> torch.device | None:
    ds = str(device_str or "").strip()
    if not torch.cuda.is_available():
        return None
    if ds == "cuda":
        return torch.device("cuda:0")
    if ds.startswith("cuda"):
        return torch.device(ds)
    return None


def _require_requested_device_available(device_str: str) -> None:
    ds = str(device_str or "").strip().lower()
    if ds.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested a CUDA device but the current Python environment reports "
            "torch.cuda.is_available() == False. Run this script inside a CUDA-enabled PyTorch env."
        )


def _peak_memory_mb(device_str: str) -> Dict[str, float | None]:
    dev = _cuda_device_for_stats(device_str)
    if dev is None:
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    torch.cuda.synchronize(dev)
    return {
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(dev)) / (1024.0**2),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(dev)) / (1024.0**2),
    }


def _reset_cuda_stats(device_str: str) -> None:
    dev = _cuda_device_for_stats(device_str)
    if dev is None:
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)


def _clear_cuda(device_str: str) -> None:
    dev = _cuda_device_for_stats(device_str)
    if dev is None:
        return
    try:
        torch.cuda.synchronize(dev)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _build_stage3_hf_cfg(cfg_yaml: Mapping[str, Any], *, device_str: str):
    scratch_init_seed = int(_resolve_training_seed(cfg_yaml))
    hf_epochs_cfg = int(cfg_yaml.get("hf_epochs", 0) or 0)
    hf_inst_cfg = int(cfg_yaml.get("hf_instances_per_epoch", 0) or 0)
    K = int(cfg_yaml.get("f1_steps", 32) or 32)
    cfg_hf = dict(cfg_yaml)
    cfg_hf["f1_steps"] = int(K)
    if not (hf_epochs_cfg > 0 and hf_inst_cfg > 0):
        cfg_hf["hf_epochs"] = 0
        cfg_hf["hf_instances_per_epoch"] = 0
    hf_cfg = _build_hf_cfg(cfg_hf, seed=int(scratch_init_seed), device_str=str(device_str))
    return K, scratch_init_seed, hf_cfg


def _evaluate_one_loss(
    *,
    config_path: Path,
    run_dir: Path,
    f_id: str,
    device_str: str,
) -> Dict[str, Any]:
    cfg_yaml = _read_yaml(config_path)
    losses = _load_loss_entries(
        run_dir,
        generation=None,
        include_ref=True,
        include_compile_failures=True,
        max_losses=None,
    )
    target = next((rec for rec in losses if str(rec.get("id") or rec.get("f_id")) == str(f_id)), None)
    if target is None:
        raise KeyError(f"Loss id not found in {run_dir / 'losses.jsonl'}: {f_id}")

    compiled_builder = compile_preference_builder(_ref_builder_ir(), operator_whitelist=[])
    adapter = _CompiledBuilderAdapter(compiled_builder)

    ir_payload = target.get("ir")
    if not isinstance(ir_payload, Mapping):
        raise ValueError(f"Loss {f_id} has no IR payload")
    compiled_loss = compile_free_loss(free_loss_ir_from_json(ir_payload), operator_whitelist=[])

    K, scratch_seed, hf_cfg = _build_stage3_hf_cfg(cfg_yaml, device_str=device_str)
    init_specs = _stage3_init_specs_from_baseline_cfg(cfg_yaml)
    valid_sizes = [int(v) for v in cfg_yaml.get("valid_problem_sizes", list(hf_cfg.valid_problem_sizes))]
    if not valid_sizes:
        valid_sizes = [int(hf_cfg.train_problem_size)]

    result: Dict[str, Any] = {
        "f_id": str(f_id),
        "generation": int(target.get("generation", -1) or -1),
        "family": target.get("family"),
        "family_signature": target.get("family_signature"),
        "device": str(device_str),
        "train_batch_size": int(cfg_yaml.get("train_batch_size", 64) or 64),
        "validation_batch_size": int(cfg_yaml.get("validation_batch_size", 64) or 64),
        "pomo_size": cfg_yaml.get("pomo_size"),
        "f1_steps": int(K),
        "hf_epochs": int(cfg_yaml.get("hf_epochs", 0) or 0),
        "hf_instances_per_epoch": int(cfg_yaml.get("hf_instances_per_epoch", 0) or 0),
        "scratch_init_seed": int(scratch_seed),
        "valid_problem_sizes": list(valid_sizes),
        "per_init": [],
    }

    for init_name, init_ckpt in init_specs:
        gc.collect()
        _clear_cuda(device_str)
        _reset_cuda_stats(device_str)
        started = time.time()
        per_init: Dict[str, Any] = {
            "name": str(init_name),
            "init_checkpoint": str(init_ckpt) if init_ckpt else None,
        }
        try:
            free_cfg = FreeLossFidelityConfig(
                hf=hf_cfg,
                f1_steps=int(K),
                f2_steps=0,
                f3_enabled=False,
                init_checkpoint_path=_abs_from_repo_root(str(init_ckpt)) if init_ckpt else None,
                init_checkpoint_epoch=None,
                scratch_hf_epochs=int(cfg_yaml.get("scratch_hf_epochs", 0) or 0),
                warmstart_hf_epochs=int(cfg_yaml.get("warmstart_hf_epochs", 0) or 0),
                baseline_epoch_compare_offset=int(cfg_yaml.get("baseline_epoch_compare_offset", 0) or 0),
                baseline_epoch_violation_weight=float(cfg_yaml.get("baseline_epoch_violation_weight", 1.0)),
                baseline_epoch_tail_frac=float(cfg_yaml.get("baseline_epoch_tail_frac", 1.0) or 1.0),
                baseline_epoch_window_k=int(cfg_yaml.get("baseline_epoch_window_k", 10) or 10),
                baseline_epoch_window_violation_weight=float(
                    cfg_yaml.get("baseline_epoch_window_violation_weight", 1.0) or 1.0
                ),
            )
            fitness = evaluate_free_loss_candidate(compiled_loss, free_cfg, pref_builder=adapter)
            size_objectives_raw = fitness.get("size_objectives", {})
            size_objectives: Dict[str, float] = {}
            if isinstance(size_objectives_raw, Mapping):
                for k, v in size_objectives_raw.items():
                    try:
                        size_objectives[str(int(k))] = float(v)
                    except Exception:
                        continue
            cand_agg = None
            if size_objectives:
                vals = [float(size_objectives[str(int(sz))]) for sz in valid_sizes if str(int(sz)) in size_objectives]
                if vals:
                    cand_agg = float(sum(vals) / float(len(vals)))
            per_init["ok"] = True
            per_init["oom"] = False
            per_init["aggregated_objective"] = cand_agg
            per_init["size_objectives"] = size_objectives
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
            per_init.update(_peak_memory_mb(device_str))
            per_init["elapsed_s"] = float(time.time() - started)
            result["per_init"].append(per_init)
            gc.collect()
            _clear_cuda(device_str)

    result["any_oom"] = bool(any(bool(item.get("oom")) for item in result["per_init"]))
    result["all_ok"] = bool(all(bool(item.get("ok")) for item in result["per_init"]))
    result["oom_init_names"] = [str(item.get("name")) for item in result["per_init"] if bool(item.get("oom"))]
    return result


def _summarize(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    oom_results = [r for r in results if bool(r.get("any_oom"))]
    ok_results = [r for r in results if bool(r.get("all_ok"))]
    per_init_oom: Dict[str, int] = {}
    worst_peak = 0.0
    worst_peak_fid = None
    for rec in results:
        for init_rec in rec.get("per_init", []) or []:
            if bool(init_rec.get("oom")):
                name = str(init_rec.get("name"))
                per_init_oom[name] = int(per_init_oom.get(name, 0)) + 1
            peak = init_rec.get("peak_allocated_mb")
            if peak is not None and float(peak) >= float(worst_peak):
                worst_peak = float(peak)
                worst_peak_fid = str(rec.get("f_id"))
    return {
        "total_losses": int(total),
        "oom_losses": int(len(oom_results)),
        "all_ok_losses": int(len(ok_results)),
        "non_oom_but_failed_losses": int(total - len(oom_results) - len(ok_results)),
        "per_init_oom_counts": per_init_oom,
        "worst_peak_allocated_mb": float(worst_peak) if worst_peak else None,
        "worst_peak_loss_id": worst_peak_fid,
        "oom_loss_ids": [str(rec.get("f_id")) for rec in oom_results],
    }


def _worker_main(args: argparse.Namespace) -> int:
    _require_requested_device_available(str(args.device))
    config_path = Path(args.config).resolve()
    run_dir = _resolve_run_dir(config_path, args.run_dir)
    payload = _evaluate_one_loss(
        config_path=config_path,
        run_dir=run_dir,
        f_id=str(args.f_id),
        device_str=str(args.device),
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def _coordinator_main(args: argparse.Namespace) -> int:
    _require_requested_device_available(str(args.device))
    config_path = Path(args.config).resolve()
    run_dir = _resolve_run_dir(config_path, args.run_dir)
    selected = _load_loss_entries(
        run_dir,
        generation=args.generation,
        include_ref=bool(args.include_ref),
        include_compile_failures=bool(args.include_compile_failures),
        max_losses=(int(args.max_losses) if args.max_losses is not None else None),
    )
    if not selected:
        raise SystemExit("No matching losses found to probe.")

    repo_root = _repo_root_dir()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else (run_dir / f"stage3_oom_probe_{str(args.device).replace(':', '_')}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "stage3_oom_probe_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    py_paths = [str(repo_root), str(repo_root / "PTP")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    results: List[Dict[str, Any]] = []
    python_exe = str(Path(args.python or sys.executable).resolve())
    script_path = Path(__file__).resolve()

    for idx, rec in enumerate(selected, start=1):
        f_id = str(rec.get("id") or rec.get("f_id"))
        cmd = [
            str(python_exe),
            str(script_path),
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--device",
            str(args.device),
            "--worker",
            "--f-id",
            str(f_id),
        ]
        started = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(repo_root),
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        log_path = log_dir / f"{idx:03d}_{f_id}.log"
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

        payload: Dict[str, Any]
        if stdout:
            try:
                payload = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError:
                payload = {
                    "f_id": str(f_id),
                    "any_oom": False,
                    "all_ok": False,
                    "per_init": [],
                    "error_type": "worker_output_parse_error",
                    "error": stdout[-4000:],
                }
        else:
            payload = {
                "f_id": str(f_id),
                "any_oom": False,
                "all_ok": False,
                "per_init": [],
                "error_type": "worker_no_output",
                "error": stderr[-4000:],
            }

        payload["worker_returncode"] = int(proc.returncode)
        payload["worker_elapsed_s"] = float(time.time() - started)
        payload["worker_log"] = os.path.relpath(str(log_path), start=str(run_dir))
        results.append(payload)

        summary = _summarize(results)
        checkpoint_payload = {
            "config": str(config_path),
            "run_dir": str(run_dir),
            "device": str(args.device),
            "worker_python": str(python_exe),
            "generation": args.generation,
            "results": results,
            "summary": summary,
        }
        output_path.write_text(json.dumps(checkpoint_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{idx}/{len(selected)}] f_id={f_id} any_oom={bool(payload.get('any_oom'))} "
            f"all_ok={bool(payload.get('all_ok'))} rc={int(proc.returncode)}",
            flush=True,
        )

    final_payload = {
        "config": str(config_path),
        "run_dir": str(run_dir),
        "device": str(args.device),
        "worker_python": str(python_exe),
        "generation": args.generation,
        "results": results,
        "summary": _summarize(results),
    }
    output_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_payload["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"report_path={output_path}", flush=True)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay pref-loss stage3 candidate evals and report which losses still OOM."
    )
    p.add_argument("--config", required=True, type=str, help="Path to pref_loss_coevo YAML config.")
    p.add_argument("--run-dir", default=None, type=str, help="Run directory to replay. Defaults to latest run.")
    p.add_argument("--device", default="cuda:0", type=str, help="Device for replay, e.g. cuda:0.")
    p.add_argument(
        "--generation",
        default=None,
        type=str,
        help='Optional generation filter. Pass an integer or "latest". Default probes all losses in the run.',
    )
    p.add_argument("--max-losses", default=None, type=int, help="Optional cap on number of losses to replay.")
    p.add_argument("--output", default=None, type=str, help="Optional JSON report path.")
    p.add_argument("--python", default=None, type=str, help="Python executable to use for worker subprocesses.")
    p.add_argument("--include-ref", action="store_true", help="Include f_ref if present in losses.jsonl.")
    p.add_argument(
        "--include-compile-failures",
        action="store_true",
        help="Include losses marked compile_ok=false in losses.jsonl.",
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--f-id", default=None, type=str, help=argparse.SUPPRESS)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.worker:
        if not args.f_id:
            raise SystemExit("--worker requires --f-id")
        return _worker_main(args)
    return _coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
