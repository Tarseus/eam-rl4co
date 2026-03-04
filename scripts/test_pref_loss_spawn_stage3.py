from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml


def _repo_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _abs_from_repo_root(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_repo_root_dir() / p).resolve()


def _parse_devices(raw: str) -> List[str]:
    devices: List[str] = []
    for part in str(raw or "").split(","):
        s = part.strip()
        if s:
            devices.append(s)
    return devices


def _run(cmd: Sequence[str], *, env: Dict[str, str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(list(cmd), check=True, env=env, cwd=str(cwd))


def _find_latest_run_dir(out_root: Path) -> Path:
    candidates = [
        p for p in out_root.iterdir() if p.is_dir() and (p / "checkpoint.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run directories under: {out_root}")
    return sorted(candidates)[-1]


def _torch_runtime_info(python_exe: str, *, env: Dict[str, str], cwd: Path) -> Dict[str, Any]:
    code = (
        "import json, torch; "
        "print(json.dumps({"
        "'python': __import__('sys').version, "
        "'torch_version': getattr(torch, '__version__', None), "
        "'torch_cuda_version': getattr(getattr(torch, 'version', None), 'cuda', None), "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_count': int(torch.cuda.device_count())"
        "}))"
    )
    res = subprocess.run(
        [python_exe, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
    )
    return dict(json.loads(res.stdout.strip()))


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _collect_pair_summary(pairs_jsonl: Path) -> Dict[str, Any]:
    stage_ctr: collections.Counter[str] = collections.Counter()
    reason_ctr: collections.Counter[str] = collections.Counter()
    device_ctr: collections.Counter[str] = collections.Counter()
    hf_records = 0
    hf_failures = 0
    pair_failures = 0
    crash_reasons = {
        "child_exception",
        "child_timeout",
        "child_exit_no_result",
        "high_fidelity_failed",
        "stage3_fatal",
        "stage3_runtime_error",
    }
    bad_examples: List[Dict[str, Any]] = []

    with pairs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stage = str(rec.get("stage", ""))
            reason = str(rec.get("pair_reason", ""))
            stage_ctr[stage] += 1
            reason_ctr[reason] += 1
            if not bool(rec.get("pair_ok", False)):
                pair_failures += 1
            if stage == "high_fidelity":
                hf_records += 1
                device_ctr[str(rec.get("device", rec.get("device_str", "")))] += 1
                if reason in crash_reasons or not bool(rec.get("pair_ok", False)):
                    hf_failures += 1
                    if len(bad_examples) < 8:
                        bad_examples.append(
                            {
                                "pair_index": rec.get("pair_index"),
                                "g_id": rec.get("g_id"),
                                "f_id": rec.get("f_id"),
                                "pair_reason": reason,
                                "pair_ok": rec.get("pair_ok"),
                                "stage": stage,
                            }
                        )

    return {
        "stages": dict(stage_ctr),
        "pair_reasons": dict(reason_ctr),
        "hf_device_counts": dict(device_ctr),
        "hf_records": int(hf_records),
        "hf_failures": int(hf_failures),
        "pair_failures": int(pair_failures),
        "bad_examples": bad_examples,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real stage3 spawn smoke test for pref_loss_coevo on a multi-device environment."
    )
    p.add_argument(
        "--devices",
        required=True,
        type=str,
        help='CSV device list, e.g. "cuda:0,cuda:1,cuda:2,cuda:3"',
    )
    p.add_argument("--work-dir", default="tmp_spawn_gpu_smoke", type=str)
    p.add_argument("--processes", default=4, type=int)
    p.add_argument("--pairing-budget", default=8, type=int)
    p.add_argument("--pop-g", default=6, type=int)
    p.add_argument("--pop-f", default=6, type=int)
    p.add_argument("--elite-g", default=2, type=int)
    p.add_argument("--elite-f", default=2, type=int)
    p.add_argument("--f1-steps", default=20, type=int)
    p.add_argument("--train-problem-size", default=20, type=int)
    p.add_argument("--valid-problem-sizes", nargs="+", default=[20], type=int)
    p.add_argument("--offline-train-size", default=64, type=int)
    p.add_argument("--offline-val-size", default=32, type=int)
    p.add_argument("--num-validation-episodes", default=16, type=int)
    p.add_argument("--train-batch-size", default=8, type=int)
    p.add_argument("--validation-batch-size", default=16, type=int)
    p.add_argument("--pomo-size", default=16, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--scratch-init-seed", default=12345, type=int)
    p.add_argument("--env-name", default="tsp", type=str)
    p.add_argument("--policy-name", default="pomo", type=str)
    p.add_argument("--reuse-baseline", action="store_true")
    p.add_argument("--ckpt-135", default="baseline/epoch_135.ckpt", type=str)
    p.add_argument("--ckpt-409", default="baseline/epoch_409.ckpt", type=str)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--keep-going-on-run-failure", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = _repo_root_dir()
    work_dir = _abs_from_repo_root(str(args.work_dir))
    offline_dir = work_dir / "offline_data"
    runs_dir = work_dir / "runs"
    baseline_json = work_dir / f"baseline_minitrain_{args.env_name}_K{int(args.f1_steps)}.json"
    config_path = work_dir / "spawn_stage3_smoke.yaml"
    report_path = work_dir / "report.json"

    devices = _parse_devices(args.devices)
    if not devices:
        raise SystemExit("--devices must not be empty")

    python_exe = sys.executable
    env = dict(os.environ)
    py_paths = [str(repo_root), str(repo_root / "PTP")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    work_dir.mkdir(parents=True, exist_ok=True)
    offline_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    runtime = _torch_runtime_info(python_exe, env=env, cwd=repo_root)
    print("[env]", json.dumps(runtime, indent=2), flush=True)

    wants_cuda = any(str(d).startswith("cuda") for d in devices)
    if wants_cuda and not bool(runtime.get("cuda_available")):
        msg = (
            "Requested CUDA devices but current torch runtime reports cuda_available=False. "
            "Use a CUDA-enabled PyTorch environment before running this script."
        )
        if args.allow_cpu:
            print(f"[warn] {msg}", flush=True)
        else:
            raise SystemExit(msg)
    if args.require_cuda and not bool(runtime.get("cuda_available")):
        raise SystemExit("--require-cuda was set but this Python environment cannot use CUDA")

    train_size = int(args.train_problem_size)
    valid_sizes = [int(v) for v in args.valid_problem_sizes]
    first_device = str(devices[0])

    offline_train = offline_dir / f"{args.env_name}{train_size}_train.pt"
    missing_offline = [offline_train] + [
        offline_dir / f"{args.env_name}{int(sz)}_val.pt" for sz in valid_sizes
    ]
    if any(not p.is_file() for p in missing_offline):
        _run(
            [
                python_exe,
                "scripts/precompute_offline_instances.py",
                "--env",
                str(args.env_name),
                "--sizes",
                *[str(v) for v in [train_size] + valid_sizes],
                "--train_size",
                str(int(args.offline_train_size)),
                "--val_size",
                str(int(args.offline_val_size)),
                "--seed",
                str(int(args.seed)),
                "--out_dir",
                str(offline_dir),
            ],
            env=env,
            cwd=repo_root,
        )

    baseline_cfg = {
        "mini_eval_path": str(baseline_json.relative_to(repo_root)),
        "checkpoints": [str(args.ckpt_135), str(args.ckpt_409)],
        "include_scratch": True,
        "multiseed_compare_enabled": False,
    }
    cfg: Dict[str, Any] = {
        "seed": int(args.seed),
        "output_root": str(runs_dir.relative_to(repo_root)),
        "generations": 1,
        "pop_g": int(args.pop_g),
        "pop_f": int(args.pop_f),
        "elite_g": int(args.elite_g),
        "elite_f": int(args.elite_f),
        "pairing_budget_per_gen": int(args.pairing_budget),
        "cheap_gate_on": True,
        "high_fidelity_on": True,
        "eval_stages": {
            "stage0_gate": True,
            "stage1_proxy": False,
            "stage2_micro_unroll": False,
            "stage3_high_fidelity": True,
        },
        "devices": [str(d) for d in devices],
        "mp": {
            "enabled": True,
            "processes": int(args.processes),
            "start_method": "spawn",
        },
        "high_fidelity_top_m": int(args.pairing_budget),
        "backend": "rl4co",
        "env_name": str(args.env_name),
        "policy_name": str(args.policy_name),
        "policy_kwargs": {"po4cops_compat": True},
        "generator_params": {
            "offline_train_path": str(offline_train.relative_to(repo_root)),
            "offline_val_paths": {
                str(int(sz)): str((offline_dir / f"{args.env_name}{int(sz)}_val.pt").relative_to(repo_root))
                for sz in valid_sizes
            },
        },
        "f1_steps": int(args.f1_steps),
        "hf_epochs": 0,
        "hf_instances_per_epoch": 0,
        "train_problem_size": int(train_size),
        "valid_problem_sizes": list(valid_sizes),
        "train_batch_size": int(args.train_batch_size),
        "num_validation_episodes": int(args.num_validation_episodes),
        "validation_batch_size": int(args.validation_batch_size),
        "pomo_size": int(args.pomo_size),
        "scratch_init_seed": int(args.scratch_init_seed),
        "device": str(first_device),
        "seed_with_po4cops_default": False,
        "improve_eps": 0.0,
        "improve_eps_calibration": {"enabled": False},
        "baseline": baseline_cfg,
        "builder_llm": {"enabled": False},
        "loss_llm": {"enabled": False},
    }
    _write_yaml(config_path, cfg)

    if not baseline_json.is_file() or not bool(args.reuse_baseline):
        _run(
            [
                python_exe,
                "scripts/eval_baseline_minitrain.py",
                "--config",
                str(config_path),
                "--K",
                str(int(args.f1_steps)),
                "--train_problem_size",
                str(int(train_size)),
                "--valid_problem_sizes",
                *[str(v) for v in valid_sizes],
                "--num_validation_episodes",
                str(int(args.num_validation_episodes)),
                "--train_batch_size",
                str(int(args.train_batch_size)),
                "--offline_train",
                str(offline_train.relative_to(repo_root)),
                *sum(
                    [
                        [
                            "--offline_val",
                            str(int(sz)),
                            str((offline_dir / f"{args.env_name}{int(sz)}_val.pt").relative_to(repo_root)),
                        ]
                        for sz in valid_sizes
                    ],
                    [],
                ),
                "--scratch_init_seed",
                str(int(args.scratch_init_seed)),
                "--ckpt_135",
                str(args.ckpt_135),
                "--ckpt_409",
                str(args.ckpt_409),
                "--out",
                str(baseline_json.relative_to(repo_root)),
            ],
            env=env,
            cwd=repo_root,
        )

    before_runs = {p.name for p in runs_dir.iterdir() if p.is_dir()}
    run_failed = False
    try:
        _run(
            [
                python_exe,
                "-u",
                "PTP/ptp_discovery/run_pref_loss_coevo.py",
                "--config",
                str(config_path),
            ],
            env=env,
            cwd=repo_root,
        )
    except subprocess.CalledProcessError as exc:
        run_failed = True
        if not args.keep_going_on_run_failure:
            raise SystemExit(exc.returncode) from exc

    after_runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    created_runs = [p for p in after_runs if p.name not in before_runs]
    run_dir = sorted(created_runs)[-1] if created_runs else _find_latest_run_dir(runs_dir)
    pairs_jsonl = run_dir / "pairs.jsonl"
    summary_json = run_dir / "summary.json"
    checkpoint_json = run_dir / "checkpoint.json"

    if not pairs_jsonl.is_file():
        raise SystemExit(f"Missing pairs.jsonl: {pairs_jsonl}")
    if not checkpoint_json.is_file():
        raise SystemExit(f"Missing checkpoint.json: {checkpoint_json}")

    pair_summary = _collect_pair_summary(pairs_jsonl)
    expected_hf = int(args.pairing_budget)
    hf_records = int(pair_summary["hf_records"])
    hf_failures = int(pair_summary["hf_failures"])
    pass_ok = (not run_failed) and hf_records >= expected_hf and hf_failures == 0

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_executable": str(python_exe),
        "runtime": runtime,
        "devices": devices,
        "work_dir": str(work_dir),
        "config_path": str(config_path),
        "baseline_json": str(baseline_json),
        "run_dir": str(run_dir),
        "summary_json": str(summary_json) if summary_json.is_file() else None,
        "checkpoint_json": str(checkpoint_json),
        "pairs_jsonl": str(pairs_jsonl),
        "run_failed": bool(run_failed),
        "expected_hf_records": int(expected_hf),
        "pass": bool(pass_ok),
        "pair_summary": pair_summary,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[report]", json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not pass_ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
