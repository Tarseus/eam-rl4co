from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import yaml


G_REF_ID = "g_ref"


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return dict(data)


def _write_yaml(path: Path, cfg: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, allow_unicode=False)
    os.replace(tmp, path)


def _budget_generations(cfg: Mapping[str, Any]) -> int:
    budgets = cfg.get("budgets", {}) or {}
    if not isinstance(budgets, Mapping):
        budgets = {}
    raw = budgets.get("generations", cfg.get("generations", 1))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _output_root(repo: Path, cfg: Mapping[str, Any]) -> Path:
    raw = str(cfg.get("output_root", "runs/pref_loss_coevo") or "runs/pref_loss_coevo")
    p = Path(raw)
    if not p.is_absolute():
        p = repo / p
    return p


def _latest_run_dir(repo: Path, cfg_path: Path) -> Path | None:
    cfg = _load_yaml(cfg_path)
    root = _output_root(repo, cfg)
    if not root.is_dir():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name, reverse=True)[0]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _best_from_payload(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    best = payload.get("best_so_far")
    return best if isinstance(best, Mapping) else None


def _record_is_improvement(rec: Mapping[str, Any]) -> bool:
    if rec.get("better_than_incumbent") is True:
        if rec.get("pair_ok") is False:
            return False
        return True
    return False


def _score_beats_target(
    score: Any,
    *,
    target_score: float | None,
    metric_mode: str,
    eps: float,
) -> bool:
    if target_score is None:
        return True
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return False
    if metric_mode == "maximize":
        return score_f > float(target_score) + float(eps)
    return score_f < float(target_score) - float(eps)


def _run_has_improvement(
    run_dir: Path | None,
    *,
    target_score: float | None,
    require_better_than_target: bool,
    metric_mode: str,
    eps: float,
) -> tuple[bool, str]:
    if run_dir is None:
        return False, "no run dir yet"

    best_sources = [run_dir / "summary.json", run_dir / "checkpoint.json"]
    for path in best_sources:
        payload = _load_json(path)
        best = _best_from_payload(payload)
        if not isinstance(best, Mapping):
            continue
        builder_id = str(best.get("builder_id", "") or "")
        try:
            generation = int(best.get("generation", -1) or -1)
        except (TypeError, ValueError):
            generation = -1
        if builder_id and builder_id != G_REF_ID and generation >= 0:
            score = best.get("score")
            if _score_beats_target(score, target_score=target_score, metric_mode=metric_mode, eps=eps):
                return True, f"{path.name}: best builder={builder_id} gen={generation} score={score}"
            if require_better_than_target:
                return False, (
                    f"{path.name}: best builder={builder_id} gen={generation} score={score} "
                    f"has not beaten target={target_score}"
                )
            return True, f"{path.name}: best builder={builder_id} gen={generation} score={score}"

    pairs_path = run_dir / "pairs.jsonl"
    if pairs_path.is_file():
        try:
            with pairs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, Mapping) and _record_is_improvement(rec):
                        gid = rec.get("g_id")
                        fid = rec.get("f_id")
                        score = rec.get("final_score", rec.get("score"))
                        gen = rec.get("generation")
                        if not _score_beats_target(
                            score,
                            target_score=target_score,
                            metric_mode=metric_mode,
                            eps=eps,
                        ):
                            continue
                        return True, f"pairs.jsonl: improved pair=({gid},{fid}) gen={gen} score={score}"
        except Exception as exc:
            return False, f"failed reading pairs.jsonl: {exc}"

    return False, f"no improvement in {run_dir}"


def _run_success_command(args: argparse.Namespace, run_dir: Path | None, reason: str) -> None:
    if not args.success_command:
        return
    if run_dir is None:
        raise RuntimeError("success_command requested but no run_dir is available")
    best_pair = run_dir / "best_pair.json"
    replacements = {
        "run_dir": str(run_dir),
        "best_pair": str(best_pair),
        "config": str(args.config),
        "job_name": str(args.job_name),
        "reason": str(reason),
    }
    command = str(args.success_command)
    for key, value in replacements.items():
        command = command.replace("{" + key + "}", shlex.quote(value))

    log_dir = args.repo / "logs" / "codex_remote"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{args.job_name}_success_{stamp}.log"
    _log(f"launching success command: {command}")
    _log(f"success log: {log_path}")
    with log_path.open("ab", buffering=0) as f:
        proc = subprocess.Popen(
            command,
            cwd=str(args.repo),
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _log(f"success command pid={proc.pid}")


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int, *, timeout_s: float = 10.0) -> None:
    if pid <= 0 or not _pid_running(pid):
        return
    _log(f"terminating active search pid={pid}")
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            _log(f"failed to terminate pid={pid}: {exc}")
            return

    deadline = time.time() + max(0.1, float(timeout_s))
    while time.time() < deadline:
        if not _pid_running(pid):
            return
        time.sleep(0.25)

    if _pid_running(pid):
        _log(f"pid={pid} still running after SIGTERM; sending SIGKILL")
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception as exc:
                _log(f"failed to kill pid={pid}: {exc}")


def _next_generation(run_dir: Path | None) -> int:
    if run_dir is None:
        return 0
    payload = _load_json(run_dir / "checkpoint.json")
    if not isinstance(payload, Mapping):
        return 0
    try:
        return max(0, int(payload.get("next_generation", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _ensure_budget_extends(cfg_path: Path, repo: Path, chunk: int) -> int:
    cfg = _load_yaml(cfg_path)
    run_dir = _latest_run_dir(repo, cfg_path)
    next_gen = _next_generation(run_dir)
    current_budget = _budget_generations(cfg)
    target_budget = max(current_budget, next_gen + max(1, int(chunk)))
    if target_budget <= current_budget and next_gen >= current_budget:
        target_budget = current_budget + max(1, int(chunk))
    if target_budget != current_budget:
        budgets = dict(cfg.get("budgets", {}) or {})
        budgets["generations"] = int(target_budget)
        cfg["budgets"] = budgets
        _write_yaml(cfg_path, cfg)
        _log(f"extended budget: {current_budget} -> {target_budget} (next_generation={next_gen})")
    else:
        _log(f"budget already sufficient: {current_budget} (next_generation={next_gen})")
    return int(target_budget)


def _launch_search(args: argparse.Namespace) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo) + os.pathsep + str(args.repo / "PTP") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("LOG_TZ", "Asia/Shanghai")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    log_dir = args.repo / "logs" / "codex_remote"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    child_log = log_dir / f"{args.job_name}_watch_child_{stamp}.log"
    cmd = [
        str(args.python),
        "-u",
        "PTP/ptp_discovery/run_pref_loss_coevo.py",
        "--config",
        str(args.config),
    ]
    _log(f"launching child: {' '.join(cmd)}")
    _log(f"child log: {child_log}")
    f = child_log.open("ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(args.repo),
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    proc._watch_log_handle = f  # type: ignore[attr-defined]
    return proc


def _close_proc_log(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    handle = getattr(proc, "_watch_log_handle", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue a target weighting search until it beats the incumbent.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--job-name", type=str, required=True)
    parser.add_argument("--existing-pid", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", type=str, default="")
    parser.add_argument("--chunk-generations", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--target-score", type=float, default=None)
    parser.add_argument("--metric-mode", type=str, default="minimize", choices=["minimize", "maximize"])
    parser.add_argument("--target-eps", type=float, default=1e-9)
    parser.add_argument("--require-better-than-target", action="store_true")
    parser.add_argument(
        "--success-command",
        type=str,
        default="",
        help="Optional shell command to launch once target is achieved. Supports {run_dir}, {best_pair}, {config}.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.repo = args.repo.resolve()
    if not args.config.is_absolute():
        args.config = args.repo / args.config
    args.config = args.config.resolve()
    args.python = args.python.resolve()

    _log(f"watchdog started job={args.job_name} config={args.config}")
    current_pid = int(args.existing_pid or 0)
    child: subprocess.Popen[Any] | None = None

    while True:
        run_dir = _latest_run_dir(args.repo, args.config)
        ok, reason = _run_has_improvement(
            run_dir,
            target_score=args.target_score,
            require_better_than_target=bool(args.require_better_than_target),
            metric_mode=str(args.metric_mode),
            eps=float(args.target_eps),
        )
        _log(f"improvement_check ok={ok} reason={reason}")
        if ok:
            _terminate_pid(current_pid)
            _run_success_command(args, run_dir, reason)
            _log("target achieved; watchdog exiting")
            return 0

        if current_pid and _pid_running(current_pid):
            _log(f"observing existing pid={current_pid}")
            time.sleep(max(5, int(args.poll_seconds)))
            continue

        if child is not None:
            rc = child.poll()
            if rc is None:
                current_pid = int(child.pid)
                _log(f"observing child pid={current_pid}")
                time.sleep(max(5, int(args.poll_seconds)))
                continue
            _log(f"child exited rc={rc}")
            _close_proc_log(child)
            child = None
            current_pid = 0

        _ensure_budget_extends(args.config, args.repo, int(args.chunk_generations))
        child = _launch_search(args)
        current_pid = int(child.pid)
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _log("watchdog interrupted")
        raise
