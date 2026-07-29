from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from copy import deepcopy


RUN_ID = "tsp100-main-record-hf1-replay-weighting-r3-20260728-2320"
SOURCE_COMMIT = "8ee9d8e551d9b0fc9251c1eeb8919b79425a079b"
SOURCE = Path("/data1/gushengda/codex_worktrees/tsp100-main-record-hf1-replay-src")
PREDECESSOR = Path(
    "/data1/gushengda/eam-rl4co_runs/"
    "tsp100-main-record-hf1-replay-r3-20260728-2135"
)
ORIGINAL_RUN = Path(
    "/data1/gushengda/eam-rl4co/runs/"
    "pref_builder_weight_search_tsp100/20260401-223759"
)
BASELINE_CACHE = (
    PREDECESSOR
    / "baseline_cache"
    / "baseline_minitrain_tsp100_epoch1_inst100000.json"
)
PYTHON = "/data1/gushengda/anaconda3/envs/rlco1/bin/python"

RECORD_SETTERS = [
    {
        "name": "01_g000_000_59b6b66b",
        "g_id": "g000_000_59b6b66b",
        "generation": 0,
        "pair_index": 0,
        "original_payload": ORIGINAL_RUN
        / "hf_subprocess/gen000_pair000_cuda_0/payload.json",
    },
    {
        "name": "02_g000_002_15cfdaab",
        "g_id": "g000_002_15cfdaab",
        "generation": 0,
        "pair_index": 2,
        "original_payload": ORIGINAL_RUN
        / "hf_subprocess/gen000_pair002_cuda_2/payload.json",
    },
    {
        "name": "03_g001_009_7723440a",
        "g_id": "g001_009_7723440a",
        "generation": 1,
        "pair_index": 3,
        "original_payload": ORIGINAL_RUN
        / "hf_subprocess/gen001_pair003_cuda_2/payload.json",
    },
    {
        "name": "04_g001_004_a9cbc4ab",
        "g_id": "g001_004_a9cbc4ab",
        "generation": 1,
        "pair_index": 4,
        "original_payload": ORIGINAL_RUN
        / "hf_subprocess/gen001_pair004_cuda_3/payload.json",
    },
    {
        "name": "05_g003_005_57a9c8c8",
        "g_id": "g003_005_57a9c8c8",
        "generation": 3,
        "pair_index": 0,
        "original_payload": ORIGINAL_RUN
        / "hf_subprocess/gen003_pair000_cuda_0/payload.json",
    },
]


run = Path(__file__).resolve().parent
status_path = run / "runtime_status.json"
manifest_path = run / "manifest.json"
tasks_path = run / "tasks.json"

status = {
    "status": "waiting",
    "pid": os.getpid(),
    "started_at": time.time(),
    "stage": "waiting_for_loss_replay",
    "current": None,
    "completed": [],
}


def atomic_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def save_status() -> None:
    status["heartbeat_at"] = time.time()
    atomic_json(status_path, status)


def save_manifest(state: str) -> None:
    manifest = {
        "run_id": RUN_ID,
        "host": "g51",
        "state": state,
        "source_commit": SOURCE_COMMIT,
        "source_worktree": str(SOURCE),
        "original_run": str(ORIGINAL_RUN),
        "predecessor_run": str(PREDECESSOR),
        "physical_gpu": 7,
        "protocol": {
            "record_setters_only": True,
            "stage": "weighting",
            "hf_epochs_per_init": 1,
            "init_sources": ["scratch", "ckpt_135"],
            "hf_instances_per_epoch": 100000,
            "num_validation_episodes": 10000,
            "aggregation": (
                "mean(candidate-baseline across scratch and ckpt_135)"
            ),
            "shared_baseline_cache": str(BASELINE_CACHE),
        },
        "record_setters": [
            {
                **{k: v for k, v in spec.items() if k != "original_payload"},
                "original_payload": str(spec["original_payload"]),
            }
            for spec in RECORD_SETTERS
        ],
        "result_branch": f"results/{RUN_ID}/g51",
        "pid": os.getpid(),
        "command": "CUDA_VISIBLE_DEVICES=7 python -u launch_weighting_replay.py",
        "launcher_log": str(run / "launcher.log"),
    }
    atomic_json(manifest_path, manifest)


def fail(reason: str, detail: object | None = None) -> None:
    status.update(
        {
            "status": "failed",
            "stage": "failed",
            "failure": {"reason": reason, "detail": detail},
            "finished_at": time.time(),
        }
    )
    save_status()
    save_manifest("failed")
    print("weighting_replay_failed", reason, detail, flush=True)
    raise SystemExit(1)


def prepare_payload(spec: dict[str, object]) -> dict[str, object]:
    original_payload = Path(spec["original_payload"])
    if not original_payload.is_file():
        fail("missing_original_payload", str(original_payload))
    payload = json.loads(original_payload.read_text(encoding="utf-8"))
    actual_g_id = (payload.get("g_entry") or {}).get("id")
    if actual_g_id != spec["g_id"]:
        fail(
            "original_payload_id_mismatch",
            {
                "path": str(original_payload),
                "expected": spec["g_id"],
                "actual": actual_g_id,
            },
        )

    task_dir = run / "tasks" / str(spec["name"])
    task_dir.mkdir(parents=True, exist_ok=True)
    cfg = payload["cfg_yaml"]
    matched_cfg = json.loads(
        (PREDECESSOR / "baseline_config.json").read_text(encoding="utf-8")
    )
    matched_fields = (
        "backend", "env_name", "generator_params", "policy_name",
        "policy_kwargs", "rollout_strategy", "objective_sign",
        "train_problem_size", "valid_problem_sizes", "train_batch_size",
        "num_validation_episodes", "validation_batch_size", "pomo_size",
        "learning_rate", "weight_decay", "scratch_init_seed",
        "hf_instances_per_epoch", "f1_steps",
    )
    for field in matched_fields:
        cfg[field] = deepcopy(matched_cfg[field])
    cfg["baseline"] = deepcopy(matched_cfg["baseline"])
    cfg["hf_epochs"] = 1
    cfg["scratch_hf_epochs"] = 0
    cfg["warmstart_hf_epochs"] = 0
    cfg["budgets"]["hf_epochs"] = 1
    cfg["devices"] = ["cuda:0"]
    cfg["mp"]["enabled"] = False
    cfg["mp"]["processes"] = 1
    cfg.setdefault("stage3_early_prune", {})["enabled"] = False
    cache_path = str(BASELINE_CACHE)
    cfg["baseline"]["mini_eval_paths"]["epoch1_inst100000"] = cache_path
    for scenario in cfg["baseline"].get("scenarios", []):
        scenario["baseline"]["mini_eval_paths"][
            "epoch1_inst100000"
        ] = cache_path

    payload["eval_budget_signature"] = "record-replay-hf1-per-init-weighting"
    payload["run_dir"] = str(task_dir)
    payload_path = run / "payloads" / f"{spec['name']}.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload_path, payload)
    return {
        "name": spec["name"],
        "g_id": spec["g_id"],
        "f_id": (payload.get("f_entry") or {}).get("id"),
        "generation": spec["generation"],
        "pair_index": spec["pair_index"],
        "payload": str(payload_path),
        "result": str(task_dir / "result.json"),
        "original_payload": str(original_payload),
    }


(run / "launcher.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
save_manifest("waiting")
save_status()
print(
    f"weighting_replay_wait run={run} predecessor={PREDECESSOR}",
    flush=True,
)

while True:
    try:
        predecessor_status = json.loads(
            (PREDECESSOR / "runtime_status.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        status["wait_detail"] = f"predecessor_status_unavailable: {exc}"
        save_status()
        time.sleep(30)
        continue
    predecessor_state = predecessor_status.get("status")
    status["predecessor_status"] = predecessor_state
    status["predecessor_stage"] = predecessor_status.get("stage")
    status["predecessor_current"] = predecessor_status.get("current")
    save_status()
    if predecessor_state == "completed":
        break
    if predecessor_state == "failed":
        fail("predecessor_failed", predecessor_status.get("failure"))
    time.sleep(30)

if not BASELINE_CACHE.is_file():
    fail("shared_baseline_cache_missing", str(BASELINE_CACHE))

actual_commit = subprocess.check_output(
    ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
    text=True,
).strip()
if actual_commit != SOURCE_COMMIT:
    fail(
        "source_commit_mismatch",
        {"expected": SOURCE_COMMIT, "actual": actual_commit},
    )

baseline_data = json.loads(BASELINE_CACHE.read_text(encoding="utf-8"))
per_init_baseline = baseline_data.get("per_init") or {}
required = {"scratch", "ckpt_135"}
if not required <= set(per_init_baseline):
    fail("baseline_missing_init", sorted(per_init_baseline))
baseline = {
    key: float(per_init_baseline[key]["aggregated_objective"])
    for key in sorted(required)
}

tasks = [prepare_payload(spec) for spec in RECORD_SETTERS]
atomic_json(tasks_path, tasks)
status.update(
    {
        "status": "running",
        "stage": "candidates",
        "baseline": baseline,
        "current": None,
    }
)
save_manifest("running")
save_status()
print("weighting_replay_start", json.dumps(baseline), flush=True)

for task in tasks:
    name = task["name"]
    status["current"] = name
    status["task_started_at"] = time.time()
    save_status()
    print(
        f"task_start name={name} g_id={task['g_id']} f_id={task['f_id']}",
        flush=True,
    )
    task_dir = run / "tasks" / name
    subprocess_log = task_dir / "subprocess.log"
    command = [
        PYTHON,
        "-u",
        str(SOURCE / "PTP/ptp_discovery/run_hf_pair_eval.py"),
        "--payload",
        task["payload"],
        "--result",
        task["result"],
    ]
    with subprocess_log.open("w", encoding="utf-8") as output:
        returncode = subprocess.run(
            command,
            cwd=str(SOURCE),
            stdout=output,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        ).returncode
    result_path = Path(task["result"])
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else None
    )
    if (
        returncode != 0
        or not isinstance(result, dict)
        or not result.get("pair_ok")
    ):
        fail(
            "candidate_failed",
            {
                "name": name,
                "returncode": returncode,
                "pair_reason": (
                    result.get("pair_reason")
                    if isinstance(result, dict)
                    else "missing_result"
                ),
            },
        )
    per_init = (result.get("fitness") or {}).get("per_init") or {}
    if not required <= set(per_init):
        fail(
            "candidate_missing_init",
            {"name": name, "keys": sorted(per_init)},
        )
    objectives = {
        key: float(per_init[key]["obj_cand"]) for key in sorted(required)
    }
    deltas = {
        key: objectives[key] - baseline[key] for key in sorted(required)
    }
    mean_delta = sum(deltas.values()) / len(deltas)
    item = {
        "name": name,
        "g_id": task["g_id"],
        "f_id": task["f_id"],
        "objectives": objectives,
        "deltas": deltas,
        "mean_delta": mean_delta,
        "reported_score": result.get("score"),
        "result": str(result_path),
        "subprocess_log": str(subprocess_log),
    }
    status["completed"].append(item)
    save_status()
    print("task_done", json.dumps(item, ensure_ascii=False), flush=True)

metrics = {"baseline": baseline, "candidates": status["completed"]}
atomic_json(run / "metrics.json", metrics)
status.update(
    {
        "status": "completed",
        "stage": "done",
        "current": None,
        "finished_at": time.time(),
    }
)
save_status()
save_manifest("completed")
print("weighting_replay_completed", json.dumps(metrics), flush=True)
