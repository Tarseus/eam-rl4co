from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import classical_solver_benchmark as bench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight constructive and metaheuristic baselines for final result tables."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["tsp50", "tsp100", "cvrp50", "cvrp100", "ffsp50", "ffsp100", "jssp10x10", "jssp15x15"],
        choices=list(bench.SCENARIOS.keys()),
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "heuristic_baseline_supplement")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--require-test-data", action="store_true")
    parser.add_argument("--jssp-data-dir", type=Path, default=None)
    parser.add_argument("--ffsp-data-file", type=Path, default=None)
    parser.add_argument(
        "--tsp-heuristics",
        nargs="+",
        default=["nn_2opt", "nearest_insertion"],
        choices=["nn_2opt", "nearest_insertion"],
    )
    parser.add_argument(
        "--two-opt-passes",
        type=int,
        default=2,
        help="Maximum improvement passes for TSP 2-opt. Keeps the simple heuristic full-benchmark friendly.",
    )
    parser.add_argument(
        "--cvrp-heuristics",
        nargs="+",
        default=["nearest_feasible", "sweep", "savings"],
        choices=["nearest_feasible", "sweep", "savings"],
    )
    parser.add_argument(
        "--scheduling-heuristics",
        nargs="+",
        default=["ga", "pso"],
        choices=["ga", "pso"],
    )
    parser.add_argument("--ga-population", type=int, default=32)
    parser.add_argument("--ga-generations", type=int, default=50)
    parser.add_argument("--pso-particles", type=int, default=32)
    parser.add_argument("--pso-iterations", type=int, default=50)
    parser.add_argument("--mutation-sigma", type=float, default=0.15)
    return parser.parse_args()


def _scaled_euc_dist_matrix(coords: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.asarray(coords, dtype=np.float64) * 100_000.0)
    diff = scaled[:, None, :] - scaled[None, :, :]
    return np.floor(np.sqrt(np.sum(diff * diff, axis=-1)) + 0.5).astype(np.int64)


def _tour_cost(dist: np.ndarray, tour: list[int]) -> int:
    total = 0
    for idx, node in enumerate(tour):
        total += int(dist[node, tour[(idx + 1) % len(tour)]])
    return total


def _two_opt(dist: np.ndarray, tour: list[int], max_passes: int) -> list[int]:
    n = len(tour)
    for _ in range(max(0, int(max_passes))):
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            for k in range(i + 2, n if i > 0 else n - 1):
                c, d = tour[k], tour[(k + 1) % n]
                if int(dist[a, c] + dist[b, d]) < int(dist[a, b] + dist[c, d]):
                    tour[i + 1 : k + 1] = reversed(tour[i + 1 : k + 1])
                    improved = True
        if not improved:
            break
    return tour


def tsp_nn_2opt(coords: np.ndarray, two_opt_passes: int = 2) -> int:
    dist = _scaled_euc_dist_matrix(coords)
    n = int(dist.shape[0])
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda node: (int(dist[current, node]), node))
        unvisited.remove(nxt)
        tour.append(nxt)
        current = nxt
    return _tour_cost(dist, _two_opt(dist, tour, two_opt_passes))


def tsp_nearest_insertion(coords: np.ndarray, two_opt_passes: int = 2) -> int:
    dist = _scaled_euc_dist_matrix(coords)
    n = int(dist.shape[0])
    first, second = min(
        ((i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda pair: (int(dist[pair[0], pair[1]]), pair[0], pair[1]),
    )
    tour = [first, second]
    unvisited = set(range(n)) - {first, second}
    while unvisited:
        node = min(unvisited, key=lambda item: (min(int(dist[item, t]) for t in tour), item))
        best_pos = 0
        best_delta = math.inf
        for pos in range(len(tour)):
            left = tour[pos]
            right = tour[(pos + 1) % len(tour)]
            delta = int(dist[left, node] + dist[node, right] - dist[left, right])
            if delta < best_delta:
                best_delta = delta
                best_pos = pos + 1
        tour.insert(best_pos, node)
        unvisited.remove(node)
    return _tour_cost(dist, _two_opt(dist, tour, two_opt_passes))


def _cvrp_distance(instance: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    coords = np.concatenate(
        [np.asarray(instance["depot"], dtype=np.float32)[None, :], np.asarray(instance["locs"], dtype=np.float32)],
        axis=0,
    )
    demand = np.concatenate([[0.0], np.asarray(instance["demand"], dtype=np.float32)])
    return _scaled_euc_dist_matrix(coords), demand, float(instance["capacity"])


def _routes_cost(dist: np.ndarray, routes: list[list[int]]) -> int:
    total = 0
    for route in routes:
        prev = 0
        for node in route:
            total += int(dist[prev, node])
            prev = node
        total += int(dist[prev, 0])
    return total


def cvrp_nearest_feasible(instance: dict[str, np.ndarray]) -> int:
    dist, demand, capacity = _cvrp_distance(instance)
    unserved = set(range(1, int(dist.shape[0])))
    routes: list[list[int]] = []
    while unserved:
        route: list[int] = []
        load = 0.0
        current = 0
        while True:
            feasible = [node for node in unserved if load + float(demand[node]) <= capacity]
            if not feasible:
                break
            nxt = min(feasible, key=lambda node: (int(dist[current, node]), node))
            route.append(nxt)
            unserved.remove(nxt)
            load += float(demand[nxt])
            current = nxt
        routes.append(route)
    return _routes_cost(dist, routes)


def cvrp_sweep(instance: dict[str, np.ndarray]) -> int:
    dist, demand, capacity = _cvrp_distance(instance)
    depot = np.asarray(instance["depot"], dtype=np.float64)
    locs = np.asarray(instance["locs"], dtype=np.float64)
    angles = np.arctan2(locs[:, 1] - depot[1], locs[:, 0] - depot[0])
    order = [int(idx) + 1 for idx in np.argsort(angles)]
    routes: list[list[int]] = []
    route: list[int] = []
    load = 0.0
    for node in order:
        node_demand = float(demand[node])
        if route and load + node_demand > capacity:
            routes.append(route)
            route = []
            load = 0.0
        route.append(node)
        load += node_demand
    if route:
        routes.append(route)
    return _routes_cost(dist, routes)


def cvrp_savings(instance: dict[str, np.ndarray]) -> int:
    dist, demand, capacity = _cvrp_distance(instance)
    n = int(dist.shape[0]) - 1
    routes: dict[int, list[int]] = {node: [node] for node in range(1, n + 1)}
    loads: dict[int, float] = {node: float(demand[node]) for node in range(1, n + 1)}
    route_of = {node: node for node in range(1, n + 1)}
    savings = [
        (int(dist[0, i] + dist[0, j] - dist[i, j]), i, j)
        for i in range(1, n + 1)
        for j in range(i + 1, n + 1)
    ]
    savings.sort(reverse=True)
    for _, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj or loads[ri] + loads[rj] > capacity:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        merged: list[int] | None = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            merged = route_j + route_i
        elif route_i[0] == i and route_j[0] == j:
            merged = list(reversed(route_i)) + route_j
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + list(reversed(route_j))
        if merged is None:
            continue
        routes[ri] = merged
        loads[ri] += loads[rj]
        del routes[rj]
        del loads[rj]
        for node in merged:
            route_of[node] = ri
    return _routes_cost(dist, list(routes.values()))


def _ffsp_schedule_by_keys(run_time: np.ndarray, num_stage: int, num_machine: int, keys: np.ndarray) -> int:
    order = [int(idx) for idx in np.argsort(keys, kind="mergesort")]
    return ffsp_sequence_makespan(run_time, num_stage, num_machine, order)


def ffsp_sequence_makespan(
    run_time: np.ndarray,
    num_stage: int,
    num_machine: int,
    job_sequence: list[int],
) -> int:
    machine_ready = np.zeros((num_stage, num_machine), dtype=int)
    job_ready = np.zeros(run_time.shape[0], dtype=int)
    for job in job_sequence:
        for stage in range(num_stage):
            durations = run_time[job, stage * num_machine : (stage + 1) * num_machine]
            completion_times = np.maximum(job_ready[job], machine_ready[stage]) + durations
            local_machine = int(np.argmin(completion_times))
            completion = int(completion_times[local_machine])
            machine_ready[stage, local_machine] = completion
            job_ready[job] = completion
    return int(job_ready.max())


def _jssp_schedule_by_keys(jobs: list[list[tuple[int, int]]], num_machines: int, keys: np.ndarray) -> int:
    machine_ready = [0] * num_machines
    job_ready = [0] * len(jobs)
    next_op = [0] * len(jobs)
    total_ops = sum(len(job) for job in jobs)
    scheduled_ops = 0
    while scheduled_ops < total_ops:
        ready_jobs = [job for job in range(len(jobs)) if next_op[job] < len(jobs[job])]
        chosen = min(
            ready_jobs,
            key=lambda job: (
                max(job_ready[job], machine_ready[jobs[job][next_op[job]][0]]),
                float(keys[job]),
                job,
            ),
        )
        machine, duration = jobs[chosen][next_op[chosen]]
        start = max(job_ready[chosen], machine_ready[machine])
        end = start + int(duration)
        job_ready[chosen] = end
        machine_ready[machine] = end
        next_op[chosen] += 1
        scheduled_ops += 1
    return max(job_ready)


def _ga_minimize(
    dim: int,
    evaluate,
    rng: np.random.Generator,
    population: int,
    generations: int,
    mutation_sigma: float,
) -> int:
    population = max(4, int(population))
    generations = max(1, int(generations))
    pop = rng.uniform(size=(population, dim))
    scores = np.array([evaluate(ind) for ind in pop], dtype=np.float64)
    elite_count = max(1, population // 8)
    for _ in range(generations):
        elite_idx = np.argsort(scores)[:elite_count]
        children = [pop[idx].copy() for idx in elite_idx]
        while len(children) < population:
            contenders = rng.integers(0, population, size=6)
            p1 = pop[contenders[:3][np.argmin(scores[contenders[:3]])]]
            p2 = pop[contenders[3:][np.argmin(scores[contenders[3:]])]]
            mask = rng.random(dim) < 0.5
            child = np.where(mask, p1, p2)
            child = child + rng.normal(0.0, mutation_sigma, size=dim)
            children.append(np.mod(child, 1.0))
        pop = np.asarray(children)
        scores = np.array([evaluate(ind) for ind in pop], dtype=np.float64)
    return int(scores.min())


def _pso_minimize(
    dim: int,
    evaluate,
    rng: np.random.Generator,
    particles: int,
    iterations: int,
) -> int:
    particles = max(4, int(particles))
    iterations = max(1, int(iterations))
    pos = rng.uniform(size=(particles, dim))
    vel = rng.normal(0.0, 0.1, size=(particles, dim))
    personal = pos.copy()
    personal_scores = np.array([evaluate(ind) for ind in pos], dtype=np.float64)
    global_best = personal[int(np.argmin(personal_scores))].copy()
    global_score = float(personal_scores.min())
    for _ in range(iterations):
        r1 = rng.random(size=(particles, dim))
        r2 = rng.random(size=(particles, dim))
        vel = 0.72 * vel + 1.49 * r1 * (personal - pos) + 1.49 * r2 * (global_best - pos)
        pos = np.mod(pos + vel, 1.0)
        scores = np.array([evaluate(ind) for ind in pos], dtype=np.float64)
        improved = scores < personal_scores
        personal[improved] = pos[improved]
        personal_scores[improved] = scores[improved]
        if float(personal_scores.min()) < global_score:
            global_score = float(personal_scores.min())
            global_best = personal[int(np.argmin(personal_scores))].copy()
    return int(global_score)


def _run_tsp_task(
    solver: str,
    cfg: bench.ScenarioConfig,
    coords: np.ndarray,
    idx: int,
    two_opt_passes: int,
) -> bench.PerInstanceResult:
    t0 = time.perf_counter()
    if solver == "nn_2opt":
        objective = tsp_nn_2opt(coords, two_opt_passes)
    elif solver == "nearest_insertion":
        objective = tsp_nearest_insertion(coords, two_opt_passes)
    else:
        raise ValueError(f"Unsupported TSP heuristic: {solver}")
    return bench.PerInstanceResult(cfg.name, solver, f"{cfg.name}_{idx:05d}", "ok", float(objective), time.perf_counter() - t0)


def _run_cvrp_task(solver: str, cfg: bench.ScenarioConfig, instance: dict[str, np.ndarray], idx: int) -> bench.PerInstanceResult:
    t0 = time.perf_counter()
    if solver == "nearest_feasible":
        objective = cvrp_nearest_feasible(instance)
    elif solver == "sweep":
        objective = cvrp_sweep(instance)
    elif solver == "savings":
        objective = cvrp_savings(instance)
    else:
        raise ValueError(f"Unsupported CVRP heuristic: {solver}")
    return bench.PerInstanceResult(cfg.name, solver, f"{cfg.name}_{idx:05d}", "ok", float(objective), time.perf_counter() - t0)


def _run_ffsp_task(
    solver: str,
    cfg: bench.ScenarioConfig,
    run_time: np.ndarray,
    idx: int,
    seed: int,
    args: argparse.Namespace,
) -> bench.PerInstanceResult:
    assert cfg.ffsp_stages is not None and cfg.ffsp_machines is not None
    rng = np.random.default_rng(seed + 100_003 * idx + (0 if solver == "ga" else 10_000_000))
    evaluate = lambda keys: _ffsp_schedule_by_keys(run_time, cfg.ffsp_stages, cfg.ffsp_machines, keys)
    t0 = time.perf_counter()
    if solver == "ga":
        objective = _ga_minimize(run_time.shape[0], evaluate, rng, args.ga_population, args.ga_generations, args.mutation_sigma)
    elif solver == "pso":
        objective = _pso_minimize(run_time.shape[0], evaluate, rng, args.pso_particles, args.pso_iterations)
    else:
        raise ValueError(f"Unsupported FFSP heuristic: {solver}")
    return bench.PerInstanceResult(cfg.name, solver, f"{cfg.name}_{idx:05d}", "ok", float(objective), time.perf_counter() - t0)


def _run_jssp_task(
    solver: str,
    cfg: bench.ScenarioConfig,
    path: Path,
    idx: int,
    seed: int,
    args: argparse.Namespace,
) -> bench.PerInstanceResult:
    num_machines, jobs = bench.parse_jssp_file(path)
    rng = np.random.default_rng(seed + 200_003 * idx + (0 if solver == "ga" else 10_000_000))
    evaluate = lambda keys: _jssp_schedule_by_keys(jobs, num_machines, keys)
    t0 = time.perf_counter()
    if solver == "ga":
        objective = _ga_minimize(len(jobs), evaluate, rng, args.ga_population, args.ga_generations, args.mutation_sigma)
    elif solver == "pso":
        objective = _pso_minimize(len(jobs), evaluate, rng, args.pso_particles, args.pso_iterations)
    else:
        raise ValueError(f"Unsupported JSSP heuristic: {solver}")
    return bench.PerInstanceResult(cfg.name, solver, path.name, "ok", float(objective), time.perf_counter() - t0)


def _execute(solver: str, tasks: list[tuple], worker_fn, workers: int) -> tuple[list[bench.PerInstanceResult], bench.SolverBatchSummary]:
    worker_count = max(1, min(int(workers), len(tasks) if tasks else 1))
    t0 = time.perf_counter()
    rows: list[bench.PerInstanceResult] = []
    if worker_count == 1:
        for task in tasks:
            rows.append(worker_fn(*task))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(worker_fn, *task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row.instance_id)
    return rows, bench.SolverBatchSummary(
        scenario=rows[0].scenario if rows else "",
        solver=solver,
        wall_clock_s=time.perf_counter() - t0,
        workers=worker_count,
        solver_threads=1,
    )


def benchmark_scenario(cfg: bench.ScenarioConfig, args: argparse.Namespace) -> tuple[list[bench.PerInstanceResult], list[bench.SolverBatchSummary]]:
    rows: list[bench.PerInstanceResult] = []
    summaries: list[bench.SolverBatchSummary] = []
    if cfg.problem == "tsp":
        instances = bench._load_tsp_instances(cfg, args.seed, args.max_instances, args.require_test_data)
        for solver in args.tsp_heuristics:
            tasks = [(solver, cfg, coords, idx, args.two_opt_passes) for idx, coords in enumerate(instances)]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            solver_rows, summary = _execute(solver, tasks, _run_tsp_task, args.workers)
            rows.extend(solver_rows)
            summaries.append(summary)
    elif cfg.problem == "cvrp":
        instances = bench._load_cvrp_instances(cfg, args.seed, args.max_instances, args.require_test_data)
        for solver in args.cvrp_heuristics:
            tasks = [(solver, cfg, instance, idx) for idx, instance in enumerate(instances)]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            solver_rows, summary = _execute(solver, tasks, _run_cvrp_task, args.workers)
            rows.extend(solver_rows)
            summaries.append(summary)
    elif cfg.problem == "ffsp":
        instances = bench._load_ffsp_instances(cfg, args.seed, args.max_instances, args.ffsp_data_file)
        for solver in args.scheduling_heuristics:
            tasks = [(solver, cfg, run_time, idx, args.seed, args) for idx, run_time in enumerate(instances)]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            solver_rows, summary = _execute(solver, tasks, _run_ffsp_task, args.workers)
            rows.extend(solver_rows)
            summaries.append(summary)
    elif cfg.problem == "jssp":
        paths = bench._load_jssp_instances(cfg, args.max_instances, args.seed, args.jssp_data_dir)
        for solver in args.scheduling_heuristics:
            tasks = [(solver, cfg, path, idx, args.seed, args) for idx, path in enumerate(paths)]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            solver_rows, summary = _execute(solver, tasks, _run_jssp_task, args.workers)
            rows.extend(solver_rows)
            summaries.append(summary)
    else:
        raise ValueError(f"Unsupported problem: {cfg.problem}")
    return rows, summaries


def main() -> int:
    args = parse_args()
    all_rows: list[bench.PerInstanceResult] = []
    all_summaries: list[bench.SolverBatchSummary] = []
    for scenario_name in args.scenarios:
        cfg = bench.SCENARIOS[scenario_name]
        rows, summaries = benchmark_scenario(cfg, args)
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        print(f"[{scenario_name}] finished {len(rows)} heuristic-runs", flush=True)
    bench.write_outputs(all_rows, all_summaries, args.output_dir.resolve())
    print(f"Wrote heuristic supplement outputs to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
