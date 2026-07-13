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
        description="Run lightweight routing metaheuristic baselines for TSP/CVRP result tables."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["tsp50", "tsp100", "cvrp50", "cvrp100"],
        choices=["tsp50", "tsp100", "cvrp50", "cvrp100"],
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "routing_metaheuristic_supplement")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--require-test-data", action="store_true")
    parser.add_argument(
        "--tsp-solvers",
        nargs="+",
        default=["two_opt", "ga", "pso"],
        choices=["two_opt", "ga", "pso"],
    )
    parser.add_argument(
        "--cvrp-solvers",
        nargs="+",
        default=["two_opt", "ga", "pso"],
        choices=["two_opt", "ga", "pso"],
    )
    parser.add_argument("--two-opt-passes", type=int, default=2)
    parser.add_argument("--ga-population", type=int, default=64)
    parser.add_argument("--ga-generations", type=int, default=100)
    parser.add_argument("--pso-particles", type=int, default=64)
    parser.add_argument("--pso-iterations", type=int, default=100)
    parser.add_argument(
        "--local-select-top",
        type=int,
        default=8,
        help="Number of top raw metaheuristic candidates to re-rank after route-level local improvement.",
    )
    parser.add_argument("--mutation-sigma", type=float, default=0.15)
    return parser.parse_args()


def _scaled_euc_dist_matrix(coords: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.asarray(coords, dtype=np.float64) * 100_000.0)
    diff = scaled[:, None, :] - scaled[None, :, :]
    return np.floor(np.sqrt(np.sum(diff * diff, axis=-1)) + 0.5).astype(np.int64)


def _tour_cost(dist: np.ndarray, tour: list[int]) -> int:
    total = 0
    n = len(tour)
    for idx, node in enumerate(tour):
        total += int(dist[node, tour[(idx + 1) % n]])
    return total


def _two_opt_tour(dist: np.ndarray, tour: list[int], max_passes: int) -> list[int]:
    n = len(tour)
    for _ in range(max(0, int(max_passes))):
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            stop = n if i > 0 else n - 1
            for k in range(i + 2, stop):
                c, d = tour[k], tour[(k + 1) % n]
                if int(dist[a, c] + dist[b, d]) < int(dist[a, b] + dist[c, d]):
                    tour[i + 1 : k + 1] = reversed(tour[i + 1 : k + 1])
                    improved = True
        if not improved:
            break
    return tour


def _nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> list[int]:
    n = int(dist.shape[0])
    start = int(start) % n
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda node: (int(dist[current, node]), node))
        unvisited.remove(nxt)
        tour.append(nxt)
        current = nxt
    return tour


def _keys_from_order(order: np.ndarray, dim: int, offset: int = 0) -> np.ndarray:
    keys = np.empty(dim, dtype=np.float64)
    denom = max(1, dim - 1)
    for rank, raw_item in enumerate(order):
        keys[int(raw_item) - offset] = rank / denom
    return keys


def _seeded_key_population(
    base_keys: np.ndarray,
    rng: np.random.Generator,
    population: int,
    noise: float = 0.08,
    random_fraction: float = 0.25,
) -> np.ndarray:
    population = max(4, int(population))
    dim = int(base_keys.shape[0])
    pop = np.empty((population, dim), dtype=np.float64)
    pop[0] = base_keys
    random_count = int(round(population * random_fraction))
    for idx in range(1, population):
        if idx > population - random_count:
            pop[idx] = rng.uniform(size=dim)
        else:
            pop[idx] = np.clip(base_keys + rng.normal(0.0, noise, size=dim), 0.0, 1.0)
    return pop


def _tsp_multistart_key_population(
    dist: np.ndarray,
    rng: np.random.Generator,
    population: int,
    noise: float,
    random_fraction: float,
) -> np.ndarray:
    population = max(4, int(population))
    dim = int(dist.shape[0])
    pop = np.empty((population, dim), dtype=np.float64)
    random_count = int(round(population * random_fraction))
    for idx in range(population):
        if idx >= population - random_count:
            pop[idx] = rng.uniform(size=dim)
            continue
        start = 0 if idx == 0 else int(rng.integers(0, dim))
        base_order = np.asarray(_nearest_neighbor_tour(dist, start=start), dtype=np.int64)
        base_keys = _keys_from_order(base_order, dim)
        pop[idx] = np.clip(base_keys + rng.normal(0.0, noise, size=dim), 0.0, 1.0)
    return pop


def _order_tsp_cost(dist: np.ndarray, order: np.ndarray) -> int:
    return int(dist[order, np.roll(order, -1)].sum())


def _ga_minimize(
    dim: int,
    evaluate,
    rng: np.random.Generator,
    population: int,
    generations: int,
    mutation_sigma: float,
    initial_pop: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    population = max(4, int(population))
    generations = max(1, int(generations))
    if initial_pop is None:
        pop = rng.uniform(size=(population, dim))
    else:
        pop = np.asarray(initial_pop, dtype=np.float64).copy()
        population = int(pop.shape[0])
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
    best_idx = int(np.argmin(scores))
    return pop[best_idx].copy(), int(scores[best_idx]), pop.copy(), scores.copy()


def _pso_minimize(
    dim: int,
    evaluate,
    rng: np.random.Generator,
    particles: int,
    iterations: int,
    initial_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    particles = max(4, int(particles))
    iterations = max(1, int(iterations))
    if initial_pos is None:
        pos = rng.uniform(size=(particles, dim))
    else:
        pos = np.asarray(initial_pos, dtype=np.float64).copy()
        particles = int(pos.shape[0])
    vel = rng.normal(0.0, 0.1, size=(particles, dim))
    personal = pos.copy()
    personal_scores = np.array([evaluate(ind) for ind in pos], dtype=np.float64)
    best_idx = int(np.argmin(personal_scores))
    global_best = personal[best_idx].copy()
    global_score = float(personal_scores[best_idx])
    for _ in range(iterations):
        r1 = rng.random(size=(particles, dim))
        r2 = rng.random(size=(particles, dim))
        vel = 0.72 * vel + 1.49 * r1 * (personal - pos) + 1.49 * r2 * (global_best - pos)
        pos = np.mod(pos + vel, 1.0)
        scores = np.array([evaluate(ind) for ind in pos], dtype=np.float64)
        improved = scores < personal_scores
        personal[improved] = pos[improved]
        personal_scores[improved] = scores[improved]
        best_idx = int(np.argmin(personal_scores))
        if float(personal_scores[best_idx]) < global_score:
            global_score = float(personal_scores[best_idx])
            global_best = personal[best_idx].copy()
    return global_best, int(global_score), personal.copy(), personal_scores.copy()


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


def _split_order_by_capacity(order: np.ndarray, demand: np.ndarray, capacity: float) -> list[list[int]]:
    routes: list[list[int]] = []
    route: list[int] = []
    load = 0.0
    for raw_node in order:
        node = int(raw_node)
        node_demand = float(demand[node])
        if route and load + node_demand > capacity:
            routes.append(route)
            route = []
            load = 0.0
        route.append(node)
        load += node_demand
    if route:
        routes.append(route)
    return routes


def _two_opt_open_route(dist: np.ndarray, route: list[int], max_passes: int) -> list[int]:
    if len(route) < 4:
        return route
    path = [0] + route + [0]
    for _ in range(max(0, int(max_passes))):
        improved = False
        for i in range(len(path) - 3):
            a, b = path[i], path[i + 1]
            for k in range(i + 2, len(path) - 1):
                c, d = path[k], path[k + 1]
                if int(dist[a, c] + dist[b, d]) < int(dist[a, b] + dist[c, d]):
                    path[i + 1 : k + 1] = reversed(path[i + 1 : k + 1])
                    improved = True
        if not improved:
            break
    return path[1:-1]


def _cvrp_savings_routes(dist: np.ndarray, demand: np.ndarray, capacity: float) -> list[list[int]]:
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
    return list(routes.values())


def _top_candidate_keys(
    initial_keys: np.ndarray,
    final_keys: np.ndarray,
    final_scores: np.ndarray,
    best_keys: np.ndarray,
    top_k: int,
) -> list[np.ndarray]:
    candidates = [np.asarray(best_keys, dtype=np.float64)]
    if final_keys.size:
        order = np.argsort(final_scores)[: max(1, int(top_k))]
        candidates.extend(np.asarray(final_keys[idx], dtype=np.float64) for idx in order)
    candidates.extend(np.asarray(row, dtype=np.float64) for row in initial_keys)

    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for keys in candidates:
        signature = np.argsort(keys, kind="mergesort").astype(np.int32).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(keys)
    return unique


def _best_tsp_local_objective(dist: np.ndarray, key_candidates: list[np.ndarray], two_opt_passes: int) -> int:
    best = _tour_cost(dist, _two_opt_tour(dist, _nearest_neighbor_tour(dist), two_opt_passes))
    for keys in key_candidates:
        tour = list(np.argsort(keys, kind="mergesort"))
        objective = _tour_cost(dist, _two_opt_tour(dist, tour, two_opt_passes))
        if objective < best:
            best = objective
    return int(best)


def _best_cvrp_local_objective(
    dist: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    key_candidates: list[np.ndarray],
    two_opt_passes: int,
) -> int:
    base_routes = [_two_opt_open_route(dist, route, two_opt_passes) for route in _cvrp_savings_routes(dist, demand, capacity)]
    best = _routes_cost(dist, base_routes)
    nodes = np.arange(1, dist.shape[0])
    for keys in key_candidates:
        routes = _split_order_by_capacity(nodes[np.argsort(keys, kind="mergesort")], demand, capacity)
        objective = _routes_cost(dist, [_two_opt_open_route(dist, route, two_opt_passes) for route in routes])
        if objective < best:
            best = objective
    return int(best)


def _run_tsp_task(
    solver: str,
    cfg: bench.ScenarioConfig,
    coords: np.ndarray,
    idx: int,
    seed: int,
    args: argparse.Namespace,
) -> bench.PerInstanceResult:
    dist = _scaled_euc_dist_matrix(coords)
    rng = np.random.default_rng(seed + 100_003 * idx + {"two_opt": 0, "ga": 10_000_000, "pso": 20_000_000}[solver])

    t0 = time.perf_counter()
    if solver == "two_opt":
        objective = _tour_cost(dist, _two_opt_tour(dist, _nearest_neighbor_tour(dist), args.two_opt_passes))
    elif solver == "ga":
        initial_pop = _tsp_multistart_key_population(
            dist,
            rng,
            args.ga_population,
            noise=0.06,
            random_fraction=0.2,
        )
        evaluate = lambda keys: _order_tsp_cost(dist, np.argsort(keys, kind="mergesort"))
        best_keys, _, final_pop, final_scores = _ga_minimize(
            dist.shape[0],
            evaluate,
            rng,
            args.ga_population,
            args.ga_generations,
            args.mutation_sigma,
            initial_pop=initial_pop,
        )
        objective = _best_tsp_local_objective(
            dist,
            _top_candidate_keys(initial_pop, final_pop, final_scores, best_keys, args.local_select_top),
            args.two_opt_passes,
        )
    elif solver == "pso":
        initial_pos = _tsp_multistart_key_population(
            dist,
            rng,
            args.pso_particles,
            noise=0.06,
            random_fraction=0.2,
        )
        evaluate = lambda keys: _order_tsp_cost(dist, np.argsort(keys, kind="mergesort"))
        best_keys, _, final_pos, final_scores = _pso_minimize(
            dist.shape[0], evaluate, rng, args.pso_particles, args.pso_iterations, initial_pos=initial_pos
        )
        objective = _best_tsp_local_objective(
            dist,
            _top_candidate_keys(initial_pos, final_pos, final_scores, best_keys, args.local_select_top),
            args.two_opt_passes,
        )
    else:
        raise ValueError(f"Unsupported TSP solver: {solver}")
    return bench.PerInstanceResult(cfg.name, solver, f"{cfg.name}_{idx:05d}", "ok", float(objective), time.perf_counter() - t0)


def _run_cvrp_task(
    solver: str,
    cfg: bench.ScenarioConfig,
    instance: dict[str, np.ndarray],
    idx: int,
    seed: int,
    args: argparse.Namespace,
) -> bench.PerInstanceResult:
    dist, demand, capacity = _cvrp_distance(instance)
    rng = np.random.default_rng(seed + 200_003 * idx + {"two_opt": 0, "ga": 10_000_000, "pso": 20_000_000}[solver])

    t0 = time.perf_counter()
    if solver == "two_opt":
        routes = _cvrp_savings_routes(dist, demand, capacity)
        routes = [_two_opt_open_route(dist, route, args.two_opt_passes) for route in routes]
        objective = _routes_cost(dist, routes)
    elif solver == "ga":
        nodes = np.arange(1, dist.shape[0])
        base_order = np.asarray([node for route in _cvrp_savings_routes(dist, demand, capacity) for node in route], dtype=np.int64)
        initial_pop = _seeded_key_population(
            _keys_from_order(base_order, len(nodes), offset=1),
            rng,
            args.ga_population,
            noise=0.08,
            random_fraction=0.2,
        )
        evaluate = lambda keys: _routes_cost(dist, _split_order_by_capacity(nodes[np.argsort(keys, kind="mergesort")], demand, capacity))
        best_keys, _, final_pop, final_scores = _ga_minimize(
            len(nodes),
            evaluate,
            rng,
            args.ga_population,
            args.ga_generations,
            args.mutation_sigma,
            initial_pop=initial_pop,
        )
        objective = _best_cvrp_local_objective(
            dist,
            demand,
            capacity,
            _top_candidate_keys(initial_pop, final_pop, final_scores, best_keys, args.local_select_top),
            args.two_opt_passes,
        )
    elif solver == "pso":
        nodes = np.arange(1, dist.shape[0])
        base_order = np.asarray([node for route in _cvrp_savings_routes(dist, demand, capacity) for node in route], dtype=np.int64)
        initial_pos = _seeded_key_population(
            _keys_from_order(base_order, len(nodes), offset=1),
            rng,
            args.pso_particles,
            noise=0.08,
            random_fraction=0.2,
        )
        evaluate = lambda keys: _routes_cost(dist, _split_order_by_capacity(nodes[np.argsort(keys, kind="mergesort")], demand, capacity))
        best_keys, _, final_pos, final_scores = _pso_minimize(
            len(nodes), evaluate, rng, args.pso_particles, args.pso_iterations, initial_pos=initial_pos
        )
        objective = _best_cvrp_local_objective(
            dist,
            demand,
            capacity,
            _top_candidate_keys(initial_pos, final_pos, final_scores, best_keys, args.local_select_top),
            args.two_opt_passes,
        )
    else:
        raise ValueError(f"Unsupported CVRP solver: {solver}")
    return bench.PerInstanceResult(cfg.name, solver, f"{cfg.name}_{idx:05d}", "ok", float(objective), time.perf_counter() - t0)


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
        for solver in args.tsp_solvers:
            tasks = [(solver, cfg, coords, idx, args.seed, args) for idx, coords in enumerate(instances)]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            solver_rows, summary = _execute(solver, tasks, _run_tsp_task, args.workers)
            rows.extend(solver_rows)
            summaries.append(summary)
    elif cfg.problem == "cvrp":
        instances = bench._load_cvrp_instances(cfg, args.seed, args.max_instances, args.require_test_data)
        for solver in args.cvrp_solvers:
            tasks = [(solver, cfg, instance, idx, args.seed, args) for idx, instance in enumerate(instances)]
            print(f"[{cfg.name}/{solver}] launching {len(tasks)} instances", flush=True)
            solver_rows, summary = _execute(solver, tasks, _run_cvrp_task, args.workers)
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
    print(f"Wrote routing metaheuristic supplement outputs to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
