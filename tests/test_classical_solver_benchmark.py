from __future__ import annotations

import numpy as np

from scripts import classical_solver_benchmark as mod
from scripts import heuristic_baseline_supplement as heur


def test_ffsp_neh_makespan_returns_valid_schedule_cost() -> None:
    run_time = np.array(
        [
            [3, 5, 2, 4],
            [2, 6, 3, 1],
            [4, 1, 2, 3],
        ],
        dtype=np.int32,
    )

    makespan = mod.ffsp_neh_makespan(run_time, num_stage=2, num_machine=2)

    assert isinstance(makespan, int)
    assert makespan > 0
    assert makespan <= int(run_time.sum())


def test_ffsp_scenarios_include_neh_solver() -> None:
    assert "neh" in mod.SCENARIOS["ffsp50"].solvers
    assert "neh" in mod.SCENARIOS["ffsp100"].solvers


def test_tsp_simple_heuristics_return_finite_costs() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    assert heur.tsp_nn_2opt(coords) > 0
    assert heur.tsp_nearest_insertion(coords) > 0


def test_cvrp_sweep_respects_basic_instance() -> None:
    instance = {
        "depot": np.array([0.0, 0.0], dtype=np.float32),
        "locs": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "demand": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "capacity": 2.0,
    }

    assert heur.cvrp_nearest_feasible(instance) > 0
    assert heur.cvrp_sweep(instance) > 0
    assert heur.cvrp_savings(instance) > 0


def test_metaheuristic_helpers_improve_valid_scheduling_costs() -> None:
    run_time = np.array(
        [
            [3, 5, 2, 4],
            [2, 6, 3, 1],
            [4, 1, 2, 3],
        ],
        dtype=np.int32,
    )
    rng = np.random.default_rng(123)
    objective = heur._ga_minimize(
        3,
        lambda keys: heur._ffsp_schedule_by_keys(run_time, 2, 2, keys),
        rng,
        population=6,
        generations=3,
        mutation_sigma=0.1,
    )

    assert objective > 0
    assert objective <= int(run_time.sum())
