from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_scale1000_scenarios_include_lkh_solver() -> None:
    assert mod.SCENARIOS["tsp1000"].test_file.name == "tsp1000_test_seed1234.npz"
    assert mod.SCENARIOS["tsp1000"].solvers == ("lkh",)
    assert mod.SCENARIOS["cvrp1000"].test_file.name == "agfn_vrp1000_capacity50_test128.npz"
    assert mod.SCENARIOS["cvrp1000"].solvers == ("lkh",)
    assert mod.SCENARIOS["cvrp1000"].lkh_runs == 1
    assert mod.SCENARIOS["cvrp1000"].lkh_max_trials == 1000


def test_parse_lkh_mtsp_solution_extracts_instance_local_routes(tmp_path: Path) -> None:
    solution = tmp_path / "solution.txt"
    solution.write_text(
        "tiny, Cost: 0_400000\n"
        "The tours traveled by the 2 salesmen are:\n"
        "1 2 3 1 (#2)  Cost: 200000\n"
        "1 4 1 (#1)  Cost: 200000\n",
        encoding="utf-8",
    )

    assert mod._parse_lkh_mtsp_solution(solution) == [[2, 3], [4]]


def test_concorde_nonzero_returncode_is_not_marked_optimal(
    tmp_path: Path, monkeypatch
) -> None:
    concorde = tmp_path / "tools/concorde/TSP/concorde"
    concorde.parent.mkdir(parents=True)
    concorde.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=255,
            stdout="Optimal Solution: 12345\n",
            stderr="temporary-file collision",
        ),
    )

    result = mod.run_tsp_concorde(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "tsp_test_00000",
        "tsp_test",
        tmp_path / "output",
    )

    assert result.status == "error_255"
    assert result.objective is None


def test_concorde_255_with_valid_exact_certificate_is_optimal(
    tmp_path: Path, monkeypatch
) -> None:
    concorde = tmp_path / "tools/concorde/TSP/concorde"
    concorde.parent.mkdir(parents=True)
    concorde.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    def fake_run(args, **kwargs):
        solution_path = Path(args[args.index("-o") + 1])
        solution_path.write_text("3\n0 1 2\n", encoding="utf-8")
        return SimpleNamespace(
            returncode=255,
            stdout=(
                "Exact lower bound: 341421.000000\n"
                "DIFF: 0.000000\n"
                "Optimal Solution: 341421.00\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    result = mod.run_tsp_concorde(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "tsp_test_00000",
        "tsp_test",
        tmp_path / "output",
    )

    assert result.status == "optimal"
    assert result.objective == 341421.0
    assert "certificate=validated" in result.notes


def test_tsp_lkh_nonzero_returncode_is_not_marked_ok(
    tmp_path: Path, monkeypatch
) -> None:
    lkh = tmp_path / "tools/LKH-3.0.13/LKH"
    lkh.parent.mkdir(parents=True)
    lkh.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="invalid parameter file",
        ),
    )

    result = mod.run_tsp_lkh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        seed=1234,
        instance_id="tsp_test_00000",
        scenario_name="tsp_test",
        time_limit_sec=1.0,
        runs=1,
        output_dir=tmp_path / "output",
    )

    assert result.status == "error_1"
    assert result.objective is None


def test_cvrp_lkh_route_is_recomputed_and_audited(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / "tools/LKH-3.0.13/LKH"
    executable.parent.mkdir(parents=True)
    executable.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    observed: dict[str, str] = {}

    def fake_run(*args, **kwargs):
        observed["parameters"] = (Path(kwargs["cwd"]) / "tiny.par").read_text(
            encoding="utf-8"
        )
        (Path(kwargs["cwd"]) / "tiny.solution").write_text(
            "tiny, Cost: 0_541421\n"
            "The tours traveled by the 2 salesmen are:\n"
            "1 2 3 1 (#2)  Cost: 341421\n"
            "1 4 1 (#1)  Cost: 200000\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    result = mod.run_cvrp_lkh(
        {
            "depot": np.array([0.0, 0.0], dtype=np.float32),
            "locs": np.array(
                [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
            ),
            "demand": np.ones(3, dtype=np.float32),
            "capacity": 2.0,
        },
        seed=1234,
        instance_id="tiny",
        scenario_name="cvrp3",
        time_limit_sec=1.0,
        runs=1,
        max_trials=1000,
        output_dir=tmp_path / "output",
    )

    assert result.status == "ok"
    assert result.objective == 541421.0
    assert "routes=2" in result.notes
    assert "SPECIAL\n" in observed["parameters"]
    assert "SUBGRADIENT = NO\n" in observed["parameters"]
    assert "MAX_TRIALS = 1000\n" in observed["parameters"]


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
