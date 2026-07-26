"""Falsify matched-branch screening against a terminal-score ordinary-b baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np

from .matched_branch_probe import SCHEMA


EPS = 1e-12
SELECTORS = ("all", "adjacent", "best_anchor", "large_gap")
MARGINS = (
    "zero",
    "cost_gap",
    "rank_span",
    "depth_adaptive",
    "negative_one",
    "positive_two",
    "reverse_gap",
    "late_switch",
)


@dataclass(frozen=True)
class BranchProbe:
    policy_state: str
    depth: int
    depth_fraction: float
    local_logp: np.ndarray
    terminal_logp: np.ndarray
    normalized_cost: np.ndarray


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    return exp_values / float(exp_values.sum())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])


def _matrices(
    count: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    gap = np.zeros((count - 1, count), dtype=float)
    for boundary in range(count - 1):
        gap[boundary, boundary] = 1.0
        gap[boundary, boundary + 1] = -1.0
    edges = list(combinations(range(count), 2))
    interval = np.zeros((len(edges), count - 1), dtype=float)
    for row, (winner, loser) in enumerate(edges):
        interval[row, winner:loser] = 1.0
    return gap, interval, edges


def _select_edges(
    name: str,
    edges: list[tuple[int, int]],
    costs: np.ndarray,
) -> np.ndarray:
    objective_gaps = np.asarray([costs[j] - costs[i] for i, j in edges])
    if name == "all":
        selected = np.ones(len(edges), dtype=bool)
    elif name == "adjacent":
        selected = np.asarray([j == i + 1 for i, j in edges], dtype=bool)
    elif name == "best_anchor":
        selected = np.asarray([i == 0 for i, _ in edges], dtype=bool)
    elif name == "large_gap":
        selected = objective_gaps >= float(np.median(objective_gaps))
    else:
        raise KeyError(name)
    if not selected.any():
        selected[int(np.argmax(objective_gaps))] = True
    return selected


def _margin_targets(
    name: str,
    edges: list[tuple[int, int]],
    costs: np.ndarray,
    depth_fraction: float,
) -> np.ndarray:
    denominator = max(len(costs) - 1, 1)
    objective_gaps = np.asarray([costs[j] - costs[i] for i, j in edges])
    rank_spans = np.asarray([(j - i) / denominator for i, j in edges])
    if name == "zero":
        target = np.zeros(len(edges))
    elif name == "cost_gap":
        target = 2.0 * objective_gaps
    elif name == "rank_span":
        target = 1.5 * rank_spans - 0.25
    elif name == "depth_adaptive":
        target = (
            (1.0 + 2.0 * depth_fraction) * objective_gaps
            - 0.5 * rank_spans
        )
    elif name == "negative_one":
        target = np.full(len(edges), -1.0)
    elif name == "positive_two":
        target = np.full(len(edges), 2.0)
    elif name == "reverse_gap":
        target = 1.0 - 3.0 * objective_gaps
    elif name == "late_switch":
        target = (
            (4.0 * depth_fraction - 1.0) * objective_gaps - rank_spans
        )
    else:
        raise KeyError(name)
    return np.clip(target, -4.0, 4.0)


def _program_boundary(
    log_scores: np.ndarray,
    costs: np.ndarray,
    depth_fraction: float,
    selector: str,
    margin: str,
) -> tuple[np.ndarray, np.ndarray]:
    gap, interval, edges = _matrices(len(log_scores))
    selected = _select_edges(selector, edges, costs)
    proposed = _margin_targets(
        margin, edges, costs, depth_fraction
    )
    objective_gaps = np.asarray([costs[j] - costs[i] for i, j in edges])
    centered = proposed - float(np.mean(proposed[selected]))
    target = 2.0 * objective_gaps + np.clip(centered, -2.0, 2.0)
    pair_margin = interval @ (gap @ log_scores)
    normalizer = max(int(selected.sum()), 1)
    pressure = (
        selected.astype(float)
        * _sigmoid(target - pair_margin)
        / (4.0 * normalizer)
    )
    return interval.T @ pressure, gap


def _virtual_improvement(
    log_scores: np.ndarray,
    costs: np.ndarray,
    boundary: np.ndarray,
    gap: np.ndarray,
    step_size: float,
) -> float:
    before = float(_softmax(log_scores) @ costs)
    after_scores = log_scores + step_size * (gap.T @ boundary)
    after = float(_softmax(after_scores) @ costs)
    return before - after


def _load_bank(path: Path) -> tuple[list[BranchProbe], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}")
    problem_size = int(payload["problem_size"])
    bank: list[BranchProbe] = []
    for record in payload["probes"]:
        local = np.asarray(record["local_logp"], dtype=float)
        terminal = np.asarray(record["terminal_logp"], dtype=float)
        objective = np.asarray(record["objective"], dtype=float)
        if local.shape != terminal.shape or local.shape != objective.shape:
            raise ValueError("probe arrays must share [instance, branch] shape")
        for row in range(local.shape[0]):
            order = np.argsort(objective[row], kind="mergesort")
            sorted_cost = objective[row, order]
            cost_range = max(
                float(sorted_cost[-1] - sorted_cost[0]), EPS
            )
            bank.append(
                BranchProbe(
                    policy_state=str(record["policy_state"]),
                    depth=int(record["depth"]),
                    depth_fraction=float(record["depth"]) / problem_size,
                    local_logp=local[row, order],
                    terminal_logp=terminal[row, order],
                    normalized_cost=(
                        sorted_cost - float(sorted_cost[0])
                    )
                    / cost_range,
                )
            )
    if not bank:
        raise ValueError("probe bank is empty")
    return bank, payload


def _program_values(
    bank: list[BranchProbe],
    step_size: float,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    candidates = [
        (selector, margin)
        for selector in SELECTORS
        for margin in MARGINS
    ]
    local_value = np.zeros((len(bank), len(candidates)), dtype=float)
    terminal_score = np.zeros_like(local_value)
    for probe_index, probe in enumerate(bank):
        for candidate_index, (selector, margin) in enumerate(candidates):
            local_boundary, gap = _program_boundary(
                probe.local_logp,
                probe.normalized_cost,
                probe.depth_fraction,
                selector,
                margin,
            )
            terminal_boundary, terminal_gap = _program_boundary(
                probe.terminal_logp,
                probe.normalized_cost,
                probe.depth_fraction,
                selector,
                margin,
            )
            local_value[probe_index, candidate_index] = _virtual_improvement(
                probe.local_logp,
                probe.normalized_cost,
                local_boundary,
                gap,
                step_size,
            )
            terminal_score[probe_index, candidate_index] = (
                _virtual_improvement(
                    probe.terminal_logp,
                    probe.normalized_cost,
                    terminal_boundary,
                    terminal_gap,
                    step_size,
                )
            )
    return candidates, local_value, terminal_score


def _budget_trials(
    local_value: np.ndarray,
    terminal_score: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, float | int]]:
    population_value = local_value.mean(axis=0)
    oracle = int(np.argmax(population_value))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    maximum = len(local_value)
    requested = (4, 8, 16, 32, 64)
    for probe_count in tuple(value for value in requested if value <= maximum):
        local_hits = 0
        terminal_hits = 0
        staged_hits = 0
        random_hits = 0
        local_regret = []
        terminal_regret = []
        staged_regret = []
        random_regret = []
        for _ in range(repeats):
            indices = rng.choice(maximum, probe_count, replace=False)
            local_screen = local_value[indices].mean(axis=0)
            local_pick = int(np.argmax(local_screen))
            terminal_pick = int(
                np.argmax(terminal_score[indices].mean(axis=0))
            )
            local_matrix = local_screen.reshape(
                len(SELECTORS), len(MARGINS)
            )
            staged_selector = int(np.argmax(local_matrix[:, 0]))
            staged_margin = int(
                np.argmax(local_matrix[staged_selector])
            )
            staged_pick = staged_selector * len(MARGINS) + staged_margin
            random_pick = int(rng.integers(0, len(population_value)))
            local_hits += int(local_pick == oracle)
            terminal_hits += int(terminal_pick == oracle)
            staged_hits += int(staged_pick == oracle)
            random_hits += int(random_pick == oracle)
            local_regret.append(
                population_value[oracle] - population_value[local_pick]
            )
            terminal_regret.append(
                population_value[oracle] - population_value[terminal_pick]
            )
            staged_regret.append(
                population_value[oracle] - population_value[staged_pick]
            )
            random_regret.append(
                population_value[oracle] - population_value[random_pick]
            )
        rows.append(
            {
                "probe_count": probe_count,
                "local_hit_rate": local_hits / repeats,
                "terminal_hit_rate": terminal_hits / repeats,
                "staged_hit_rate": staged_hits / repeats,
                "random_hit_rate": random_hits / repeats,
                "local_mean_regret": float(np.mean(local_regret)),
                "terminal_mean_regret": float(np.mean(terminal_regret)),
                "staged_mean_regret": float(np.mean(staged_regret)),
                "random_mean_regret": float(np.mean(random_regret)),
            }
        )
    return rows


def analyze(
    input_path: Path,
    *,
    step_size: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    bank, source = _load_bank(input_path)
    candidates, local_value, terminal_score = _program_values(
        bank, step_size
    )
    population_value = local_value.mean(axis=0)
    terminal_population = terminal_score.mean(axis=0)
    oracle = int(np.argmax(population_value))
    terminal_pick = int(np.argmax(terminal_population))
    local_matrix = population_value.reshape(len(SELECTORS), len(MARGINS))
    staged_selector = int(np.argmax(local_matrix[:, 0]))
    staged_margin = int(np.argmax(local_matrix[staged_selector]))
    staged_pick = staged_selector * len(MARGINS) + staged_margin
    oracle_value = float(population_value[oracle])
    terminal_regret = float(oracle_value - population_value[terminal_pick])
    states: dict[str, Any] = {}
    for state in sorted({probe.policy_state for probe in bank}):
        indices = np.asarray(
            [probe.policy_state == state for probe in bank], dtype=bool
        )
        state_value = local_value[indices].mean(axis=0)
        state_terminal = terminal_score[indices].mean(axis=0)
        states[state] = {
            "probe_count": int(indices.sum()),
            "local_vs_terminal_pooled_spearman": _spearman(
                local_value[indices].ravel(),
                terminal_score[indices].ravel(),
            ),
            "oracle_program": candidates[int(np.argmax(state_value))],
            "terminal_selected_program": candidates[
                int(np.argmax(state_terminal))
            ],
        }
    return {
        "schema": "nco-matched-branch-analysis-v2",
        "source_config_sha256": source["config_sha256"],
        "probe_count": len(bank),
        "program_count": len(candidates),
        "step_size": step_size,
        "local_vs_terminal_pooled_spearman": _spearman(
            local_value.ravel(), terminal_score.ravel()
        ),
        "local_vs_terminal_program_spearman": _spearman(
            population_value, terminal_population
        ),
        "oracle_program": candidates[oracle],
        "oracle_mean_improvement": oracle_value,
        "terminal_selected_program": candidates[terminal_pick],
        "terminal_selected_mean_improvement": float(
            population_value[terminal_pick]
        ),
        "terminal_selection_regret": terminal_regret,
        "terminal_selection_relative_regret": terminal_regret
        / max(abs(oracle_value), EPS),
        "staged_selected_program": candidates[staged_pick],
        "staged_selected_mean_improvement": float(population_value[staged_pick]),
        "staged_selection_regret": float(
            oracle_value - population_value[staged_pick]
        ),
        "random_expected_mean_improvement": float(population_value.mean()),
        "random_expected_regret": float(
            oracle_value - population_value.mean()
        ),
        "states": states,
        "budget_trials": _budget_trials(
            local_value,
            terminal_score,
            repeats=repeats,
            seed=seed,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        args.input,
        step_size=float(args.step_size),
        repeats=int(args.repeats),
        seed=int(args.seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
