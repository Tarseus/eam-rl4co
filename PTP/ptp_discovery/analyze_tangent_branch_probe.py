"""Screen two source programs in exact policy-parameter tangent geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_matched_branch_probe import BranchProbe, EPS, _matrices, _spearman
from .stress_matched_branch_search import (
    _coordinate_ascent,
    _coordinate_basins,
    _edge_features,
    _make_sources,
    _margin_values,
    _selector_masks,
    _sigmoid,
)
from .tangent_branch_probe import SCHEMA


def _program_log_score_gradients(
    probe: BranchProbe,
    log_scores: np.ndarray,
    selectors,
    margins,
) -> np.ndarray:
    gap_matrix, interval, edges = _matrices(len(log_scores))
    features = _edge_features(probe, edges)
    selector = _selector_masks(features, selectors)
    margin = _margin_values(features, margins)
    objective_gap = features[:, 1]
    selector = selector & (objective_gap[None, :] > EPS)
    empty = ~selector.any(axis=1)
    if empty.any():
        selector[empty, int(np.argmax(objective_gap))] = True
    normalizer = selector.sum(axis=1).astype(float)
    selected_mean = np.einsum("se,me->sm", selector, margin)
    selected_mean /= normalizer[:, None]
    target = 2.0 * objective_gap[None, None, :]
    target = target + np.clip(
        margin[None, :, :] - selected_mean[:, :, None], -2.0, 2.0
    )
    pair_margin = interval @ (gap_matrix @ log_scores)
    pressure = selector[:, None, :] * _sigmoid(
        target - pair_margin[None, None, :]
    )
    pressure /= 4.0 * normalizer[:, None, None]
    boundary = np.einsum("sme,ek->smk", pressure, interval)
    descent_direction = np.einsum("smk,kn->smn", boundary, gap_matrix)
    return -descent_direction


def _load_tangent_bank(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}")
    problem_size = int(payload["problem_size"])
    bank: list[dict[str, Any]] = []
    for record in payload["probes"]:
        arrays = {
            name: np.asarray(record[name], dtype=float)
            for name in (
                "local_logp",
                "terminal_logp",
                "objective",
                "local_target_influence",
                "terminal_target_influence",
            )
        }
        shape = arrays["objective"].shape
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("all tangent arrays must share [instance, branch] shape")
        for row in range(shape[0]):
            order = np.argsort(arrays["objective"][row], kind="mergesort")
            sorted_objective = arrays["objective"][row, order]
            objective_range = max(
                float(sorted_objective[-1] - sorted_objective[0]), EPS
            )
            normalized = (
                sorted_objective - float(sorted_objective[0])
            ) / objective_range
            target_gradient_norm = float(record["target_gradient_norm"])
            if not np.isfinite(target_gradient_norm) or target_gradient_norm <= 0.0:
                raise ValueError("target_gradient_norm must be finite and positive")
            bank.append(
                {
                    "probe": BranchProbe(
                        policy_state=str(record["policy_state"]),
                        depth=int(record["depth"]),
                        depth_fraction=float(record["depth"]) / problem_size,
                        local_logp=arrays["local_logp"][row, order],
                        terminal_logp=arrays["terminal_logp"][row, order],
                        normalized_cost=normalized,
                    ),
                    "local_influence": arrays["local_target_influence"][row, order]
                    / target_gradient_norm,
                    "terminal_influence": arrays["terminal_target_influence"][row, order]
                    / target_gradient_norm,
                }
            )
    if not bank:
        raise ValueError("tangent bank is empty")
    return bank, payload


def _score_bank(bank, selectors, margins) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(bank), len(selectors), len(margins))
    local_score = np.empty(shape, dtype=float)
    terminal_score = np.empty(shape, dtype=float)
    for index, item in enumerate(bank):
        probe = item["probe"]
        local_gradient = _program_log_score_gradients(
            probe, probe.local_logp, selectors, margins
        )
        terminal_gradient = _program_log_score_gradients(
            probe, probe.terminal_logp, selectors, margins
        )
        local_score[index] = np.einsum(
            "smk,k->sm", local_gradient, item["local_influence"]
        )
        terminal_score[index] = np.einsum(
            "smk,k->sm", terminal_gradient, item["terminal_influence"]
        )
    return local_score, terminal_score


def _aggregate_scores(
    scores: np.ndarray,
    bank,
    mode: str,
) -> np.ndarray:
    if mode == "mean":
        return scores.mean(axis=0)
    if mode == "probe_cvar_20":
        tail_count = max(1, int(np.ceil(0.2 * len(scores))))
        return np.sort(scores, axis=0)[:tail_count].mean(axis=0)
    keys = [
        (item["probe"].policy_state, item["probe"].depth)
        for item in bank
    ]
    strata = []
    for key in sorted(set(keys)):
        mask = np.asarray([value == key for value in keys], dtype=bool)
        strata.append(scores[mask].mean(axis=0))
    stratum_scores = np.stack(strata, axis=0)
    if mode == "stratum_min":
        return stratum_scores.min(axis=0)
    if mode == "stratum_cvar_50":
        tail_count = max(1, int(np.ceil(0.5 * len(stratum_scores))))
        return np.sort(stratum_scores, axis=0)[:tail_count].mean(axis=0)
    raise KeyError(mode)


def _aggregation_report(
    local_population: np.ndarray,
    terminal_population: np.ndarray,
) -> dict[str, Any]:
    oracle = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(local_population)), local_population.shape
        )
    )
    terminal_pick = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(terminal_population)), terminal_population.shape
        )
    )
    oracle_value = float(local_population[oracle])
    return {
        "local_vs_terminal_program_spearman": _spearman(
            local_population.ravel(), terminal_population.ravel()
        ),
        "oracle_program_indices": oracle,
        "oracle_value": oracle_value,
        "terminal_program_indices": terminal_pick,
        "terminal_selection_regret": float(
            oracle_value - local_population[terminal_pick]
        ),
        "random_expected_regret": float(
            oracle_value - local_population.mean()
        ),
        "coordinate_basins": _coordinate_basins(local_population),
    }


def _selection_trials(
    local: np.ndarray,
    terminal: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    rows = []
    candidate_count = local.shape[1] * local.shape[2]
    methods = ("local", "terminal", "coordinate", "coordinate4", "random")
    for probe_count in (4, 8, 16, 32):
        if probe_count >= len(local):
            continue
        regrets = {name: [] for name in methods}
        values = {name: [] for name in methods}
        for _ in range(repeats):
            train_indices = rng.choice(len(local), probe_count, replace=False)
            test_mask = np.ones(len(local), dtype=bool)
            test_mask[train_indices] = False
            train_local = local[train_indices].mean(axis=0)
            train_terminal = terminal[train_indices].mean(axis=0)
            test_value = local[test_mask].mean(axis=0)
            oracle = float(test_value.max())
            local_pick = tuple(
                int(value)
                for value in np.unravel_index(
                    int(np.argmax(train_local)), train_local.shape
                )
            )
            terminal_pick = tuple(
                int(value)
                for value in np.unravel_index(
                    int(np.argmax(train_terminal)), train_terminal.shape
                )
            )
            random_pick = tuple(
                int(value)
                for value in np.unravel_index(
                    int(rng.integers(candidate_count)), train_local.shape
                )
            )
            coordinate_pick = _coordinate_ascent(train_local, random_pick)[0]
            restart_picks = [
                _coordinate_ascent(
                    train_local,
                    tuple(
                        int(value)
                        for value in np.unravel_index(
                            int(rng.integers(candidate_count)),
                            train_local.shape,
                        )
                    ),
                )[0]
                for _ in range(4)
            ]
            coordinate4_pick = max(
                restart_picks, key=lambda pick: train_local[pick]
            )
            picks = {
                "local": local_pick,
                "terminal": terminal_pick,
                "coordinate": coordinate_pick,
                "coordinate4": coordinate4_pick,
                "random": random_pick,
            }
            for name, pick in picks.items():
                selected = float(test_value[pick])
                values[name].append(selected)
                regrets[name].append(oracle - selected)
        row: dict[str, float | int] = {"probe_count": probe_count}
        for name in methods:
            row[f"{name}_test_value"] = float(np.mean(values[name]))
            row[f"{name}_test_regret"] = float(np.mean(regrets[name]))
        rows.append(row)
    return rows


def analyze(
    input_path: Path,
    *,
    source_count: int,
    source_seed: int,
    repeats: int,
    split_seed: int,
) -> dict[str, Any]:
    bank, source = _load_tangent_bank(input_path)
    selectors, margins = _make_sources(count=source_count, seed=source_seed)
    local, terminal = _score_bank(bank, selectors, margins)
    local_population = local.mean(axis=0)
    terminal_population = terminal.mean(axis=0)
    oracle = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(local_population)), local_population.shape
        )
    )
    terminal_pick = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(terminal_population)), terminal_population.shape
        )
    )
    oracle_value = float(local_population[oracle])
    return {
        "schema": "nco-tangent-branch-analysis-v1",
        "source_config_sha256": source["config_sha256"],
        "probe_count": len(bank),
        "selector_count": len(selectors),
        "margin_count": len(margins),
        "combination_count": len(selectors) * len(margins),
        "local_vs_terminal_program_spearman": _spearman(
            local_population.ravel(), terminal_population.ravel()
        ),
        "oracle_program_indices": oracle,
        "oracle_tangent_alignment": oracle_value,
        "terminal_program_indices": terminal_pick,
        "terminal_selection_regret": float(
            oracle_value - local_population[terminal_pick]
        ),
        "random_expected_regret": float(
            oracle_value - local_population.mean()
        ),
        "coordinate_basins": _coordinate_basins(local_population),
        "aggregation_reports": {
            mode: _aggregation_report(
                _aggregate_scores(local, bank, mode),
                _aggregate_scores(terminal, bank, mode),
            )
            for mode in (
                "mean",
                "stratum_min",
                "stratum_cvar_50",
                "probe_cvar_20",
            )
        },
        "selection_trials": _selection_trials(
            local, terminal, repeats=repeats, seed=split_seed
        ),
        "selected_sources": {
            "oracle_selector": asdict(selectors[oracle[0]]),
            "oracle_margin": asdict(margins[oracle[1]]),
            "terminal_selector": asdict(selectors[terminal_pick[0]]),
            "terminal_margin": asdict(margins[terminal_pick[1]]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-count", type=int, default=32)
    parser.add_argument("--source-seed", type=int, default=20260726)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--split-seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        args.input,
        source_count=int(args.source_count),
        source_seed=int(args.source_seed),
        repeats=int(args.repeats),
        split_seed=int(args.split_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
