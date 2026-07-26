"""Stress-test matched-branch program selection with nonlinear source programs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_matched_branch_probe import (
    BranchProbe,
    EPS,
    _load_bank,
    _matrices,
    _softmax,
    _spearman,
)


FEATURE_NAMES = (
    "bias",
    "objective_gap",
    "rank_span",
    "rank_midpoint",
    "depth_fraction",
    "gap_x_depth",
    "span_x_depth",
    "gap_squared",
    "span_squared",
    "sin_rank_midpoint",
)


@dataclass(frozen=True)
class SelectorSource:
    coefficients: tuple[float, ...]
    density: float
    all_edges: bool = False


@dataclass(frozen=True)
class MarginSource:
    coefficients: tuple[float, ...]
    amplitude: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _edge_features(
    probe: BranchProbe,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    count = len(probe.normalized_cost)
    denominator = max(count - 1, 1)
    winner = np.asarray([i for i, _ in edges], dtype=float)
    loser = np.asarray([j for _, j in edges], dtype=float)
    gap = np.asarray(
        [probe.normalized_cost[j] - probe.normalized_cost[i] for i, j in edges]
    )
    span = (loser - winner) / denominator
    midpoint = (winner + loser) / (2.0 * denominator)
    depth = np.full(len(edges), probe.depth_fraction)
    return np.stack(
        (
            np.ones(len(edges)),
            gap,
            span,
            midpoint,
            depth,
            gap * depth,
            span * depth,
            gap**2,
            span**2,
            np.sin(np.pi * midpoint),
        ),
        axis=-1,
    )


def _make_sources(
    *, count: int, seed: int
) -> tuple[list[SelectorSource], list[MarginSource]]:
    if count < 2:
        raise ValueError("source count must be at least two")
    rng = np.random.default_rng(seed)
    selectors = [
        SelectorSource(
            coefficients=tuple(0.0 for _ in FEATURE_NAMES),
            density=1.0,
            all_edges=True,
        )
    ]
    margins = [
        MarginSource(
            coefficients=tuple(0.0 for _ in FEATURE_NAMES),
            amplitude=0.0,
        )
    ]
    for _ in range(count - 1):
        selector_coefficients = rng.normal(size=len(FEATURE_NAMES))
        selector_coefficients /= max(
            float(np.linalg.norm(selector_coefficients)), EPS
        )
        selectors.append(
            SelectorSource(
                coefficients=tuple(float(v) for v in selector_coefficients),
                density=float(rng.uniform(0.1, 0.9)),
            )
        )
        margin_coefficients = rng.normal(size=len(FEATURE_NAMES))
        margin_coefficients /= max(
            float(np.linalg.norm(margin_coefficients)), EPS
        )
        margins.append(
            MarginSource(
                coefficients=tuple(float(v) for v in margin_coefficients),
                amplitude=float(rng.uniform(0.25, 4.0)),
            )
        )
    return selectors, margins


def _selector_masks(
    features: np.ndarray,
    selectors: list[SelectorSource],
) -> np.ndarray:
    masks = np.zeros((len(selectors), len(features)), dtype=bool)
    for index, source in enumerate(selectors):
        if source.all_edges:
            masks[index] = True
            continue
        score = features @ np.asarray(source.coefficients)
        keep = max(1, int(round(source.density * len(score))))
        chosen = np.argpartition(score, -keep)[-keep:]
        masks[index, chosen] = True
    return masks


def _margin_values(
    features: np.ndarray,
    margins: list[MarginSource],
) -> np.ndarray:
    values = np.zeros((len(margins), len(features)), dtype=float)
    for index, source in enumerate(margins):
        linear = features @ np.asarray(source.coefficients)
        values[index] = source.amplitude * np.tanh(linear)
    return values


def _candidate_values(
    probe: BranchProbe,
    selectors: list[SelectorSource],
    margins: list[MarginSource],
    *,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    gap_matrix, interval, edges = _matrices(len(probe.local_logp))
    features = _edge_features(probe, edges)
    selector = _selector_masks(features, selectors)
    margin = _margin_values(features, margins)
    objective_gap = features[:, FEATURE_NAMES.index("objective_gap")]
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
    selector_scale = selector[:, None, :] / (
        4.0 * normalizer[:, None, None]
    )

    def values(log_scores: np.ndarray) -> np.ndarray:
        pair_margin = interval @ (gap_matrix @ log_scores)
        pressure = selector_scale * _sigmoid(
            target - pair_margin[None, None, :]
        )
        boundary = np.einsum("sme,ek->smk", pressure, interval)
        direction = np.einsum("smk,kn->smn", boundary, gap_matrix)
        after_scores = log_scores[None, None, :] + step_size * direction
        shifted = after_scores - after_scores.max(axis=-1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=-1, keepdims=True)
        before = float(_softmax(log_scores) @ probe.normalized_cost)
        after = np.einsum("smn,n->sm", probability, probe.normalized_cost)
        return before - after

    return values(probe.local_logp), values(probe.terminal_logp)


def _value_bank(
    bank: list[BranchProbe],
    selectors: list[SelectorSource],
    margins: list[MarginSource],
    *,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(bank), len(selectors), len(margins))
    local = np.empty(shape, dtype=float)
    terminal = np.empty(shape, dtype=float)
    for index, probe in enumerate(bank):
        local[index], terminal[index] = _candidate_values(
            probe, selectors, margins, step_size=step_size
        )
    return local, terminal


def _selection_trials(
    local: np.ndarray,
    terminal: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    candidate_count = local.shape[1] * local.shape[2]
    for probe_count in (8, 16, 32, 64, 128):
        if probe_count >= len(local):
            continue
        regret = {
            name: []
            for name in (
                "local",
                "terminal",
                "staged",
                "coordinate",
                "coordinate4",
                "random",
            )
        }
        value = {name: [] for name in regret}
        for _ in range(repeats):
            train_indices = rng.choice(len(local), probe_count, replace=False)
            test_mask = np.ones(len(local), dtype=bool)
            test_mask[train_indices] = False
            train_local = local[train_indices].mean(axis=0)
            train_terminal = terminal[train_indices].mean(axis=0)
            test_value = local[test_mask].mean(axis=0)
            oracle = float(test_value.max())
            picks = {
                "local": np.unravel_index(
                    int(np.argmax(train_local)), train_local.shape
                ),
                "terminal": np.unravel_index(
                    int(np.argmax(train_terminal)), train_terminal.shape
                ),
            }
            staged_selector = int(np.argmax(train_local[:, 0]))
            picks["staged"] = (
                staged_selector,
                int(np.argmax(train_local[staged_selector])),
            )
            random_pick = tuple(
                int(value)
                for value in np.unravel_index(
                    int(rng.integers(candidate_count)), train_local.shape
                )
            )
            picks["random"] = random_pick
            picks["coordinate"] = _coordinate_ascent(
                train_local, random_pick
            )[0]
            coordinate_restarts = [
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
            picks["coordinate4"] = max(
                coordinate_restarts, key=lambda pick: train_local[pick]
            )
            for name, pick in picks.items():
                selected_value = float(test_value[pick])
                value[name].append(selected_value)
                regret[name].append(oracle - selected_value)
        row: dict[str, float | int] = {"probe_count": probe_count}
        for name in regret:
            row[f"{name}_test_value"] = float(np.mean(value[name]))
            row[f"{name}_test_regret"] = float(np.mean(regret[name]))
        rows.append(row)
    return rows


def _coordinate_ascent(
    values: np.ndarray,
    start: tuple[int, int],
    *,
    max_rounds: int = 64,
) -> tuple[tuple[int, int], int]:
    selector, margin = start
    for round_index in range(1, max_rounds + 1):
        next_selector = int(np.argmax(values[:, margin]))
        next_margin = int(np.argmax(values[next_selector]))
        next_pair = (next_selector, next_margin)
        if next_pair == (selector, margin):
            return next_pair, round_index
        selector, margin = next_pair
    raise RuntimeError("coordinate ascent did not converge")


def _coordinate_basins(values: np.ndarray) -> dict[str, Any]:
    counts: dict[tuple[int, int], int] = {}
    max_rounds = 0
    for selector in range(values.shape[0]):
        for margin in range(values.shape[1]):
            sink, rounds = _coordinate_ascent(values, (selector, margin))
            counts[sink] = counts.get(sink, 0) + 1
            max_rounds = max(max_rounds, rounds)
    oracle = tuple(
        int(value)
        for value in np.unravel_index(int(np.argmax(values)), values.shape)
    )
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "sink_count": len(counts),
        "oracle_basin_fraction": counts.get(oracle, 0) / values.size,
        "max_rounds": max_rounds,
        "largest_sinks": [
            {
                "program_indices": sink,
                "basin_size": count,
                "regret": float(values[oracle] - values[sink]),
            }
            for sink, count in ordered[:10]
        ],
    }


def stress_test(
    input_path: Path,
    *,
    source_count: int,
    source_seed: int,
    step_size: float,
    repeats: int,
    split_seed: int,
) -> dict[str, Any]:
    bank, source = _load_bank(input_path)
    selectors, margins = _make_sources(count=source_count, seed=source_seed)
    local, terminal = _value_bank(
        bank, selectors, margins, step_size=step_size
    )
    local_population = local.mean(axis=0)
    terminal_population = terminal.mean(axis=0)
    local_pick = tuple(
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
    staged_selector = int(np.argmax(local_population[:, 0]))
    staged_pick = (
        staged_selector,
        int(np.argmax(local_population[staged_selector])),
    )
    oracle = float(local_population[local_pick])
    return {
        "schema": "nco-matched-branch-stress-v1",
        "source_config_sha256": source["config_sha256"],
        "probe_count": len(bank),
        "selector_count": len(selectors),
        "margin_count": len(margins),
        "combination_count": len(selectors) * len(margins),
        "local_vs_terminal_program_spearman": _spearman(
            local_population.ravel(), terminal_population.ravel()
        ),
        "oracle_program_indices": local_pick,
        "oracle_mean_improvement": oracle,
        "terminal_program_indices": terminal_pick,
        "terminal_selection_regret": float(
            oracle - local_population[terminal_pick]
        ),
        "staged_program_indices": staged_pick,
        "staged_selection_regret": float(
            oracle - local_population[staged_pick]
        ),
        "random_expected_regret": float(oracle - local_population.mean()),
        "coordinate_basins": _coordinate_basins(local_population),
        "selection_trials": _selection_trials(
            local,
            terminal,
            repeats=repeats,
            seed=split_seed,
        ),
        "selected_sources": {
            "oracle_selector": asdict(selectors[local_pick[0]]),
            "oracle_margin": asdict(margins[local_pick[1]]),
            "terminal_selector": asdict(selectors[terminal_pick[0]]),
            "terminal_margin": asdict(margins[terminal_pick[1]]),
        },
        "feature_names": FEATURE_NAMES,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-count", type=int, default=32)
    parser.add_argument("--source-seed", type=int, default=20260726)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--split-seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = stress_test(
        args.input,
        source_count=int(args.source_count),
        source_seed=int(args.source_seed),
        step_size=float(args.step_size),
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
