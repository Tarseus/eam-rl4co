from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "rfps_feature_pilot"
MINIMALITY = HERE.parent / "rfps_minimality"
sys.path[:0] = [str(PILOT), str(MINIMALITY)]

import run_pilot as pilot  # noqa: E402
from run_two_point_baselines import sphere_parallel_transport, unit  # noqa: E402


ELL = 0.03
MIN_ESS_RATIO = 0.40
N_ORDERS = 200
SEED = 20260728
CONDITIONS = (
    "uniform",
    "anchor_is",
    "all_pair_endpoint",
    "gap_endpoint",
    "anchor_gap_endpoint",
    "anchor_hard_endpoint",
)
ANCHORS = ("scratch", "warm")
METHODS = ("one_point", "euclidean_two_point", "fisher_two_point")
SOURCE_RUN = PILOT / "results_dual_scratch_seed1_curvature"
TARGET_RUNS = {
    "scratch": PILOT / "results_dual_scratch_seed1_curvature",
    "warm": PILOT / "results_dual_ckpt135_seed1_curvature",
}


def read_descriptors(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "descriptors.npz") as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_candidates() -> tuple[list[pilot.Candidate], np.ndarray, dict[str, np.ndarray]]:
    metadata = json.loads(
        (SOURCE_RUN / "results.json").read_text(encoding="utf-8")
    )
    candidates, failures = pilot.load_candidates(
        Path(metadata["data_source"]), metadata["score_target"]
    )
    if failures:
        raise RuntimeError(f"candidate loading failed: {failures[:3]}")
    source = read_descriptors(SOURCE_RUN)
    by_key = {candidate.key: candidate for candidate in candidates}
    ordered = [by_key[str(key)] for key in source["keys"]]
    targets: dict[str, np.ndarray] = {}
    for name, path in TARGET_RUNS.items():
        data = read_descriptors(path)
        index = {str(key): position for position, key in enumerate(data["keys"])}
        targets[name] = np.asarray(
            [data["scores"][index[str(key)]] for key in source["keys"]]
        )
    return ordered, source["keys"], targets


def load_shared_probes(
    shared_dir: Path,
    *,
    probe_limit: int | None = None,
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for seed_index in (1, 2):
        with np.load(shared_dir / f"shared_seed{seed_index}.npz") as archive:
            arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
        for instance in range(arrays["objective"].shape[0]):
            probes.append(
                {
                    "seed_index": seed_index,
                    "instance": instance,
                    "objective": arrays["objective"][instance],
                    "p_scratch": arrays["p_scratch"][instance],
                    "p_warm": arrays["p_warm"][instance],
                    "log_mu": arrays["log_mu"][instance],
                    "seq_len": arrays["seq_len"][instance],
                    "advantage": arrays["advantage"][instance],
                    "entropy": arrays["entropy"][instance],
                }
            )
    return probes[:probe_limit] if probe_limit else probes


def normalized_ess(q: np.ndarray | torch.Tensor) -> float:
    values = np.asarray(q, dtype=np.float64)
    return float(1.0 / np.square(values).sum() / values.size)


def ensure_ess(q: np.ndarray, minimum: float = MIN_ESS_RATIO) -> tuple[np.ndarray, float]:
    q = np.asarray(q, dtype=np.float64)
    # Fisher--Rao geometry is defined on the simplex interior. Cross-policy
    # likelihood ratios can underflow to exact zero, so preserve full support
    # before checking whether additional uniform mixing is needed.
    q = np.maximum(q, pilot.T_FLOOR)
    q = q / q.sum()
    if normalized_ess(q) >= minimum:
        return q, 0.0
    uniform = np.full_like(q, 1.0 / q.size)
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        mixed = (1.0 - middle) * q + middle * uniform
        if normalized_ess(mixed) >= minimum:
            high = middle
        else:
            low = middle
    mixed = (1.0 - high) * q + high * uniform
    return mixed / mixed.sum(), high


def endpoint_marginal(
    objective: np.ndarray,
    base: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    objective = np.asarray(objective)
    base = np.asarray(base, dtype=np.float64)
    pair = base[:, None] * base[None, :] * exposure
    pair *= objective[:, None] < objective[None, :]
    total = pair.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("pair exposure has no positive mass")
    marginal = (pair.sum(axis=0) + pair.sum(axis=1)) / (2.0 * total)
    return marginal / marginal.sum()


def reference_measures(probe: dict[str, Any], anchor: str) -> dict[str, dict[str, Any]]:
    objective = np.asarray(probe["objective"], dtype=np.float64)
    p_anchor = np.asarray(probe[f"p_{anchor}"], dtype=np.float64)
    log_mu = np.asarray(probe["log_mu"], dtype=np.float64)
    size = objective.size
    uniform = np.full(size, 1.0 / size)
    log_ratio = p_anchor - log_mu
    log_ratio -= np.max(log_ratio)
    anchor_base = np.exp(log_ratio)
    anchor_base /= anchor_base.sum()

    order = np.argsort(objective)
    ranks = np.empty(size, dtype=np.float64)
    ranks[order] = np.arange(size, dtype=np.float64)
    gap = np.abs(ranks[:, None] - ranks[None, :]) / max(1, size - 1)
    all_exposure = np.ones((size, size), dtype=np.float64)
    hard = expit(-0.05 * (p_anchor[:, None] - p_anchor[None, :]))

    raw = {
        "uniform": uniform,
        "anchor_is": anchor_base,
        "all_pair_endpoint": endpoint_marginal(
            objective, uniform, all_exposure
        ),
        "gap_endpoint": endpoint_marginal(objective, uniform, gap),
        "anchor_gap_endpoint": endpoint_marginal(
            objective, anchor_base, gap
        ),
        "anchor_hard_endpoint": endpoint_marginal(
            objective, anchor_base, hard
        ),
    }
    output: dict[str, dict[str, Any]] = {}
    for name, values in raw.items():
        q, mixture = ensure_ess(values)
        output[name] = {
            "q": torch.as_tensor(q, dtype=pilot.DTYPE),
            "raw_ess": normalized_ess(values),
            "ess": normalized_ess(q),
            "uniform_mixture": mixture,
            "max_over_uniform": float(q.max() * size),
        }
    return output


def make_probe(probe: dict[str, Any], anchor: str) -> pilot.Probe:
    objective = torch.as_tensor(
        probe["objective"][None, :], dtype=pilot.DTYPE
    )
    return pilot.Probe(
        objective=objective,
        p0=torch.as_tensor(probe[f"p_{anchor}"][None, :], dtype=pilot.DTYPE),
        seq_len=torch.as_tensor(probe["seq_len"][None, :], dtype=pilot.DTYPE),
        advantage=torch.as_tensor(
            probe["advantage"][None, :], dtype=pilot.DTYPE
        ),
        entropy=torch.as_tensor(probe["entropy"][None, :], dtype=pilot.DTYPE),
    )


def coefficient_field(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    q: torch.Tensor,
    q_reference: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    log_density_ratio = torch.log(q.clamp_min(pilot.T_FLOOR)) - torch.log(
        q_reference.clamp_min(pilot.T_FLOOR)
    )
    p = (probe.p0 + log_density_ratio.unsqueeze(0)).detach().requires_grad_(True)
    batch, model_output = pilot.pairwise_inputs_importance(probe, p, q)
    loss = candidate.compiled.loss_fn(
        batch, model_output, {"alpha": candidate.alpha}
    )
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss, dtype=pilot.DTYPE)
    if loss.numel() != 1:
        loss = loss.mean()
    gradient = torch.autograd.grad(loss, p, allow_unused=True)[0]
    c = torch.zeros_like(q) if gradient is None else -gradient.reshape(-1)
    u = c - q * c.sum()
    speed = float(
        torch.sqrt((u.square() / q.clamp_min(pilot.T_FLOOR)).sum()).item()
    )
    return u.detach(), speed


def descriptor_blocks(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    q0: torch.Tensor,
) -> dict[str, np.ndarray]:
    u0, speed0 = coefficient_field(candidate, probe, q0, q0)
    h0 = torch.sqrt(q0)
    v0 = u0 / (2.0 * h0)
    d0 = unit(v0)

    z1 = torch.log(q0) + ELL * unit(u0)
    q_euclidean = torch.softmax(z1, dim=0)
    u_euclidean, _ = coefficient_field(candidate, probe, q_euclidean, q0)
    euclidean = torch.cat([d0, unit(u_euclidean)]) / math.sqrt(2.0)

    if speed0 <= 1e-14:
        q_fisher = q0.clone()
    else:
        q_fisher = pilot.retract_fisher_sphere(q0, u0, ELL / speed0)
    u_fisher, _ = coefficient_field(candidate, probe, q_fisher, q0)
    h_fisher = torch.sqrt(q_fisher)
    v_fisher = u_fisher / (2.0 * h_fisher)
    transported = sphere_parallel_transport(h_fisher, h0, v_fisher)
    fisher = torch.cat([d0, unit(transported)]) / math.sqrt(2.0)
    return {
        "one_point": d0.numpy(),
        "euclidean_two_point": euclidean.numpy(),
        "fisher_two_point": fisher.numpy(),
    }


def flatten_blocks(blocks: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    selected = blocks if subset is None else blocks[:, subset, :]
    return selected.reshape(selected.shape[0], -1) / math.sqrt(selected.shape[1])


def descriptor_metrics(matrix: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    distances = pdist(matrix)
    gaps = pdist(scores[:, None])
    rho = float(spearmanr(distances, gaps).statistic)
    square = squareform(distances)
    np.fill_diagonal(square, np.inf)
    nearest = np.argmin(square, axis=1)
    nn_gap = np.abs(scores - scores[nearest])
    return {
        "rho": rho,
        "nn_median": float(np.median(nn_gap)),
        "nn_mean": float(np.mean(nn_gap)),
    }


def order_risk(
    matrix: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    distance = squareform(pdist(matrix))
    rng = np.random.default_rng(seed)
    false_rates = []
    skipped_mae = []
    for _ in range(N_ORDERS):
        order = rng.permutation(len(scores))
        nearest_distance = []
        nearest_gap = []
        for position in range(1, len(order)):
            current = int(order[position])
            previous = order[:position]
            neighbor = int(previous[np.argmin(distance[current, previous])])
            nearest_distance.append(distance[current, neighbor])
            nearest_gap.append(abs(scores[current] - scores[neighbor]))
        distances = np.asarray(nearest_distance)
        gaps = np.asarray(nearest_gap)
        selected = distances <= np.quantile(distances, 0.20)
        false_rates.append(float(np.mean(gaps[selected] > 0.01)))
        skipped_mae.append(float(np.mean(gaps[selected])))
    return {
        "false_skip_mean": float(np.mean(false_rates)),
        "false_skip_q90": float(np.quantile(false_rates, 0.90)),
        "skipped_mae_mean": float(np.mean(skipped_mae)),
    }


def agreement(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_dist = pdist(first)
    second_dist = pdist(second)
    first_square = squareform(first_dist)
    second_square = squareform(second_dist)
    np.fill_diagonal(first_square, np.inf)
    np.fill_diagonal(second_square, np.inf)
    first_order = np.argsort(first_square, axis=1)
    second_order = np.argsort(second_square, axis=1)
    return {
        "ef_distance_rank": float(spearmanr(first_dist, second_dist).statistic),
        "ef_top1": float(np.mean(first_order[:, 0] == second_order[:, 0])),
        "ef_top5_overlap": float(
            np.mean(
                [
                    len(
                        set(first_order[i, :5]).intersection(
                            second_order[i, :5]
                        )
                    )
                    / 5.0
                    for i in range(first.shape[0])
                ]
            )
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", type=Path, default=HERE / "shared_banks")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--probe-limit", type=int)
    args = parser.parse_args()
    torch.set_default_dtype(pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    candidates, keys, targets = load_candidates()
    if args.candidate_limit:
        candidates = candidates[: args.candidate_limit]
        keys = keys[: args.candidate_limit]
        targets = {
            name: values[: args.candidate_limit] for name, values in targets.items()
        }
    shared = load_shared_probes(args.shared_dir, probe_limit=args.probe_limit)

    raw: dict[tuple[str, str, str], list[list[np.ndarray]]] = {}
    q_stats: list[dict[str, Any]] = []
    max_all_pair_error = 0.0
    measure_cache: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for probe_index, probe_data in enumerate(shared):
        for anchor in ANCHORS:
            measures = reference_measures(probe_data, anchor)
            measure_cache[(probe_index, anchor)] = measures
            max_all_pair_error = max(
                max_all_pair_error,
                float(
                    torch.max(
                        torch.abs(
                            measures["uniform"]["q"]
                            - measures["all_pair_endpoint"]["q"]
                        )
                    ).item()
                ),
            )
            for condition, stats in measures.items():
                q_stats.append(
                    {
                        "probe": probe_index,
                        "seed_index": probe_data["seed_index"],
                        "anchor": anchor,
                        "condition": condition,
                        "raw_ess": stats["raw_ess"],
                        "ess": stats["ess"],
                        "uniform_mixture": stats["uniform_mixture"],
                        "max_over_uniform": stats["max_over_uniform"],
                    }
                )

    for candidate_index, candidate in enumerate(candidates):
        candidate_store: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
        for probe_index, probe_data in enumerate(shared):
            for anchor in ANCHORS:
                probe = make_probe(probe_data, anchor)
                measures = measure_cache[(probe_index, anchor)]
                for condition in CONDITIONS:
                    blocks = descriptor_blocks(
                        candidate, probe, measures[condition]["q"]
                    )
                    for method, vector in blocks.items():
                        candidate_store[(anchor, condition, method)].append(vector)
        for key, vectors in candidate_store.items():
            raw.setdefault(key, []).append(vectors)
        print(
            f"[nonuniform-q0] {candidate_index + 1:03d}/{len(candidates):03d}",
            flush=True,
        )

    matrices = {
        key: np.asarray(candidate_vectors)
        for key, candidate_vectors in raw.items()
    }
    rows: list[dict[str, Any]] = []
    subsets = {
        "all": np.arange(len(shared)),
        "seed1": np.asarray(
            [index for index, probe in enumerate(shared) if probe["seed_index"] == 1]
        ),
        "seed2": np.asarray(
            [index for index, probe in enumerate(shared) if probe["seed_index"] == 2]
        ),
    }
    for subset_name, subset in subsets.items():
        if subset.size == 0:
            continue
        for anchor in ANCHORS:
            for condition in CONDITIONS:
                euclidean_matrix = flatten_blocks(
                    matrices[(anchor, condition, "euclidean_two_point")], subset
                )
                fisher_matrix = flatten_blocks(
                    matrices[(anchor, condition, "fisher_two_point")], subset
                )
                ef = agreement(euclidean_matrix, fisher_matrix)
                for target_name, scores in targets.items():
                    for method in METHODS:
                        matrix = flatten_blocks(
                            matrices[(anchor, condition, method)], subset
                        )
                        row: dict[str, Any] = {
                            "subset": subset_name,
                            "anchor": anchor,
                            "target": target_name,
                            "condition": condition,
                            "method": method,
                        }
                        row.update(descriptor_metrics(matrix, scores))
                        row.update(
                            order_risk(
                                matrix,
                                scores,
                                seed=SEED
                                + (0 if subset_name == "all" else int(subset[-1])),
                            )
                        )
                        row.update(ef)
                        rows.append(row)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "q_stats.csv", q_stats)
    np.savez_compressed(
        output / "descriptors.npz",
        keys=keys,
        **{
            "__".join(key): value
            for key, value in matrices.items()
        },
    )
    payload = {
        "ell": ELL,
        "n_candidates": len(candidates),
        "n_probes": len(shared),
        "max_all_pair_uniform_error": max_all_pair_error,
        "metrics": rows,
        "q_stats": q_stats,
    }
    (output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary = [
        row
        for row in rows
        if row["subset"] == "all"
        and row["anchor"] == row["target"]
        and row["method"] == "fisher_two_point"
    ]
    lines = [
        "# Non-uniform q0 results",
        "",
        f"- candidates: {len(candidates)}",
        f"- probes: {len(shared)}",
        f"- all-pair/uniform max error: {max_all_pair_error:.3e}",
        "",
        "| Anchor | Condition | rho | false skip | NN median | E/F rank | E/F top-1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {anchor} | {condition} | {rho:.3f} | {false_skip_mean:.3f} | "
            "{nn_median:.5f} | {ef_distance_rank:.3f} | {ef_top1:.3f} |".format(
                **row
            )
        )
    summary = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
