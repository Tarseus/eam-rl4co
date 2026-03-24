import os
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_CACHE_VERSION = 2


def standardize(input: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    means = input.mean(dim=dim, keepdim=True)
    stds = input.std(dim=dim, keepdim=True) + eps
    return (input - means) / stds


def _read_basic(fpath: str):
    with open(fpath, encoding="utf-8") as f:
        lines = f.readlines()
    name = os.path.basename(fpath).split(".", 1)[0]

    n_str, m_str, _ = re.split(r"[ \t]+", lines[0].strip())
    num_jobs, num_machines = int(n_str), int(m_str)

    max_num_ops = 0
    process_times = []
    for job_idx in range(1, num_jobs + 1):
        data = [int(token) for token in re.split(r"[ \t]+", lines[job_idx].strip())]
        num_ops = data[0]
        max_num_ops = max(max_num_ops, num_ops)
        job_proc_times = []

        ptr = 1
        for _ in range(num_ops):
            op_proc_time = [-1 for _ in range(num_machines)]
            num_assignable = data[ptr]
            for _ in range(num_assignable):
                machine = data[ptr + 1] - 1
                proc_time = data[ptr + 2]
                op_proc_time[machine] = proc_time
                ptr += 2
            ptr += 1
            job_proc_times.append(op_proc_time)
        process_times.append(job_proc_times)

    for job in process_times:
        if len(job) < max_num_ops:
            # Pad missing operations as fully invalid so they do not become zero-cost
            # executable operations on machine 0.
            job += [[-1 for _ in range(num_machines)] for _ in range(max_num_ops - len(job))]
    process_times = np.array(process_times, dtype=np.int32)
    makespan = int(lines[num_jobs + 1]) if len(lines) > num_jobs + 1 else None

    process_times = np.stack(
        [process_times[:, :, machine] for machine in range(num_machines) if np.any(process_times[:, :, machine] >= 0)],
        axis=2,
    )
    return name, process_times, makespan


def _cluster_edges(data: np.ndarray, device: str = "cpu"):
    num_jobs, num_ops, num_machines = data.shape
    node_count = -1

    def _edge_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def count() -> int:
        nonlocal node_count
        node_count += 1
        return node_count

    ops = {
        (job_idx, op_idx, machine_idx): count()
        for job_idx in range(num_jobs)
        for op_idx in range(num_ops)
        for machine_idx in range(num_machines)
        if data[job_idx, op_idx, machine_idx] >= 0
    }

    edges_job = []
    edges_ops = []
    for job_idx in range(num_jobs):
        prev_level = []
        curr_level = []
        for op_idx in range(num_ops):
            for machine_idx in range(num_machines):
                if data[job_idx, op_idx, machine_idx] < 0:
                    continue
                curr = ops[(job_idx, op_idx, machine_idx)]
                for prev in prev_level:
                    edges_job.append((prev, curr))
                for prev in curr_level:
                    edges_ops.append((prev, curr))
                    edges_ops.append((curr, prev))
                curr_level.append(curr)
            prev_level = curr_level
            curr_level = []

    edges_machine = []
    for machine_idx in range(num_machines):
        assigned = []
        machine = data[:, :, machine_idx]
        for job_idx in range(num_jobs):
            for op_idx in range(num_ops):
                if machine[job_idx, op_idx] < 0:
                    continue
                curr = ops[(job_idx, op_idx, machine_idx)]
                for prev in assigned:
                    edges_machine.append((prev, curr))
                    edges_machine.append((curr, prev))
                assigned.append(curr)

    return (
        _edge_tensor(edges_job),
        _edge_tensor(edges_ops),
        _edge_tensor(edges_machine),
    )


def _extract_features(data: np.ndarray, device: str = "cpu") -> torch.Tensor:
    num_jobs, num_ops, num_machines = data.shape
    num_valid_ops = len(data[data >= 0])
    quantiles = np.array([0.25, 0.5, 0.75])
    max_cost = data.max()
    data = data / max_cost
    valid_mask = data >= 0

    feat_job_q = np.zeros((num_valid_ops, 3), dtype=np.float32)
    feat_job_d = np.zeros((num_valid_ops, 3), dtype=np.float32)
    count = 0
    for job_idx in range(num_jobs):
        job_data = data[job_idx]
        costs = job_data[job_data >= 0]
        job_q = np.quantile(costs, quantiles).T
        feat_job_q[count] = job_q
        for cost in costs:
            feat_job_q[count] = job_q
            feat_job_d[count] = cost - job_q
            count += 1

    feat_machine_q = np.zeros((num_valid_ops, 3), dtype=np.float32)
    feat_machine_d = np.zeros((num_valid_ops, 3), dtype=np.float32)
    machine_q = np.zeros((num_machines, 3), dtype=np.float32)
    count = 0
    for machine_idx in range(num_machines):
        machine_data = data[:, :, machine_idx]
        costs = machine_data[machine_data >= 0]
        machine_q[machine_idx] = np.quantile(costs, quantiles).T
    for job_idx, op_idx, machine_idx in zip(*np.where(data >= 0)):
        feat_machine_q[count] = machine_q[machine_idx]
        feat_machine_d[count] = data[job_idx, op_idx, machine_idx] - machine_q[machine_idx]
        count += 1

    costs = data[valid_mask]
    valid_count = valid_mask.sum(axis=2)
    valid_sum = np.where(valid_mask, data, 0.0).sum(axis=2)
    avg_cost = np.divide(valid_sum, np.maximum(valid_count, 1), dtype=np.float32)
    avg_cost = np.where(valid_count > 0, avg_cost, 0.0)
    avg_sum = np.sum(avg_cost, axis=1, keepdims=True)
    avg_cumsum = np.cumsum(avg_cost, axis=1)
    avg_completion = np.divide(avg_cumsum, np.maximum(avg_sum, 1e-8), dtype=np.float32)
    avg_remain = np.divide(
        avg_sum - avg_cumsum + avg_cost,
        np.maximum(avg_sum, 1e-8),
        dtype=np.float32,
    )

    feat_ops = np.zeros((num_valid_ops, 3), dtype=np.float32)
    for idx, (job_idx, op_idx, _machine_idx) in enumerate(zip(*np.where(data >= 0))):
        feat_ops[idx, 0] = costs[idx]
        feat_ops[idx, 1] = avg_completion[job_idx, op_idx]
        feat_ops[idx, 2] = avg_remain[job_idx, op_idx]

    features = np.concatenate(
        [feat_ops, feat_job_q, feat_job_d, feat_machine_q, feat_machine_d],
        axis=1,
    )
    return standardize(torch.tensor(features, dtype=torch.float32, device=device), dim=0)


def load_instance(path: str, device: str = "cpu") -> dict:
    name, instance, makespan = _read_basic(path)
    num_jobs, num_ops, num_machines = instance.shape
    edges_job, edges_ops, edges_machine = _cluster_edges(instance, device=device)
    return {
        "name": name,
        "path": path,
        "j": num_jobs,
        "o": num_ops,
        "m": num_machines,
        "shape": f"{num_jobs}x{num_ops}x{num_machines}",
        "x": _extract_features(instance, device=device),
        "job_edges": edges_job,
        "ops_edges": edges_ops,
        "mac_edges": edges_machine,
        "data": instance,
        "makespan": makespan,
    }


def _ensure_unzipped(data_dir: str) -> None:
    path = Path(data_dir)
    if any(path.glob("*.fjs")):
        return
    for zip_path in sorted(path.glob("*.zip")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path)


def load_dataset(data_dir: str, use_cached: bool = True, device: str = "cpu") -> list[dict]:
    _ensure_unzipped(data_dir)
    data_path = Path(data_dir)
    cache_path = data_path / "cached.pt"
    if use_cached and cache_path.exists():
        cached = torch.load(cache_path, map_location=device, weights_only=False)
        if isinstance(cached, dict) and cached.get("cache_version") == _CACHE_VERSION:
            return cached["instances"]
        if isinstance(cached, list):
            # Legacy cache from older builds; rebuild to avoid stale preprocessing bugs.
            pass
        else:
            raise ValueError(f"Unexpected cache payload in {cache_path}")

    instances = []
    for file in sorted(data_path.iterdir()):
        if file.name.startswith(".") or file.suffix.lower() != ".fjs":
            continue
        instances.append(load_instance(str(file), device=device))
    torch.save({"cache_version": _CACHE_VERSION, "instances": instances}, cache_path)
    return instances


class FJSPInstanceDataset(Dataset):
    def __init__(self, instances: list[dict]):
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        return self.instances[idx]

    @staticmethod
    def collate_fn(items: list[dict]) -> dict:
        if len(items) != 1:
            raise ValueError("BOPOFJSPModel expects dataloader batch_size=1 to stay paper-faithful.")
        return items[0]
