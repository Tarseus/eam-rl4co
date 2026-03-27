import zipfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

_CACHE_VERSION = 1
_CACHE_NAME = "cached_rl4co_mgl_jssp.pt"


def standardize(input: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    means = input.mean(dim=dim, keepdim=True)
    stds = input.std(dim=dim, keepdim=True) + eps
    return (input - means) / stds


def _detect_machine_index_offset(machine_ids: list[int], num_machines: int) -> int:
    if not machine_ids:
        return 0
    min_machine = min(machine_ids)
    max_machine = max(machine_ids)
    if min_machine == 0 and max_machine <= num_machines - 1:
        return 0
    if min_machine >= 1 and max_machine <= num_machines:
        return 1
    raise ValueError(
        f"Unsupported machine indexing range [{min_machine}, {max_machine}] for {num_machines} machines."
    )


def _read_basic(path: str, device: str = "cpu") -> tuple[str, int, int, torch.Tensor, float]:
    fpath = Path(path)
    with fpath.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh.readlines() if line.strip()]

    first = lines[0].split()
    num_jobs = int(first[0])
    num_machines = int(first[1])
    parsed_jobs = [[int(token) for token in line.split()] for line in lines[1 : 1 + num_jobs]]
    machine_ids = [row[idx] for row in parsed_jobs for idx in range(0, len(row), 2)]
    machine_offset = _detect_machine_index_offset(machine_ids, num_machines)

    instance = torch.empty((num_jobs, 2 * num_machines), dtype=torch.float32, device=device)
    for job_idx, row in enumerate(parsed_jobs):
        if len(row) != 2 * num_machines:
            raise ValueError(
                f"Expected {2 * num_machines} values in job row, got {len(row)} for {fpath.as_posix()}"
            )
        fixed = row[:]
        for idx in range(0, len(fixed), 2):
            fixed[idx] -= machine_offset
        instance[job_idx] = torch.tensor(fixed, dtype=torch.float32, device=device)

    makespan = 0.0
    if len(lines) > 1 + num_jobs:
        tail = lines[1 + num_jobs].split()
        if len(tail) == 1:
            makespan = float(tail[0])

    return fpath.stem, num_jobs, num_machines, instance, makespan


def cluster_edges(
    num_jobs: int, num_machines: int, machines: torch.Tensor, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    edges_job = []
    for job_idx in range(num_jobs):
        for machine_idx in range(num_machines - 1):
            left = job_idx * num_machines + machine_idx
            right = left + 1
            edges_job.append((left, right))
            edges_job.append((right, left))

    edges_machine = []
    for machine_idx in range(num_machines):
        assigned = (machines == machine_idx).view(-1).nonzero().squeeze(-1)
        for src_idx in range(len(assigned) - 1):
            for dst_idx in range(src_idx + 1, len(assigned)):
                src = assigned[src_idx].item()
                dst = assigned[dst_idx].item()
                edges_machine.append((src, dst))
                edges_machine.append((dst, src))

    return (
        torch.tensor(edges_job, dtype=torch.long, device=device).t().contiguous(),
        torch.tensor(edges_machine, dtype=torch.long, device=device).t().contiguous(),
    )


def extract_features(
    num_jobs: int, num_machines: int, costs_t: torch.Tensor, machines_t: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    quantiles = torch.tensor([0.25, 0.5, 0.75], device=device)
    max_cost = costs_t.max().clamp_min(1.0)
    costs = costs_t / max_cost

    feat_job = torch.quantile(costs, quantiles, dim=1).T
    machine_costs = torch.empty((num_machines, num_jobs), dtype=torch.float32, device=device)
    for machine_idx in range(num_machines):
        machine_costs[machine_idx] = costs[machines_t == machine_idx]
    feat_machine = torch.quantile(machine_costs, quantiles, dim=1).T

    job_sum = costs.sum(dim=1, keepdim=True)
    cumsum = costs.cumsum(dim=1)
    completion = cumsum / job_sum.clamp_min(1e-8)
    remaining = (job_sum - cumsum + costs) / job_sum.clamp_min(1e-8)
    pos_job = costs.unsqueeze(-1) - feat_job.unsqueeze(1)
    pos_machine = costs.unsqueeze(-1) - feat_machine[machines_t]

    features = torch.cat(
        [
            feat_job.repeat_interleave(num_machines, 0),
            feat_machine[machines_t.view(-1)],
            costs.view(-1, 1),
            completion.view(-1, 1),
            remaining.view(-1, 1),
            pos_job.view(-1, 3),
            pos_machine.view(-1, 3),
        ],
        dim=1,
    )
    return standardize(features, dim=0)


def load_instance(path: str, device: str = "cpu") -> dict:
    name, num_jobs, num_machines, instance, makespan = _read_basic(path, device=device)
    costs = instance[:, 1::2]
    machines = instance[:, :-1:2].long()
    job_edges, mac_edges = cluster_edges(num_jobs, num_machines, machines, device=device)
    x = extract_features(num_jobs, num_machines, costs, machines, device=device)
    return {
        "name": name,
        "path": str(Path(path)),
        "j": num_jobs,
        "m": num_machines,
        "shape": f"{num_jobs}x{num_machines}",
        "x": x.to(device),
        "job_edges": job_edges,
        "mac_edges": mac_edges,
        "costs": costs,
        "machines": machines,
        "makespan": makespan,
    }


def _ensure_unzipped(data_dir: str) -> None:
    root = Path(data_dir)
    if any(root.glob("*.jsp")):
        return
    for zip_path in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)


def load_dataset(data_dir: str, use_cached: bool = True, device: str = "cpu") -> list[dict]:
    _ensure_unzipped(data_dir)
    root = Path(data_dir)
    cache_path = root / _CACHE_NAME
    if use_cached and cache_path.exists():
        cached = torch.load(cache_path, map_location=device, weights_only=False)
        if isinstance(cached, dict) and cached.get("cache_version") == _CACHE_VERSION:
            return cached["instances"]

    instances = []
    for file in sorted(root.iterdir()):
        if file.name.startswith(".") or file.suffix.lower() != ".jsp":
            continue
        instances.append(load_instance(file.as_posix(), device=device))
    torch.save({"cache_version": _CACHE_VERSION, "instances": instances}, cache_path)
    return instances


class JSSPInstanceDataset(Dataset):
    def __init__(self, instances: list[dict]):
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        return self.instances[idx]

    @staticmethod
    def collate_fn(items: list[dict]) -> dict:
        if len(items) != 1:
            raise ValueError("MGLJSSPModel expects dataloader batch_size=1 to stay protocol-faithful.")
        return items[0]
