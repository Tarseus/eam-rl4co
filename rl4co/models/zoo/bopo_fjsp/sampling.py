from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Solutions:
    mss: torch.Tensor
    logits: torch.Tensor
    trajs: torch.Tensor


def sro_loss(info_better: Solutions, info_worse: Solutions) -> torch.Tensor:
    logits_better = info_better.logits
    targets_better = info_better.trajs
    makespan_better = info_better.mss

    logits_worse = info_worse.logits
    targets_worse = info_worse.trajs
    makespan_worse = info_worse.mss

    batch_size, num_steps, action_dim = logits_better.shape
    logits_better = logits_better.view(-1, action_dim)
    targets_better = targets_better.view(-1)
    logits_worse = logits_worse.view(-1, action_dim)
    targets_worse = targets_worse.view(-1)

    log_prob_better = F.log_softmax(logits_better, dim=-1)
    log_prob_worse = F.log_softmax(logits_worse, dim=-1)

    gathered_better = log_prob_better[torch.arange(batch_size * num_steps), targets_better].view(batch_size, num_steps).mean(dim=-1)
    gathered_worse = log_prob_worse[torch.arange(batch_size * num_steps), targets_worse].view(batch_size, num_steps).mean(dim=-1)

    makespan_factor = makespan_worse / makespan_better
    return -torch.log(torch.sigmoid(makespan_factor * (gathered_better - gathered_worse))).mean()


class FlexibleJobShopStates:
    size = 5

    def __init__(self, device: str = "cpu", eps: float = 1e-5):
        self._eps = eps
        self._q = np.array([0.25, 0.5, 0.75])
        self.dev = device

    def init_state(self, ins: dict, batch_size: int = 1):
        self.data = ins["data"]
        self.num_j, self.num_m, self.num_o = ins["j"], ins["m"], ins["o"]
        self._factor = np.max(self.data)
        self.bs = batch_size

        self.job_ct = np.zeros((batch_size, self.num_j), dtype=np.float32)
        self.mac_ct = np.zeros((batch_size, self.num_m), dtype=np.float32)

        self.num_assign = np.sum(self.data >= 0, axis=2)
        self.max_num_ops = np.max(self.num_assign)
        self.action_dim = self.num_j * self.max_num_ops
        self.action_2_job = np.zeros((self.action_dim,), dtype=np.int32)
        self.action_2_op = np.zeros((self.action_dim,), dtype=np.int32)
        offset = 0
        for job_idx in range(self.num_j):
            self.action_2_job[offset : offset + self.max_num_ops] = job_idx
            self.action_2_op[offset : offset + self.max_num_ops] = np.arange(self.max_num_ops)
            offset += self.max_num_ops

        self.ops_No = -np.ones((self.num_j, self.num_o, self.max_num_ops), dtype=np.int32)
        self.job_idx = np.zeros((batch_size, self.num_j), dtype=np.int32)
        self.machines = np.zeros((self.num_j, self.num_o, self.max_num_ops), dtype=np.int32)
        self.costs = -np.ones((self.num_j, self.num_o, self.max_num_ops), dtype=np.float32)

        op_no = 0
        for job_idx in range(self.num_j):
            for op_idx in range(self.num_o):
                count = 0
                for machine_idx in range(self.num_m):
                    if self.data[job_idx, op_idx, machine_idx] >= 0:
                        self.machines[job_idx, op_idx, count] = machine_idx
                        self.costs[job_idx, op_idx, count] = self.data[job_idx, op_idx, machine_idx]
                        self.ops_No[job_idx, op_idx, count] = op_no
                        op_no += 1
                        count += 1
        self.costs /= self._factor

        job_state = torch.zeros((batch_size, self.action_dim, self.size), dtype=torch.float32, device=self.dev)
        machine_state = torch.zeros((batch_size, self.action_dim, self.size), dtype=torch.float32, device=self.dev)
        return (job_state, machine_state), self.mask

    @property
    def mask(self) -> torch.Tensor:
        row_idx = np.arange(self.num_j).reshape(1, -1).repeat(self.bs, axis=0).flatten()
        job_idx = self.job_idx.flatten()
        local_job_idx = job_idx % self.num_o
        num = np.where(job_idx < self.num_o, self.num_assign[row_idx, local_job_idx], 0).reshape(self.bs, -1, 1)
        idx = np.arange(self.max_num_ops).reshape(1, 1, -1).repeat(self.bs, axis=0).repeat(self.num_j, axis=1)
        mask = np.where(idx < num, 1, 0).reshape(self.bs, -1)
        return torch.tensor(mask, dtype=torch.float32, device=self.dev)

    @property
    def done(self) -> bool:
        return bool(torch.all(self.mask == 0))

    @property
    def ops(self) -> torch.Tensor:
        idx = np.arange(self.num_j).reshape(1, -1).repeat(self.bs, axis=0)
        ops = self.ops_No[idx, self.job_idx % self.num_o].reshape(self.bs, -1)
        return torch.tensor(ops, dtype=torch.long, device=self.dev)

    @property
    def makespan(self) -> np.ndarray:
        return self.mac_ct.max(-1) * self._factor

    def _schedule(self, action: np.ndarray) -> None:
        batch_idx = np.arange(self.bs)
        jobs = self.action_2_job[action]
        ops = self.action_2_op[action]
        job_idx = self.job_idx[batch_idx, jobs]
        macs = self.machines[jobs, job_idx % self.num_o, ops]
        proc_times = self.costs[jobs, job_idx % self.num_o, ops]

        mac_ct = self.mac_ct[batch_idx, macs]
        job_ct = self.job_ct[batch_idx, jobs]
        ct = np.where(mac_ct > job_ct, mac_ct, job_ct) + proc_times
        self.mac_ct[batch_idx, macs] = ct
        self.job_ct[batch_idx, jobs] = ct
        self.job_idx[batch_idx, jobs] += 1

    def update(self, action: torch.Tensor):
        self._schedule(action.cpu().numpy())
        batch_idx = np.arange(self.bs).reshape(-1, 1).repeat(self.action_dim, axis=1).flatten()
        jobs = self.action_2_job.reshape(1, -1).repeat(self.bs, axis=0).flatten()
        ops = self.action_2_op.reshape(1, -1).repeat(self.bs, axis=0).flatten()

        job_ct = self.job_ct[batch_idx, jobs].reshape(self.bs, -1)
        current_makespan = job_ct.max(axis=-1, keepdims=True) + self._eps
        job_idx = self.job_idx[batch_idx, jobs]
        macs = self.machines[jobs, job_idx % self.num_o, ops]
        mac_ct = self.mac_ct[batch_idx, macs].reshape(self.bs, -1)

        job_state = -np.ones((self.bs, self.action_dim, self.size), dtype=np.float32)
        machine_state = -np.ones((self.bs, self.action_dim, self.size), dtype=np.float32)

        q_job = np.quantile(self.job_ct, self._q, -1).T
        job_state[..., 0] = job_ct / current_makespan
        job_state[..., 1] = job_ct - self.job_ct.mean(-1, keepdims=True)
        job_state[..., 2:5] = np.expand_dims(job_ct, 2) - np.expand_dims(q_job, 1)

        q_machine = np.quantile(self.mac_ct, self._q, -1).T
        machine_state[..., 0] = mac_ct / current_makespan
        machine_state[..., 1] = mac_ct - self.mac_ct.mean(-1, keepdims=True)
        machine_state[..., 2:5] = np.expand_dims(mac_ct, 2) - np.expand_dims(q_machine, 1)

        return (
            torch.tensor(job_state, dtype=torch.float32, device=self.dev),
            torch.tensor(machine_state, dtype=torch.float32, device=self.dev),
        ), self.mask


def solve_problem(ins, batch_size, device, encoder, decoder, use_greedy=False):
    num_jobs, num_ops = ins["j"], ins["o"]
    total_ops = num_jobs * num_ops
    fjsp = FlexibleJobShopStates(device)
    state, mask = fjsp.init_state(ins, batch_size)

    trajs = -torch.ones((batch_size, total_ops), dtype=torch.long, device=device)
    logits_store = -torch.ones((batch_size, total_ops, fjsp.action_dim), dtype=torch.float32, device=device)

    embed = encoder(
        ins["x"].to(device),
        ops_egdes=ins["ops_edges"].to(device),
        job_edges=ins["job_edges"].to(device),
        mac_edges=ins["mac_edges"].to(device),
    )
    zeros = torch.zeros((batch_size, 1, encoder.out_size), dtype=torch.float32, device=device)
    last_ops = h = c = None

    for step_idx in range(total_ops):
        ops = fjsp.ops
        if last_ops is None:
            logits, (h, c) = decoder(embed[ops], state, zeros, h, c)
        else:
            logits, (h, c) = decoder(embed[ops], state, embed[last_ops], h, c)
        logits = logits + mask.log()
        policies = torch.distributions.Categorical(logits=logits)
        actions = policies.sample()
        if use_greedy:
            actions[0] = logits[0].argmax()
        trajs[:, step_idx] = actions
        logits_store[:, step_idx] = logits
        last_ops = fjsp.ops.gather(1, actions.unsqueeze(-1))
        state, mask = fjsp.update(actions)

    return trajs, logits_store, fjsp.makespan, fjsp


def sample_training_pair(
    ins: dict,
    encoder,
    decoder,
    B: int = 32,
    K: int = 16,
    use_greedy: bool = True,
    pair_mode: str = "anchor_best",
    device: str = "cpu",
):
    encoder.train()
    decoder.train()

    num_jobs, num_ops = ins["j"], ins["o"]
    total_ops = num_jobs * num_ops
    trajs, logits_store, makespan, fjsp = solve_problem(
        ins, B, device, encoder, decoder, use_greedy=use_greedy
    )

    if B % K != 0:
        raise ValueError(f"BOPO FJSP requires B % K == 0, got B={B}, K={K}.")
    all_idx = sorted(range(B), key=lambda idx: makespan[idx])
    selected = all_idx[:: B // K]
    pair_mode_norm = str(pair_mode or "anchor_best").strip().lower()
    if pair_mode_norm not in {"anchor_best", "all_pairs"}:
        raise ValueError(f"Unsupported BOPO FJSP pair_mode={pair_mode!r}; use 'anchor_best' or 'all_pairs'.")

    selected_pairs: list[tuple[int, int]] = []
    if pair_mode_norm == "anchor_best":
        selected_pairs = [(selected[0], worse_idx) for worse_idx in selected[1:]]
    else:
        for better_pos in range(len(selected)):
            for worse_pos in range(better_pos + 1, len(selected)):
                selected_pairs.append((selected[better_pos], selected[worse_pos]))
    num_pairs = len(selected_pairs)

    trajs_better = -torch.ones((num_pairs, total_ops), dtype=torch.long, device=device)
    logits_better = -torch.ones((num_pairs, total_ops, fjsp.action_dim), dtype=torch.float32, device=device)
    ms_better = torch.ones((num_pairs,), dtype=torch.float32, device=device)
    trajs_worse = -torch.ones((num_pairs, total_ops), dtype=torch.long, device=device)
    logits_worse = -torch.ones((num_pairs, total_ops, fjsp.action_dim), dtype=torch.float32, device=device)
    ms_worse = torch.ones((num_pairs,), dtype=torch.float32, device=device)

    makespan_tensor = torch.tensor(makespan, dtype=torch.float32, device=device)
    for pair_idx, (better_idx, worse_idx) in enumerate(selected_pairs):
        trajs_better[pair_idx] = trajs[better_idx]
        logits_better[pair_idx] = logits_store[better_idx]
        ms_better[pair_idx] = makespan_tensor[better_idx]
        trajs_worse[pair_idx] = trajs[worse_idx]
        logits_worse[pair_idx] = logits_store[worse_idx]
        ms_worse[pair_idx] = makespan_tensor[worse_idx]

    return (
        Solutions(trajs=trajs_better, logits=logits_better, mss=ms_better),
        Solutions(trajs=trajs_worse, logits=logits_worse, mss=ms_worse),
        makespan,
    )


@torch.no_grad()
def sampling(ins: dict, encoder, decoder, bs: int = 32, use_greedy: bool = True, device: str = "cpu"):
    encoder.eval()
    decoder.eval()
    _, _, makespan, _ = solve_problem(ins, bs, device, encoder, decoder, use_greedy=use_greedy)
    return makespan
