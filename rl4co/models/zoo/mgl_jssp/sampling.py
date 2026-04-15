from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass
class Solutions:
    mss: torch.Tensor
    logits: torch.Tensor
    trajs: torch.Tensor


def trajectory_log_probs(logits: torch.Tensor, trajs: torch.Tensor) -> torch.Tensor:
    batch_size, num_steps, action_dim = logits.shape
    flat_logits = logits.view(-1, action_dim)
    flat_trajs = trajs.view(-1)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    chosen = log_probs[
        torch.arange(batch_size * num_steps, device=logits.device), flat_trajs
    ]
    return chosen.view(batch_size, num_steps).sum(dim=-1)


def solution_ratio(makespans: torch.Tensor) -> float:
    best = makespans.min().clamp_min(1e-8)
    worst = makespans.max().clamp_min(1e-8)
    return float((worst / best).item())


def sro_loss(info_better: Solutions, info_worse: Solutions) -> tuple[torch.Tensor, float]:
    logits_better = info_better.logits
    trajs_better = info_better.trajs
    makespan_better = info_better.mss
    logits_worse = info_worse.logits
    trajs_worse = info_worse.trajs
    makespan_worse = info_worse.mss

    batch_size, num_steps, action_dim = logits_better.shape
    better_flat = F.log_softmax(logits_better.view(-1, action_dim), dim=-1)
    worse_flat = F.log_softmax(logits_worse.view(-1, action_dim), dim=-1)
    better_gathered = better_flat[
        torch.arange(batch_size * num_steps, device=logits_better.device),
        trajs_better.view(-1),
    ].view(batch_size, num_steps).mean(dim=-1)
    worse_gathered = worse_flat[
        torch.arange(batch_size * num_steps, device=logits_worse.device),
        trajs_worse.view(-1),
    ].view(batch_size, num_steps).mean(dim=-1)
    makespan_factor = makespan_worse / makespan_better.clamp_min(1e-8)
    loss = -torch.log(torch.sigmoid(makespan_factor * (better_gathered - worse_gathered))).mean()
    return loss, float(makespan_factor.max().item())


def rl_loss(samples: Solutions) -> tuple[torch.Tensor, float]:
    log_probs = trajectory_log_probs(samples.logits, samples.trajs)
    rewards = -samples.mss
    advantage = rewards - rewards.mean()
    advantage = advantage / (advantage.std(unbiased=False) + 1e-8)
    loss = -(advantage.detach() * log_probs).mean()
    return loss, solution_ratio(samples.mss)


def po_loss(samples: Solutions, impl: str = "bt") -> tuple[torch.Tensor, float]:
    if impl not in {"bt", "exponential"}:
        raise ValueError(f"Unknown po_loss impl: {impl}")

    log_probs = trajectory_log_probs(samples.logits, samples.trajs)
    makespans = samples.mss
    pair_mask = torch.triu(
        torch.ones(
            (makespans.shape[0], makespans.shape[0]),
            dtype=torch.bool,
            device=makespans.device,
        ),
        diagonal=1,
    )
    if not pair_mask.any():
        return log_probs.sum() * 0.0, solution_ratio(makespans)

    left_idx, right_idx = pair_mask.nonzero(as_tuple=True)
    left_ms = makespans[left_idx]
    right_ms = makespans[right_idx]
    unequal = left_ms != right_ms
    if not unequal.any():
        return log_probs.sum() * 0.0, solution_ratio(makespans)

    left_idx = left_idx[unequal]
    right_idx = right_idx[unequal]
    left_ms = left_ms[unequal]
    right_ms = right_ms[unequal]

    better_is_left = left_ms < right_ms
    better_idx = torch.where(better_is_left, left_idx, right_idx)
    worse_idx = torch.where(better_is_left, right_idx, left_idx)
    score_diff = log_probs[better_idx] - log_probs[worse_idx]
    if impl == "bt":
        loss = -F.logsigmoid(score_diff).mean()
    else:
        loss = -score_diff.mean()
    return loss, solution_ratio(makespans)


class JobShopStates:
    size = 11

    def __init__(self, device: str = "cpu", eps: float = 1e-5):
        self.dev = device
        self._eps = eps
        self._q = torch.tensor([0.25, 0.5, 0.75], device=device)

    def init_state(self, instances: list[dict], batch_size_per_instance: int = 1):
        # Phase 4: Support multiple instances, all same-shape
        num_instances = len(instances)
        self.num_instances = num_instances
        self.B_per_instance = batch_size_per_instance

        # Verify all instances have the same shape
        first_shape = (instances[0]["j"], instances[0]["m"])
        for i, ins in enumerate(instances):
            shape = (ins["j"], ins["m"])
            if shape != first_shape:
                raise ValueError(f"All instances must have same shape: instance 0 is {first_shape[0]}x{first_shape[1]}, instance {i} is {shape[0]}x{shape[1]}")

        self.num_j, self.num_m = first_shape

        # Concatenate machines and costs for all instances
        # Each instance has shape (10, 10), view to (-1,)
        self.machines_list = [ins["machines"].view(-1).to(self.dev) for ins in instances]
        self._factor_list = [ins["costs"].max() for ins in instances]
        self.costs_list = [ins["costs"].view(-1).to(self.dev) / factor.clamp_min(1.0)
                            for ins, factor in zip(instances, self._factor_list)]

        # Total batch size = num_instances * batch_size_per_instance
        total_bs = num_instances * batch_size_per_instance
        self.total_bs = total_bs
        self.batch_idx = torch.arange(total_bs, device=self.dev)
        self.instance_idx = torch.arange(num_instances, device=self.dev).repeat_interleave(batch_size_per_instance)

        # Job start indices for each instance
        self.job_start = torch.arange(0, self.num_j * self.num_m, self.num_m, device=self.dev)

        # State tensors with shape (total_bs, num_j, ...)
        self.job_ptr = torch.zeros((total_bs, self.num_j), dtype=torch.int32, device=self.dev)
        self.job_ct = torch.zeros((total_bs, self.num_j), dtype=torch.float32, device=self.dev)
        self.mac_ct = torch.zeros((total_bs, self.num_m), dtype=torch.float32, device=self.dev)
        states = torch.zeros(
            (total_bs, self.num_j, self.size), dtype=torch.float32, device=self.dev
        )
        return states, self.mask.to(torch.float32)

    @property
    def mask(self) -> torch.Tensor:
        return self.job_ptr < self.num_m

    @property
    def ops(self) -> torch.Tensor:
        return self.job_start + (self.job_ptr % self.num_m)

    @property
    def makespan(self) -> torch.Tensor:
        # Returns (total_bs,) tensor; reshape to (num_instances, B_per_instance) if needed
        factor_expanded = torch.tensor(self._factor_list, device=self.dev)[self.instance_idx]
        return self.mac_ct.max(-1)[0] * factor_expanded

    def _schedule(self, jobs: torch.Tensor) -> None:
        ops = self.ops[self.batch_idx, jobs]

        # Gather machines and costs based on instance_idx
        machines = torch.empty_like(ops)
        costs = torch.empty_like(ops, dtype=torch.float32)
        for i in range(self.num_instances):
            mask = self.instance_idx == i
            if mask.any():
                machines[mask] = self.machines_list[i][ops[mask]]
                costs[mask] = self.costs_list[i][ops[mask]]

        completion = torch.maximum(
            self.mac_ct[self.batch_idx, machines], self.job_ct[self.batch_idx, jobs]
        )
        completion = completion + costs
        self.mac_ct[self.batch_idx, machines] = completion
        self.job_ct[self.batch_idx, jobs] = completion
        self.job_ptr[self.batch_idx, jobs] += 1

    def update(self, jobs: torch.Tensor):
        self._schedule(jobs)

        # Gather machines based on instance_idx
        machines = torch.empty_like(self.ops)
        for i in range(self.num_instances):
            mask = self.instance_idx == i
            if mask.any():
                machines[mask] = self.machines_list[i][self.ops[mask]]

        mac_ct = self.mac_ct.gather(1, machines)
        current_makespan = self.job_ct.max(-1, keepdim=True)[0] + self._eps
        next_states = -torch.ones((self.total_bs, self.num_j, self.size), device=self.dev)
        next_states[..., 0] = self.job_ct - mac_ct
        q_job = torch.quantile(self.job_ct, self._q, -1).T
        next_states[..., 1:4] = self.job_ct.unsqueeze(-1) - q_job.unsqueeze(1)
        next_states[..., 4] = self.job_ct - self.job_ct.mean(-1, keepdim=True)
        next_states[..., 5] = self.job_ct / current_makespan
        q_machine = torch.quantile(self.mac_ct, self._q, -1).T
        next_states[..., 6:9] = mac_ct.unsqueeze(-1) - q_machine.unsqueeze(1)
        next_states[..., 9] = mac_ct - self.mac_ct.mean(-1, keepdim=True)
        next_states[..., 10] = mac_ct / current_makespan
        return next_states, self.mask.to(torch.float32)

    def __call__(self, jobs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        self._schedule(jobs)

        # Gather machines based on instance_idx
        machines = torch.empty_like(self.ops)
        for i in range(self.num_instances):
            mask = self.instance_idx == i
            if mask.any():
                machines[mask] = self.machines_list[i][self.ops[mask]]

        mac_ct = self.mac_ct.gather(1, machines)
        current_makespan = self.job_ct.max(-1, keepdim=True)[0] + self._eps
        states[..., 0] = self.job_ct - mac_ct
        q_job = torch.quantile(self.job_ct, self._q, -1).T
        states[..., 1:4] = self.job_ct.unsqueeze(-1) - q_job.unsqueeze(1)
        states[..., 4] = self.job_ct - self.job_ct.mean(-1, keepdim=True)
        states[..., 5] = self.job_ct / current_makespan
        q_machine = torch.quantile(self.mac_ct, self._q, -1).T
        states[..., 6:9] = mac_ct.unsqueeze(-1) - q_machine.unsqueeze(1)
        states[..., 9] = mac_ct - self.mac_ct.mean(-1, keepdim=True)
        states[..., 10] = mac_ct / current_makespan
        return self.mask.to(torch.float32)


def solve_jsp(
    instances: list[dict],
    batch_size_per_instance: int,
    device: str,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    use_greedy: bool = False,
):
    # Phase 4: solve for multiple instances, all same-shape
    # instances: list of instance dicts, all same shape
    # batch_size_per_instance: B, rollouts per instance
    num_instances = len(instances)
    total_batch_size = num_instances * batch_size_per_instance

    # Verify all instances have the same shape
    first_shape = (instances[0]["j"], instances[0]["m"])
    for i, ins in enumerate(instances):
        shape = (ins["j"], ins["m"])
        if shape != first_shape:
            raise ValueError(f"All instances must have same shape: instance 0 is {first_shape[0]}x{first_shape[1]}, instance {i} is {shape[0]}x{shape[1]}")

    num_jobs, num_machines = first_shape
    num_steps = num_jobs * num_machines - 1

    # Output tensors with shape (total_batch_size, ...) = (N*B, ...)
    trajs = -torch.ones((total_batch_size, num_steps), dtype=torch.long, device=device)
    logits_store = -torch.ones(
        (total_batch_size, num_steps, num_jobs), dtype=torch.float32, device=device
    )
    entropies = torch.zeros((total_batch_size, num_steps), dtype=torch.float32, device=device)

    # Encode all instances first
    embeds_list = []
    for ins in instances:
        embed = encoder(
            ins["x"].to(device),
            job_edges=ins["job_edges"].to(device),
            mac_edges=ins["mac_edges"].to(device),
        )
        embeds_list.append(embed)

    # Initialize states
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(instances, batch_size_per_instance)

    # Expand embeds to (total_batch_size, num_ops, encoder_out_size)
    # Then gather ops for each rollout
    last_ops = h = c = None
    zeros = torch.zeros((total_batch_size, 1, encoder.out_size), dtype=torch.float32, device=device)

    for step_idx in range(num_steps):
        ops = jsp.ops  # (total_batch_size, num_jobs)

        # Gather embeds for each instance's ops
        # embed[ops] for each instance
        embed_ops = torch.empty((total_batch_size, num_jobs, encoder.out_size), dtype=torch.float32, device=device)
        for i in range(num_instances):
            instance_mask = jsp.instance_idx == i
            if instance_mask.any():
                embed_ops[instance_mask] = embeds_list[i][ops[instance_mask]]

        if last_ops is None:
            logits, (h, c) = decoder(embed_ops, state, zeros, h, c)
        else:
            # Gather last_ops embeddings
            last_embed = torch.empty((total_batch_size, 1, encoder.out_size), dtype=torch.float32, device=device)
            for i in range(num_instances):
                instance_mask = jsp.instance_idx == i
                if instance_mask.any():
                    last_embed[instance_mask] = embeds_list[i][last_ops[instance_mask]]
            logits, (h, c) = decoder(embed_ops, state, last_embed, h, c)

        logits = logits + mask.log()
        dist = torch.distributions.Categorical(logits=logits)
        jobs = dist.sample()

        if use_greedy:
            # First rollout of each instance is greedy
            for i in range(num_instances):
                first_idx = i * batch_size_per_instance
                jobs[first_idx] = logits[first_idx].argmax()

        trajs[:, step_idx] = jobs
        logits_store[:, step_idx] = logits
        entropies[:, step_idx] = dist.entropy()
        last_ops = jsp.ops.gather(1, jobs.unsqueeze(-1))
        state, mask = jsp.update(jobs)

    jsp(mask.float().argmax(-1), state)
    return trajs, logits_store, jsp.makespan, entropies


def sample_training_pair(
    instances: list[dict] | dict,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    B: int = 128,
    K: int = 16,
    use_greedy: bool = False,
    pair_mode: str = "anchor_best",
    device: str = "cpu",
) -> tuple[Solutions, Solutions, torch.Tensor, int]:
    """Phase 4: Sample training pairs for multiple instances.
    Pairs are strictly constructed within each instance.
    Returns: (better_solutions, worse_solutions, best_makespan, total_num_pairs)
    """
    encoder.train()
    decoder.train()

    if isinstance(instances, dict):
        instances = [instances]

    num_instances = len(instances)

    # Verify all instances have the same shape
    first_shape = (instances[0]["j"], instances[0]["m"])
    for i, ins in enumerate(instances):
        shape = (ins["j"], ins["m"])
        if shape != first_shape:
            raise ValueError(f"All instances must have same shape: instance 0 is {first_shape[0]}x{first_shape[1]}, instance {i} is {shape[0]}x{shape[1]}")

    # Solve for all instances at once
    trajs, logits_store, makespans, _ = solve_jsp(
        instances,
        batch_size_per_instance=B,
        device=device,
        encoder=encoder,
        decoder=decoder,
        use_greedy=use_greedy,
    )

    if B % K != 0:
        raise ValueError(f"MGL JSSP requires B % K == 0, got B={B}, K={K}.")

    # Reshape to (num_instances, B, ...)
    num_steps = trajs.size(1)
    num_jobs = first_shape[0]
    trajs_reshaped = trajs.view(num_instances, B, num_steps)
    logits_reshaped = logits_store.view(num_instances, B, num_steps, num_jobs)
    makespans_reshaped = makespans.view(num_instances, B)

    pair_mode = str(pair_mode or "anchor_best").strip().lower()
    if pair_mode not in {"anchor_best", "all_pairs"}:
        raise ValueError(f"Unsupported pair_mode={pair_mode!r}")

    # Collect pairs from each instance (strictly within-instance)
    all_better_trajs = []
    all_better_logits = []
    all_better_mss = []
    all_worse_trajs = []
    all_worse_logits = []
    all_worse_mss = []
    best_makespan_list = []

    for i in range(num_instances):
        # Get this instance's candidates
        trajs_i = trajs_reshaped[i]
        logits_i = logits_reshaped[i]
        makespans_i = makespans_reshaped[i]

        # Sort and select within this instance
        sorted_idx = sorted(range(B), key=lambda idx: makespans_i[idx].item())
        selected = sorted_idx[:: B // K]

        # Select pairs within this instance
        selected_pairs: list[tuple[int, int]] = []
        if pair_mode == "anchor_best":
            selected_pairs = [(selected[0], worse_idx) for worse_idx in selected[1:]]
        else:
            for better_idx in range(len(selected)):
                for worse_idx in range(better_idx + 1, len(selected)):
                    selected_pairs.append((selected[better_idx], selected[worse_idx]))

        # Collect pairs for this instance
        for better_idx, worse_idx in selected_pairs:
            all_better_trajs.append(trajs_i[better_idx])
            all_better_logits.append(logits_i[better_idx])
            all_better_mss.append(makespans_i[better_idx])
            all_worse_trajs.append(trajs_i[worse_idx])
            all_worse_logits.append(logits_i[worse_idx])
            all_worse_mss.append(makespans_i[worse_idx])

        best_makespan_list.append(makespans_i.min())

    # Stack all pairs together
    total_num_pairs = len(all_better_trajs)
    better_sols = Solutions(
        mss=torch.stack(all_better_mss),
        logits=torch.stack(all_better_logits),
        trajs=torch.stack(all_better_trajs),
    )
    worse_sols = Solutions(
        mss=torch.stack(all_worse_mss),
        logits=torch.stack(all_worse_logits),
        trajs=torch.stack(all_worse_trajs),
    )
    best_makespan = torch.stack(best_makespan_list).min()

    return better_sols, worse_sols, best_makespan, total_num_pairs


@torch.no_grad()
def sampling(
    instances: list[dict] | dict,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    bs: int = 128,
    use_greedy: bool = False,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Phase 2: support list of instances (but eval uses single instance)
    encoder.eval()
    decoder.eval()

    if isinstance(instances, dict):
        instances = [instances]

    # For eval, we expect single instance
    trajs, logits_store, makespans, entropies = solve_jsp(
        instances,
        batch_size_per_instance=bs,
        device=device,
        encoder=encoder,
        decoder=decoder,
        use_greedy=use_greedy,
    )
    log_probs = trajectory_log_probs(logits_store, trajs)
    return makespans, entropies, log_probs
