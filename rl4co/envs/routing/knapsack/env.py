from __future__ import annotations

from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from .generator import KnapsackGenerator

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None

log = get_pylogger(__name__)


class KnapsackEnv(RL4COEnvBase):
    """0-1 Knapsack Problem environment.
    At each step, the agent chooses an item to put into the knapsack. The episode
    ends when the agent selects action 0 ("finish"). The reward is the sum of the
    values of the selected items.
    Observations:
        - weight and value of each item
        - current used capacity of the knapsack
        - visited items
    Constraints:
        - each item can be selected at most once
        - the total weight of selected items cannot exceed the capacity
    Reward:
        - sum of the values of selected items
    """

    name = "knapsack"

    def __init__(
        self,
        generator: KnapsackGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if generator is None:
            generator = KnapsackGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]
        if action.dim() == 1:
            current_item = action.unsqueeze(-1)
        elif action.dim() == 2 and action.size(-1) == 1:
            current_item = action
        else:
            raise ValueError(f"Invalid action shape: {tuple(action.shape)}")
        prev_done = td.get("done", torch.zeros_like(td["i"], dtype=torch.bool))
        if prev_done.dim() == 1:
            prev_done = prev_done.unsqueeze(-1)
        # No-op once done to prevent trajectories from "un-finishing" in batched decoding
        current_item = torch.where(
            prev_done, torch.zeros_like(current_item), current_item
        )
        n_items = td["demand"].size(-1)
        selected_weight = gather_by_index(
            td["demand"], torch.clamp(current_item - 1, 0, n_items - 1), squeeze=False
        )
        selected_value = gather_by_index(
            td["values"], torch.clamp(current_item - 1, 0, n_items - 1), squeeze=False
        )
        used_capacity = (
            td["used_capacity"] + selected_weight * (current_item != 0).float()
        )
        total_value = td["total_value"] + selected_value * (current_item != 0).float()
        assert current_item.min() >= 0, f"current_item.min()={current_item.min()} < 0"
        assert current_item.max() < td["visited"].size(
            -1
        ), f"current_item.max()={current_item.max()} >= visited.size(-1)"
        visited = td["visited"].scatter(-1, current_item, 1)
        done = prev_done | ((current_item == 0) & (td["i"] > 0))
        reward = torch.zeros_like(done)
        td.update(
            {
                "current_node": current_item,
                "used_capacity": used_capacity,
                "total_value": total_value,
                "visited": visited,
                "i": td["i"] + 1,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device
        td_reset = TensorDict(
            {
                "weights": td["weights"],
                "demand": td["demand"],
                "values": td["values"],
                "locs": td["locs"],
                "vehicle_capacity": td["vehicle_capacity"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "total_value": torch.zeros((*batch_size, 1), device=device),
                "visited": torch.zeros(
                    (*batch_size, td["demand"].shape[-1] + 1),
                    dtype=torch.bool,
                    device=device,
                ),
                "i": torch.zeros((*batch_size, 1), dtype=torch.int64, device=device),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"] + 1e-6
        mask = td["visited"][..., 1:] | exceeds_cap
        # Only allow the finish action (0) when no feasible item remains.
        item_mask = ~mask  # [B, n_items], True means feasible item
        finish_allowed = ~item_mask.any(-1, keepdim=True)  # [B, 1]
        action_mask = torch.cat((finish_allowed, item_mask), -1)
        done = td.get("done", None)
        if done is not None:
            if done.dim() == 1:
                done = done.unsqueeze(-1)
            if done.any():
                only_finish = torch.zeros_like(action_mask, dtype=torch.bool)
                only_finish[..., 0] = True
                action_mask = torch.where(
                    done.expand_as(action_mask), only_finish, action_mask
                )
        return action_mask

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            (torch.zeros_like(td["values"][..., :1]), td["values"]), dim=-1
        )
        collected = values.gather(1, actions)
        if actions.dim() == 2:
            # Only count items before the first "finish" (0) action
            ended = (actions == 0).cumsum(dim=1) > 0
            collected = collected.masked_fill(ended, 0)
        return collected.sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        # Only consider actions up to (and excluding) the first finish action (0)
        if actions.dim() != 2:
            raise ValueError(
                f"Expected actions with shape [batch, seq_len], got {tuple(actions.shape)}"
            )
        actions = actions.clone()
        ended = (actions == 0).cumsum(dim=1) > 0
        actions[ended] = 0
        sorted_actions = actions.data.sort(1)[0]
        assert (
            (sorted_actions[:, 1:] == 0)
            | (sorted_actions[:, 1:] > sorted_actions[:, :-1])
        ).all(), "Duplicates"
        weights = torch.cat(
            (torch.zeros_like(td["demand"][..., :1]), td["demand"]), dim=-1
        )
        total_weight = weights.gather(1, actions).sum(-1)
        assert (
            total_weight <= td["vehicle_capacity"].squeeze(-1) + 1e-5
        ).all(), "Capacity exceeded"
        # if not (total_weight <= td["vehicle_capacity"].squeeze(-1) + 1e-5).all():
        #     torch.set_printoptions(profile="full")
        #     mask = total_weight > 12.5
        #     print(f"Total weight > 12.5: {total_weight[mask]}")
        #     # print("Actions:", actions)
        #     print(f"Total weight.shape:", total_weight.shape)
        #     print(f"td['vehicle_capacity'].shape: {td['vehicle_capacity'].squeeze(-1).shape}")
        #     print("Actions.shape:", actions.shape)
        #     exit()

    def _make_spec(self, generator: KnapsackGenerator):
        self.observation_spec = Composite(
            locs=Bounded(
                low=0.0,
                high=1.0,
                shape=(generator.num_items + 1, 2),
                dtype=torch.float32,
            ),
            weights=Bounded(
                low=generator.min_weight,
                high=generator.max_weight,
                shape=(generator.num_items,),
                dtype=torch.float32,
            ),
            demand=Bounded(
                low=generator.min_weight,
                high=generator.max_weight,
                shape=(generator.num_items,),
                dtype=torch.float32,
            ),
            values=Bounded(
                low=generator.min_value,
                high=generator.max_value,
                shape=(generator.num_items,),
                dtype=torch.float32,
            ),
            capacity=Unbounded(shape=(1,), dtype=torch.float32),
            current_item=Unbounded(shape=(1,), dtype=torch.int64),
            used_capacity=Unbounded(shape=(1,), dtype=torch.float32),
            total_value=Unbounded(shape=(1,), dtype=torch.float32),
            visited=Unbounded(shape=(generator.num_items + 1,), dtype=torch.bool),
            i=Unbounded(shape=(1,), dtype=torch.int64),
            action_mask=Unbounded(shape=(generator.num_items + 1,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_items + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def get_optimal_solutions(
        self, td: TensorDict, max_instances: int | None = None
    ) -> float:
        """Get average optimal solution value for the knapsack problem over a batch.
        Note: This solves a MILP per instance with CBC and can be slow for large batches.
        """
        if pulp is None:
            raise RuntimeError(
                "pulp is not installed; cannot compute optimal knapsack solutions. "
                "Install pulp to enable this baseline."
            )
        if max_instances is not None:
            td = td[:max_instances]
        weights = td["demand"].cpu().numpy()
        values = td["values"].cpu().numpy()
        capacities = td["vehicle_capacity"].cpu().numpy()
        batch_size = weights.shape[0]
        objective_values: list[float] = []
        for b in range(batch_size):
            prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)
            x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(weights.shape[1])]
            prob += pulp.lpSum(values[b, i] * x[i] for i in range(len(x))), "Objective"
            prob += (
                pulp.lpSum(weights[b, i] * x[i] for i in range(len(x))) <= capacities[b],
                "CapacityConstraint",
            )
            solver = pulp.PULP_CBC_CMD(msg=False)
            prob.solve(solver)
            objective_values.append(float(pulp.value(prob.objective)))
        return float(sum(objective_values) / batch_size) if batch_size > 0 else 0.0

    def get_greedy_solutions(
        self, td: TensorDict, max_instances: int | None = None
    ) -> float:
        """Get average greedy solution value for the knapsack problem over a batch."""
        if max_instances is not None:
            td = td[:max_instances]
        weights = td["demand"].cpu().numpy()
        values = td["values"].cpu().numpy()
        capacities = td["vehicle_capacity"].cpu().numpy()
        batch_size = weights.shape[0]
        objective_values: list[float] = []
        for b in range(batch_size):
            # Avoid divide-by-zero when min_weight is 0.
            ratio = values[b] / (weights[b] + 1e-12)
            idx = ratio.argsort()[::-1]
            total_weight = 0.0
            total_value = 0.0
            cap = float(capacities[b])
            for i in idx:
                w = float(weights[b, i])
                if total_weight + w <= cap:
                    total_weight += w
                    total_value += float(values[b, i])
            objective_values.append(total_value)
        return float(sum(objective_values) / batch_size) if batch_size > 0 else 0.0
