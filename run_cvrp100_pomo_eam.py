import argparse
import os
import time

import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.loggers import CSVLogger
from tensordict.tensordict import TensorDict

from rl4co.envs.routing import (
    CVRPEnv,
    CVRPGenerator,
    KnapsackEnv,
    KnapsackGenerator,
    TSPEnv,
    TSPGenerator,
)
from rl4co.models import AttentionModel, AttentionModelPolicy, EAM, POMO
from rl4co.utils import RL4COTrainer


def _parse_milestones(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in parts] if parts else None


def _default_lr_milestones(epochs: int) -> list[int]:
    if epochs <= 1:
        return []
    first = max(1, int(epochs * 0.8))
    second = max(first + 1, int(epochs * 0.95))
    second = min(second, max(1, epochs - 1))
    if first >= epochs:
        return []
    if second <= first or second >= epochs:
        return [first]
    return [first, second]


def _normalize_problem(name: str) -> str:
    key = name.lower().strip()
    if key == "knapsack":
        key = "kp"
    return key


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise SystemExit(f"Invalid boolean value: {value!r}. Use true/false.")


class CorrelatedKnapsackGenerator(KnapsackGenerator):
    """Knapsack generator with mixed correlated values (harder but similar scale)."""

    def __init__(self, *args, value_noise: float = 0.05, corr: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_noise = value_noise
        if not 0.0 <= corr <= 1.0:
            raise ValueError(f"corr must be in [0, 1], got {corr}")
        self.corr = corr

    def _generate(self, batch_size) -> TensorDict:
        weights = self.weight_sampler.sample((*batch_size, self.num_items))
        base_values = self.value_sampler.sample((*batch_size, self.num_items))
        noise = torch.empty_like(weights).uniform_(-self.value_noise, self.value_noise)
        values = (self.corr * weights + (1.0 - self.corr) * base_values + noise).clamp(
            self.min_value, self.max_value
        )

        items = torch.stack((weights, values), dim=-1)
        depot = torch.zeros(*batch_size, 1, 2, device=items.device, dtype=items.dtype)
        locs = torch.cat((depot, items), dim=-2)
        capacity = torch.full((*batch_size, 1), self.capacity)

        return TensorDict(
            {
                "weights": weights,
                "demand": weights,
                "values": values,
                "locs": locs,
                "vehicle_capacity": capacity,
            },
            batch_size=batch_size,
        )


class RatioCapacityKnapsackGenerator(KnapsackGenerator):
    """Knapsack generator with per-instance capacity = ratio * sum(weights)."""

    def __init__(self, *args, capacity_ratio: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity_ratio = capacity_ratio

    def _generate(self, batch_size) -> TensorDict:
        td = super()._generate(batch_size)
        weights = td["weights"]
        td["vehicle_capacity"] = weights.sum(-1, keepdim=True) * self.capacity_ratio
        return td


class RatioCapacityCorrelatedKnapsackGenerator(CorrelatedKnapsackGenerator):
    """Correlated knapsack with per-instance capacity = ratio * sum(weights)."""

    def __init__(self, *args, capacity_ratio: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity_ratio = capacity_ratio

    def _generate(self, batch_size) -> TensorDict:
        td = super()._generate(batch_size)
        weights = td["weights"]
        td["vehicle_capacity"] = weights.sum(-1, keepdim=True) * self.capacity_ratio
        return td


class SubsetSumKnapsackGenerator(KnapsackGenerator):
    """Subset-sum style KP: values == weights with discrete weights.

    We keep (weight, value) features in [0, 1] to match the knapsack env's `locs` spec.
    """

    def __init__(
        self,
        *,
        num_items: int,
        w_max: int = 100,
        capacity_ratio: float = 0.5,
    ) -> None:
        super().__init__(
            num_items=num_items,
            min_weight=0.0,
            max_weight=1.0,
            min_value=0.0,
            max_value=1.0,
            weight_distribution="uniform",
            value_distribution="uniform",
            capacity=None,
        )
        self.w_max = int(w_max)
        self.capacity_ratio = float(capacity_ratio)

    def _generate(self, batch_size) -> TensorDict:
        weights_int = torch.randint(
            1, self.w_max + 1, (*batch_size, self.num_items), dtype=torch.int64
        )
        weights = weights_int.float() / float(self.w_max)
        values = weights.clone()

        items = torch.stack((weights, values), dim=-1)
        depot = torch.zeros(*batch_size, 1, 2, device=items.device, dtype=items.dtype)
        locs = torch.cat((depot, items), dim=-2)
        capacity = weights.sum(-1, keepdim=True) * self.capacity_ratio

        return TensorDict(
            {
                "weights": weights,
                "demand": weights,
                "values": values,
                "locs": locs,
                "vehicle_capacity": capacity,
            },
            batch_size=batch_size,
        )


class StronglyCorrelatedKnapsackGenerator(KnapsackGenerator):
    """Strongly-correlated integer-ish KP: values ~= weights + discrete noise."""

    def __init__(
        self,
        *,
        num_items: int,
        w_max: int = 100,
        v_noise_max: int = 10,
        capacity_ratio: float = 0.5,
    ) -> None:
        super().__init__(
            num_items=num_items,
            min_weight=0.0,
            max_weight=1.0,
            min_value=0.0,
            max_value=1.0,
            weight_distribution="uniform",
            value_distribution="uniform",
            capacity=None,
        )
        self.w_max = int(w_max)
        self.v_noise_max = int(v_noise_max)
        self.capacity_ratio = float(capacity_ratio)

    def _generate(self, batch_size) -> TensorDict:
        weights_int = torch.randint(
            1, self.w_max + 1, (*batch_size, self.num_items), dtype=torch.int64
        )
        noise_int = torch.randint(
            0, self.v_noise_max + 1, (*batch_size, self.num_items), dtype=torch.int64
        )
        denom = float(self.w_max + self.v_noise_max)
        weights = weights_int.float() / denom
        values = (weights_int + noise_int).float() / denom

        items = torch.stack((weights, values), dim=-1)
        depot = torch.zeros(*batch_size, 1, 2, device=items.device, dtype=items.dtype)
        locs = torch.cat((depot, items), dim=-2)
        capacity = weights.sum(-1, keepdim=True) * self.capacity_ratio

        return TensorDict(
            {
                "weights": weights,
                "demand": weights,
                "values": values,
                "locs": locs,
                "vehicle_capacity": capacity,
            },
            batch_size=batch_size,
        )


def _require_eam():
    if EAM is not None:
        return EAM
    try:
        from rl4co.models.zoo.earl.model import EAM as EAM_cls
    except Exception as exc:
        raise SystemExit(
            "EAM is unavailable. Install optional dependencies (e.g., numba) "
            "and ensure rl4co.models.zoo.earl is importable. "
            f"Original error: {exc}"
        ) from exc
    return EAM_cls


class KnapsackBaselineEvalCallback(Callback):
    """Log greedy/optimal baselines on a small KP batch once per val epoch.

    This is meant as a sanity check for remaining headroom; solving optimal is expensive.
    """

    def __init__(
        self,
        *,
        greedy_eval_size: int = 32,
        optimal_eval_size: int = 8,
        optimal_every: int = 1,
    ) -> None:
        super().__init__()
        self.greedy_eval_size = int(greedy_eval_size)
        self.optimal_eval_size = int(optimal_eval_size)
        # Allow disabling optimal by passing 0.
        self.optimal_every = int(optimal_every)
        self._did_log = False

    def on_validation_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        self._did_log = False

    def on_validation_batch_end(  # type: ignore[override]
        self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self._did_log or batch_idx != 0:
            return
        if not getattr(trainer, "is_global_zero", True):
            return
        env = getattr(pl_module, "env", None)
        if env is None or getattr(env, "name", None) != "knapsack":
            return

        td = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not isinstance(td, TensorDict):
            return

        # Compute on CPU and on a small subset to keep validation fast.
        max_needed = max(self.greedy_eval_size, self.optimal_eval_size, 0)
        if max_needed > 0 and td.shape[0] > max_needed:
            td = td[:max_needed]
        td = td.to("cpu")

        try:
            greedy_mean = (
                float(env.get_greedy_solutions(td, max_instances=self.greedy_eval_size))
                if self.greedy_eval_size > 0
                else float("nan")
            )
        except Exception:
            greedy_mean = float("nan")
        pl_module.log(
            "val/greedy_mean",
            greedy_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if (
            self.optimal_every > 0
            and self.optimal_eval_size > 0
            and (trainer.current_epoch % self.optimal_every) == 0
        ):
            try:
                optimal_mean = float(
                    env.get_optimal_solutions(td, max_instances=self.optimal_eval_size)
                )
            except Exception:
                optimal_mean = float("nan")
            pl_module.log(
                "val/optimal_mean",
                optimal_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self._did_log = True


class LogKpNumStartsCallback(Callback):
    """Log effective num_starts once at the beginning of training (KP only)."""

    def __init__(self) -> None:
        super().__init__()
        self._did_log = False

    def on_train_batch_start(  # type: ignore[override]
        self, trainer, pl_module, batch, batch_idx: int
    ) -> None:
        if self._did_log or batch_idx != 0:
            return
        if not getattr(trainer, "is_global_zero", True):
            return
        env = getattr(pl_module, "env", None)
        if env is None or getattr(env, "name", None) != "knapsack":
            return

        batch_td = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not isinstance(batch_td, TensorDict):
            return

        td0 = batch_td[:1] if batch_td.shape[0] > 1 else batch_td
        td_reset = env.reset(td0)
        n_start_env = int(env.get_num_starts(td_reset))
        n_start_model = getattr(pl_module, "num_starts", None)
        n_start_eff = int(n_start_model) if n_start_model is not None else n_start_env

        pl_module.log(
            "train/num_starts_env",
            float(n_start_env),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "train/num_starts_effective",
            float(n_start_eff),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self._did_log = True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train POMO or EAM-POMO on CVRP/TSP/KP (any size)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pomo",
        help="Model to train: pomo, eam-pomo, or am",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="cvrp",
        help="Problem type: cvrp, tsp, kp (or knapsack).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=int, default=0, help="Physical GPU id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--problem-size",
        type=int,
        default=100,
        help="Problem size: #customers for CVRP, #nodes for TSP, #items for KP.",
    )
    parser.add_argument(
        "--capacity",
        type=float,
        default=None,
        help="Override capacity for CVRP/KP (defaults to generator table/heuristic).",
    )
    parser.add_argument(
        "--kp-capacity-ratio",
        type=float,
        default=None,
        help="If set, capacity = ratio * sum(weights) per instance (overrides --capacity).",
    )
    parser.add_argument("--num-augment", type=int, default=8)
    parser.add_argument(
        "--num-starts",
        type=int,
        default=None,
        help="Number of multistart rollouts. For KP, recommend 5 or 10.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-data-size", type=int, default=160_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--test-data-size", type=int, default=10_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument(
        "--lr-milestones",
        type=str,
        default=None,
        help="Comma-separated epochs for MultiStepLR (default: 80% and 95% of --epochs).",
    )
    parser.add_argument(
        "--kp-correlated",
        type=str,
        default="true",
        help="Use correlated KP values (true/false). Default: true.",
    )
    parser.add_argument(
        "--kp-kind",
        type=str,
        default="uniform",
        choices=["uniform", "subset-sum", "strongly-correlated"],
        help="KP only: instance generator kind.",
    )
    parser.add_argument(
        "--kp-w-max",
        type=int,
        default=100,
        help="KP only (subset-sum/strongly-correlated): max discrete weight.",
    )
    parser.add_argument(
        "--kp-v-noise-max",
        type=int,
        default=10,
        help="KP only (strongly-correlated): max additive noise to values.",
    )
    parser.add_argument(
        "--kp-baseline-eval-size",
        type=int,
        default=32,
        help="KP only: number of val instances to compute greedy baseline on each epoch.",
    )
    parser.add_argument(
        "--kp-optimal-eval-size",
        type=int,
        default=8,
        help="KP only: number of val instances to compute optimal baseline on (0 to disable).",
    )
    parser.add_argument(
        "--kp-optimal-eval-every",
        type=int,
        default=1,
        help="KP only: compute optimal baseline every N validation epochs (0 to disable).",
    )
    parser.add_argument("--kp-min-weight", type=float, default=0.4)
    parser.add_argument("--kp-max-weight", type=float, default=0.6)
    parser.add_argument("--kp-min-value", type=float, default=0.4)
    parser.add_argument("--kp-max-value", type=float, default=0.6)
    parser.add_argument(
        "--kp-value-noise",
        type=float,
        default=0.05,
        help="Noise added to correlated KP values (default: 0.05).",
    )
    parser.add_argument(
        "--kp-corr",
        type=float,
        default=0.7,
        help="Mixing factor between weights and independent values (0..1).",
    )
    args = parser.parse_args()

    model_key = args.model.lower().replace("_", "-")
    if model_key not in {"pomo", "eam-pomo", "am"}:
        raise SystemExit("Unknown model. Use --model pomo, eam-pomo, or am.")

    os.environ.setdefault("WANDB_MODE", "offline")
    L.seed_everything(args.seed, workers=True)

    problem_key = _normalize_problem(args.problem)
    if problem_key not in {"cvrp", "tsp", "kp"}:
        raise SystemExit("Unknown problem. Use --problem cvrp, tsp, or kp.")
    if problem_key == "kp" and args.num_starts is None:
        args.num_starts = 5

    check_solution = not (problem_key == "kp" and model_key == "pomo")
    kp_correlated = _parse_bool(args.kp_correlated, default=True)
    if problem_key == "kp" and args.num_augment != 1:
        # KP 'locs' are (weight, value) features, not Euclidean coordinates, so dihedral augmentation is invalid.
        args.num_augment = 1
    if problem_key == "cvrp":
        env = CVRPEnv(
            CVRPGenerator(
                num_loc=args.problem_size,
                loc_distribution="uniform",
                num_depots=1,
                capacity=args.capacity,
            )
        )
    elif problem_key == "tsp":
        env = TSPEnv(
            TSPGenerator(
                num_loc=args.problem_size,
                loc_distribution="uniform",
            )
        )
    else:
        if args.kp_kind != "uniform":
            cap_ratio = (
                args.kp_capacity_ratio if args.kp_capacity_ratio is not None else 0.5
            )
            if args.kp_kind == "subset-sum":
                kp_gen_cls = SubsetSumKnapsackGenerator
                kp_gen_kwargs = {
                    "num_items": args.problem_size,
                    "w_max": args.kp_w_max,
                    "capacity_ratio": cap_ratio,
                }
            else:
                kp_gen_cls = StronglyCorrelatedKnapsackGenerator
                kp_gen_kwargs = {
                    "num_items": args.problem_size,
                    "w_max": args.kp_w_max,
                    "v_noise_max": args.kp_v_noise_max,
                    "capacity_ratio": cap_ratio,
                }
        else:
            kp_gen_kwargs = {
                "num_items": args.problem_size,
                "min_weight": args.kp_min_weight,
                "max_weight": args.kp_max_weight,
                "min_value": args.kp_min_value,
                "max_value": args.kp_max_value,
                "weight_distribution": "uniform",
                "value_distribution": "uniform",
                "capacity": args.capacity,
            }
            if kp_correlated:
                kp_gen_kwargs["value_noise"] = args.kp_value_noise
                kp_gen_kwargs["corr"] = args.kp_corr
            if args.kp_capacity_ratio is not None:
                kp_gen_kwargs["capacity"] = None
                kp_gen_kwargs["capacity_ratio"] = args.kp_capacity_ratio
                kp_gen_cls = (
                    RatioCapacityCorrelatedKnapsackGenerator
                    if kp_correlated
                    else RatioCapacityKnapsackGenerator
                )
            else:
                kp_gen_cls = (
                    CorrelatedKnapsackGenerator if kp_correlated else KnapsackGenerator
                )
        env = KnapsackEnv(kp_gen_cls(**kp_gen_kwargs), check_solution=check_solution)
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=6,
        num_heads=8,
        normalization="instance",
        use_graph_context=False,
    )

    metrics = {
        "train": [
            "loss",
            "reward",
            "max_reward",
            "delta_nll",
            "diversity_edge",
            "diversity_edge_entropy",
            "diversity_edge_simpson",
            "diversity_route",
            "diversity_edit_edge",
            "ga_cost_gain",
            "ga_cost_gain_rel",
            "t_decode",
            "t_ga",
            "t_diag",
        ],
        "val": [
            "reward",
            "reward_no_ls",
            "max_reward",
            "max_reward_no_ls",
            "max_aug_reward",
            "max_aug_reward_no_ls",
        ],
        "test": ["reward", "max_reward", "max_aug_reward"],
    }
    if problem_key == "kp":
        # With augmentation disabled for KP, augmented metrics are meaningless/misleading.
        metrics["val"] = [m for m in metrics["val"] if "aug" not in m]
        metrics["test"] = [m for m in metrics["test"] if "aug" not in m]

    lr_milestones = _parse_milestones(args.lr_milestones)
    if lr_milestones is None:
        lr_milestones = _default_lr_milestones(args.epochs)

    if model_key == "pomo":
        model = POMO(
            env,
            policy,
            num_starts=args.num_starts,
            reward_scale="norm",
            batch_size=args.batch_size,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": lr_milestones, "gamma": 0.1},
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_data_size=args.test_data_size,
            num_augment=args.num_augment,
            metrics=metrics,
        )
        model_tag = "pomo"
    elif model_key == "am":
        model = AttentionModel(
            env,
            policy,
            baseline="rollout",
            reward_scale="norm",
            batch_size=args.batch_size,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": lr_milestones, "gamma": 0.1},
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_data_size=args.test_data_size,
            metrics=metrics,
        )
        model_tag = "am"
    else:
        ea_kwargs = {
            "num_generations": 3,
            "mutation_rate": 0.1,
            "crossover_rate": 0.6,
            "selection_rate": 0.2,
            "batch_size": args.batch_size,
            "ea_batch_size": args.batch_size,
            "alpha": 0.5,
            "beta": 3,
            "ea_prob": 0.01,
            "ea_epoch": 700,
            "improve_mode": "ga",
            "val_improve": False,
            "val_improve_mode": "ga",
            "val_num_generations": 0,
        }
        eam_cls = _require_eam()
        model = eam_cls(
            env,
            policy,
            baseline="shared",
            num_starts=args.num_starts,
            reward_scale="norm",
            batch_size=args.batch_size,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler="MultiStepLR",
            lr_scheduler_kwargs={"milestones": lr_milestones, "gamma": 0.1},
            train_data_size=args.train_data_size,
            val_data_size=args.val_data_size,
            test_data_size=args.test_data_size,
            num_augment=args.num_augment,
            metrics=metrics,
            ea_kwargs=ea_kwargs,
        )
        model_tag = "eam_pomo"

    run_name = args.run_name or f"{model_tag}_{problem_key}{args.problem_size}"
    version = args.version or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}_gpu{args.device}"
    )
    logger = CSVLogger(save_dir=args.log_dir, name=run_name, version=version)

    callbacks: list[Callback] = []
    if problem_key == "kp":
        callbacks.append(
            KnapsackBaselineEvalCallback(
                greedy_eval_size=args.kp_baseline_eval_size,
                optimal_eval_size=args.kp_optimal_eval_size,
                optimal_every=args.kp_optimal_eval_every,
            )
        )
        callbacks.append(LogKpNumStartsCallback())
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[args.device],
        precision=32,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
