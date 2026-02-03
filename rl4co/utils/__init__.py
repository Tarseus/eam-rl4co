"""
Keep `rl4co.utils` importable in minimal environments.

Some RL4CO utilities depend on optional stacks (Hydra, Lightning, TorchMetrics, Transformers).
Importing them unconditionally makes unrelated functionality (e.g., envs / decoding / heuristics)
unusable when those optional deps are missing or incompatible.
"""

from rl4co.utils.pylogger import get_pylogger


def _optional_import_error(name: str, extra: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        f"{name} is unavailable because optional dependencies are missing/incompatible. "
        f"Install {extra} to enable it."
    )


# Optional: hydra-based instantiation helpers
try:  # pragma: no cover
    from rl4co.utils.instantiators import instantiate_callbacks, instantiate_loggers
except Exception:  # pragma: no cover
    def instantiate_callbacks(*args, **kwargs):
        raise _optional_import_error("instantiate_callbacks", "`hydra-core` (+ project extras)")

    def instantiate_loggers(*args, **kwargs):
        raise _optional_import_error("instantiate_loggers", "`hydra-core` (+ project extras)")


# Optional: rich + hydra helpers
try:  # pragma: no cover
    from rl4co.utils.rich_utils import enforce_tags, print_config_tree
except Exception:  # pragma: no cover
    def enforce_tags(*args, **kwargs):
        raise _optional_import_error("enforce_tags", "`hydra-core` and `rich`")

    def print_config_tree(*args, **kwargs):
        raise _optional_import_error("print_config_tree", "`hydra-core` and `rich`")


# Optional: trainer and experiment utilities (Lightning stack)
try:  # pragma: no cover
    from rl4co.utils.trainer import RL4COTrainer
except Exception:  # pragma: no cover
    def RL4COTrainer(*args, **kwargs):  # type: ignore[misc]
        raise _optional_import_error("RL4COTrainer", "`lightning` (+ project extras)")


try:  # pragma: no cover
    from rl4co.utils.utils import (
        extras,
        get_metric_value,
        log_hyperparameters,
        show_versions,
        task_wrapper,
    )
except Exception:  # pragma: no cover
    def extras(*args, **kwargs):
        raise _optional_import_error("extras", "`hydra-core` and `lightning` (+ project extras)")

    def get_metric_value(*args, **kwargs):
        raise _optional_import_error("get_metric_value", "`hydra-core` and `lightning` (+ project extras)")

    def log_hyperparameters(*args, **kwargs):
        raise _optional_import_error("log_hyperparameters", "`hydra-core` and `lightning` (+ project extras)")

    def show_versions(*args, **kwargs):
        raise _optional_import_error("show_versions", "`hydra-core` and `lightning` (+ project extras)")

    def task_wrapper(*args, **kwargs):
        raise _optional_import_error("task_wrapper", "`hydra-core` and `lightning` (+ project extras)")
