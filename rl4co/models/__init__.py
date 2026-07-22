"""
RL4CO models package.

Historically, this module eagerly imported the whole model zoo. That makes light-weight
use-cases (e.g., environments + heuristic / EA baselines) fragile because optional
dependencies (Lightning, Hydra, TorchMetrics, Transformers, etc.) get imported even
when they are not needed.

This file now lazily resolves symbols on demand via `__getattr__`, preserving the
previous public API when dependencies are available.
"""

from __future__ import annotations

import importlib


_EXPORTS: dict[str, tuple[str, str]] = {
    # Common constructive interfaces
    "AutoregressiveDecoder": ("rl4co.models.common.constructive.autoregressive", "AutoregressiveDecoder"),
    "AutoregressiveEncoder": ("rl4co.models.common.constructive.autoregressive", "AutoregressiveEncoder"),
    "AutoregressivePolicy": ("rl4co.models.common.constructive.autoregressive", "AutoregressivePolicy"),
    "ConstructiveDecoder": ("rl4co.models.common.constructive.base", "ConstructiveDecoder"),
    "ConstructiveEncoder": ("rl4co.models.common.constructive.base", "ConstructiveEncoder"),
    "ConstructivePolicy": ("rl4co.models.common.constructive.base", "ConstructivePolicy"),
    "NonAutoregressiveDecoder": ("rl4co.models.common.constructive.nonautoregressive", "NonAutoregressiveDecoder"),
    "NonAutoregressiveEncoder": ("rl4co.models.common.constructive.nonautoregressive", "NonAutoregressiveEncoder"),
    "NonAutoregressivePolicy": ("rl4co.models.common.constructive.nonautoregressive", "NonAutoregressivePolicy"),
    "TransductiveModel": ("rl4co.models.common.transductive", "TransductiveModel"),
    # RL
    "StepwisePPO": ("rl4co.models.rl", "StepwisePPO"),
    "A2C": ("rl4co.models.rl.a2c.a2c", "A2C"),
    "RL4COLitModule": ("rl4co.models.rl.common.base", "RL4COLitModule"),
    "PPO": ("rl4co.models.rl.ppo.ppo", "PPO"),
    "REINFORCEBaseline": ("rl4co.models.rl.reinforce.baselines", "REINFORCEBaseline"),
    "get_reinforce_baseline": ("rl4co.models.rl.reinforce.baselines", "get_reinforce_baseline"),
    "REINFORCE": ("rl4co.models.rl.reinforce.reinforce", "REINFORCE"),
    # Zoo
    "ActiveSearch": ("rl4co.models.zoo.active_search", "ActiveSearch"),
    "AttentionModel": ("rl4co.models.zoo.am", "AttentionModel"),
    "AttentionModelPolicy": ("rl4co.models.zoo.am", "AttentionModelPolicy"),
    "AMPPO": ("rl4co.models.zoo.amppo", "AMPPO"),
    "DACT": ("rl4co.models.zoo.dact", "DACT"),
    "DACTPolicy": ("rl4co.models.zoo.dact", "DACTPolicy"),
    "DeepACO": ("rl4co.models.zoo.deepaco", "DeepACO"),
    "DeepACOPolicy": ("rl4co.models.zoo.deepaco", "DeepACOPolicy"),
    "EAS": ("rl4co.models.zoo.eas", "EAS"),
    "EASEmb": ("rl4co.models.zoo.eas", "EASEmb"),
    "EASLay": ("rl4co.models.zoo.eas", "EASLay"),
    "GLOP": ("rl4co.models.zoo.glop", "GLOP"),
    "GLOPPolicy": ("rl4co.models.zoo.glop", "GLOPPolicy"),
    "GFACS": ("rl4co.models.zoo.gfacs", "GFACS"),
    "GFACSPolicy": ("rl4co.models.zoo.gfacs", "GFACSPolicy"),
    "HeterogeneousAttentionModel": ("rl4co.models.zoo.ham", "HeterogeneousAttentionModel"),
    "HeterogeneousAttentionModelPolicy": ("rl4co.models.zoo.ham", "HeterogeneousAttentionModelPolicy"),
    "L2DAttnPolicy": ("rl4co.models.zoo.l2d", "L2DAttnPolicy"),
    "L2DModel": ("rl4co.models.zoo.l2d", "L2DModel"),
    "L2DPolicy": ("rl4co.models.zoo.l2d", "L2DPolicy"),
    "L2DPolicy4PPO": ("rl4co.models.zoo.l2d", "L2DPolicy4PPO"),
    "L2DPPOModel": ("rl4co.models.zoo.l2d", "L2DPPOModel"),
    "MatNet": ("rl4co.models.zoo.matnet", "MatNet"),
    "MatNetPolicy": ("rl4co.models.zoo.matnet", "MatNetPolicy"),
    "MDAM": ("rl4co.models.zoo.mdam", "MDAM"),
    "MDAMPolicy": ("rl4co.models.zoo.mdam", "MDAMPolicy"),
    "MVMoE_AM": ("rl4co.models.zoo.mvmoe", "MVMoE_AM"),
    "MVMoE_POMO": ("rl4co.models.zoo.mvmoe", "MVMoE_POMO"),
    "N2S": ("rl4co.models.zoo.n2s", "N2S"),
    "N2SPolicy": ("rl4co.models.zoo.n2s", "N2SPolicy"),
    "NARGNNPolicy": ("rl4co.models.zoo.nargnn", "NARGNNPolicy"),
    "NeuOpt": ("rl4co.models.zoo.neuopt", "NeuOpt"),
    "NeuOptPolicy": ("rl4co.models.zoo.neuopt", "NeuOptPolicy"),
    "PolyNet": ("rl4co.models.zoo.polynet", "PolyNet"),
    "POMO": ("rl4co.models.zoo.pomo", "POMO"),
    "PointerNetwork": ("rl4co.models.zoo.ptrnet", "PointerNetwork"),
    "PointerNetworkPolicy": ("rl4co.models.zoo.ptrnet", "PointerNetworkPolicy"),
    "SymNCO": ("rl4co.models.zoo.symnco", "SymNCO"),
    "SymNCOPolicy": ("rl4co.models.zoo.symnco", "SymNCOPolicy"),
    # EAM / EA (optional: numba)
    "EAM": ("rl4co.models.zoo.earl.model", "EAM"),
    "SymEAM": ("rl4co.models.zoo.earl.model", "SymEAM"),
    "EA": ("rl4co.models.zoo.earl.evolution", "EA"),
}


def __getattr__(name: str):
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = spec
    obj = getattr(importlib.import_module(module), attr)
    globals()[name] = obj
    return obj


__all__ = sorted(_EXPORTS.keys())
