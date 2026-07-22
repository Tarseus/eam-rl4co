"""
Model zoo namespace.

The full RL4CO model zoo pulls a large optional dependency set. To keep lightweight
use-cases (e.g., GA/EA search) importable in minimal environments, we avoid eager
imports here. Import the exact model you need from its submodule.
"""

try:  # pragma: no cover
    from rl4co.models.zoo.earl.model import EAM, SymEAM
except Exception:  # pragma: no cover
    EAM = None
    SymEAM = None

