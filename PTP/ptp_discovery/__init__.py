from .problem import (
    PTPDiscoveryProblem,
    PTPDiscoveryCandidate,
    PTPDiscoveryResult,
)
from .search import PTPDiscoverySearch
from .pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

__all__ = [
    "PTPDiscoveryProblem",
    "PTPDiscoveryCandidate",
    "PTPDiscoveryResult",
    "PTPDiscoverySearch",
    "PreferenceBuilderIR",
    "PreferenceBuilderImplementationHint",
    "CompiledPreferenceBuilder",
    "PreferenceBuilderCompileError",
    "compile_preference_builder",
    "validate_pref_batch",
]


# Lazy exports to avoid import cycles between `ptp_discovery` and `fitness`.
def __getattr__(name: str):  # noqa: ANN201
    if name in {
        "CompiledPreferenceBuilder",
        "PreferenceBuilderCompileError",
        "compile_preference_builder",
        "validate_pref_batch",
    }:
        from . import pref_builder_compiler as _pbc

        return getattr(_pbc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

