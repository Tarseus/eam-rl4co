"""
Routing environments and generators.

This package is used by both training code (with optional dependencies installed) and
stand-alone scripts (where optional dependencies may be missing). To keep imports robust,
we guard all re-exports and only expose symbols that import successfully.
"""


def _try_import(module: str, names: list[str]) -> None:
    try:  # pragma: no cover
        mod = __import__(module, fromlist=names)
        for name in names:
            globals()[name] = getattr(mod, name)
    except Exception:
        return


# TSP
_try_import("rl4co.envs.routing.tsp.env", ["DenseRewardTSPEnv", "TSPEnv", "TSPkoptEnv"])
_try_import("rl4co.envs.routing.tsp.generator", ["TSPGenerator"])

# ATSP
_try_import("rl4co.envs.routing.atsp.env", ["ATSPEnv"])
_try_import("rl4co.envs.routing.atsp.generator", ["ATSPGenerator"])

# CVRP family
_try_import("rl4co.envs.routing.cvrp.env", ["CVRPEnv"])
_try_import("rl4co.envs.routing.cvrp.generator", ["CVRPGenerator"])
_try_import("rl4co.envs.routing.cvrptw.env", ["CVRPTWEnv"])
_try_import("rl4co.envs.routing.cvrptw.generator", ["CVRPTWGenerator"])
_try_import("rl4co.envs.routing.cvrpmvc.env", ["CVRPMVCEnv"])

# OP / (S)PCTSP
_try_import("rl4co.envs.routing.op.env", ["OPEnv"])
_try_import("rl4co.envs.routing.op.generator", ["OPGenerator"])
_try_import("rl4co.envs.routing.pctsp.env", ["PCTSPEnv"])
_try_import("rl4co.envs.routing.pctsp.generator", ["PCTSPGenerator"])
_try_import("rl4co.envs.routing.spctsp.env", ["SPCTSPEnv"])

# PDP variants
_try_import("rl4co.envs.routing.pdp.env", ["PDPEnv", "PDPRuinRepairEnv"])
_try_import("rl4co.envs.routing.pdp.generator", ["PDPGenerator"])

# Other routing
_try_import("rl4co.envs.routing.sdvrp.env", ["SDVRPEnv"])
_try_import("rl4co.envs.routing.shpp.env", ["SHPPEnv"])
_try_import("rl4co.envs.routing.shpp.generator", ["SHPPGenerator"])
_try_import("rl4co.envs.routing.mdcpdp.env", ["MDCPDPEnv"])
_try_import("rl4co.envs.routing.mdcpdp.generator", ["MDCPDPGenerator"])
_try_import("rl4co.envs.routing.mtsp.env", ["MTSPEnv"])
_try_import("rl4co.envs.routing.mtsp.generator", ["MTSPGenerator"])
_try_import("rl4co.envs.routing.mtvrp.env", ["MTVRPEnv"])
_try_import("rl4co.envs.routing.mtvrp.generator", ["MTVRPGenerator"])
_try_import("rl4co.envs.routing.svrp.env", ["SVRPEnv"])
_try_import("rl4co.envs.routing.svrp.generator", ["SVRPGenerator"])
