# Base environment
from rl4co.envs.common.base import RL4COEnvBase

# NOTE: Many environments depend on optional third-party packages. Importing them eagerly
# makes `import rl4co.envs` fail in minimal setups (e.g., when only routing envs are used).
# We therefore guard these imports and only register environments that successfully import.

ENV_REGISTRY: dict[str, type[RL4COEnvBase]] = {}


def _try_register(env_name: str, module: str, cls_name: str) -> None:
    try:  # pragma: no cover
        mod = __import__(module, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        ENV_REGISTRY[env_name] = cls
        globals()[cls_name] = cls  # keep old import style working when available
    except Exception:
        return


# Routing
_try_register("atsp", "rl4co.envs.routing.atsp.env", "ATSPEnv")
_try_register("cvrp", "rl4co.envs.routing.cvrp.env", "CVRPEnv")
_try_register("cvrptw", "rl4co.envs.routing.cvrptw.env", "CVRPTWEnv")
_try_register("cvrpmvc", "rl4co.envs.routing.cvrpmvc.env", "CVRPMVCEnv")
_try_register("op", "rl4co.envs.routing.op.env", "OPEnv")
_try_register("pctsp", "rl4co.envs.routing.pctsp.env", "PCTSPEnv")
_try_register("spctsp", "rl4co.envs.routing.spctsp.env", "SPCTSPEnv")
_try_register("pdp", "rl4co.envs.routing.pdp.env", "PDPEnv")
_try_register("pdp_ruin_repair", "rl4co.envs.routing.pdp.env", "PDPRuinRepairEnv")
_try_register("sdvrp", "rl4co.envs.routing.sdvrp.env", "SDVRPEnv")
_try_register("shpp", "rl4co.envs.routing.shpp.env", "SHPPEnv")
_try_register("svrp", "rl4co.envs.routing.svrp.env", "SVRPEnv")
_try_register("tsp", "rl4co.envs.routing.tsp.env", "TSPEnv")
_try_register("tsp_kopt", "rl4co.envs.routing.tsp.env", "TSPkoptEnv")
_try_register("mdcpdp", "rl4co.envs.routing.mdcpdp.env", "MDCPDPEnv")
_try_register("mtsp", "rl4co.envs.routing.mtsp.env", "MTSPEnv")
_try_register("mtvrp", "rl4co.envs.routing.mtvrp.env", "MTVRPEnv")

# EDA (optional deps e.g., robust_downloader)
_try_register("dpp", "rl4co.envs.eda.dpp.env", "DPPEnv")
_try_register("mdpp", "rl4co.envs.eda.mdpp.env", "MDPPEnv")

# Graph
_try_register("mcp", "rl4co.envs.graph.mcp.env", "MCPEnv")
_try_register("flp", "rl4co.envs.graph.flp.env", "FLPEnv")

# Scheduling (Lightning/extra deps may be required for training, but envs should still import)
_try_register("ffsp", "rl4co.envs.scheduling.ffsp.env", "FFSPEnv")
_try_register("jssp", "rl4co.envs.scheduling.jssp.env", "JSSPEnv")
_try_register("fjsp", "rl4co.envs.scheduling.fjsp.env", "FJSPEnv")
_try_register("smtwtp", "rl4co.envs.scheduling.smtwtp.env", "SMTWTPEnv")


def get_env(env_name: str, *args, **kwargs) -> RL4COEnvBase:
    """Get environment by name.

    Args:
        env_name: Environment name
        *args: Positional arguments for environment
        **kwargs: Keyword arguments for environment

    Returns:
        Environment
    """
    env_cls = ENV_REGISTRY.get(env_name, None)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment {env_name}. Available environments: {ENV_REGISTRY.keys()}"
        )
    return env_cls(*args, **kwargs)
