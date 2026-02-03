import logging

try:  # pragma: no cover
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except Exception:  # pragma: no cover
    # Keep core RL4CO utilities usable in minimal environments (e.g., without Lightning installed
    # or with a partially incompatible dependency stack). In such cases we simply log on every process.
    def rank_zero_only(fn):  # type: ignore[misc]
        return fn


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
