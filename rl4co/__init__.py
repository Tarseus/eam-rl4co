from importlib.metadata import PackageNotFoundError, version as get_version

try:
    # The package version is obtained from the installed distribution metadata
    __version__ = get_version(__package__)
except PackageNotFoundError:  # pragma: no cover
    # Fallback for source checkouts without `pip install -e .`
    __version__ = "0.0.0"
