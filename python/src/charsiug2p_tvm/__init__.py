from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

DIST_NAME = "charsiug2p-tvm"

try:
    __version__ = version(DIST_NAME)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["DIST_NAME", "__version__"]
