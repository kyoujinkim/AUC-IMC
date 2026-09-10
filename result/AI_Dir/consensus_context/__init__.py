"""Utilities for building consensus-model context datasets."""

from typing import TYPE_CHECKING

from .catalog import SnapshotFile, discover_snapshots, sha256_file

if TYPE_CHECKING:
    from .cli import build_context

__all__ = ["SnapshotFile", "build_context", "discover_snapshots", "sha256_file"]


def __getattr__(name: str):
    """Load the CLI-backed builder lazily so ``python -m`` stays warning-free."""

    if name == "build_context":
        from .cli import build_context

        return build_context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
