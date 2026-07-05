"""Centralized configuration for Beetle.

Single source of truth for:
  - the project root path,
  - the parsed ``params.yaml`` (parsed once, cached),
  - the compute device (``mps`` on Apple Silicon, ``cuda`` where available,
    else ``cpu``), computed once.

Historically these three concerns were duplicated across many modules, each
re-parsing ``params.yaml`` and each recomputing the device with a slightly
different block. This module collapses all of that into one importable
singleton, ``CONFIG``.

Importing this module does *not* import ``torch`` — the device is resolved
lazily on first call to :meth:`Config.device`, so ``config`` stays importable
in lightweight environments (tests, tooling) that have no ML stack installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Resolves project paths, parses ``params.yaml`` once, selects the device once."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        # ``src/config.py`` -> parents[1] is the repository root.
        self.PROJECT_ROOT: Path = (
            Path(project_root) if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self._params: Optional[Dict[str, Any]] = None
        self._device: Optional[str] = None

    # -- params -----------------------------------------------------------------

    @property
    def params(self) -> Dict[str, Any]:
        """The parsed ``params.yaml``, parsed exactly once and cached thereafter."""
        if self._params is None:
            params_path = self.PROJECT_ROOT / "params.yaml"
            with open(params_path, "r", encoding="utf-8") as f:
                self._params = yaml.safe_load(f) or {}
        return self._params

    # -- device -----------------------------------------------------------------

    def device(self) -> str:
        """Return the compute device, computed once and cached.

        Prefers CUDA when present, then Apple-Silicon ``mps``, falling back to
        ``cpu`` when neither is available or ``torch`` is not installed.
        """
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # -- paths ------------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        """Join ``parts`` onto the project root (e.g. ``CONFIG.path("data", "clean")``)."""
        return self.PROJECT_ROOT.joinpath(*parts)


# Module-level singleton. Import this everywhere instead of re-deriving config.
CONFIG = Config()
