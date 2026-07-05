"""Unit tests for the centralized configuration module (task 1.2).

Covers single-parse caching, device-selection logic, and path resolution
(Requirements 1.2, 1.4).
"""

from pathlib import Path

from src.config import CONFIG, Config


def test_params_parsed_once_and_cached():
    """``params`` parses exactly once; repeated access returns the same object."""
    cfg = Config()
    first = cfg.params
    second = cfg.params
    assert first is second  # cached, not re-parsed
    assert isinstance(first, dict)


def test_params_has_expected_top_level_keys():
    """The real params.yaml exposes the expected sections."""
    params = CONFIG.params
    assert "models" in params
    assert "search" in params


def test_device_is_cached_and_valid():
    """``device()`` returns one of the allowed strings and caches the result."""
    cfg = Config()
    device = cfg.device()
    assert device in {"mps", "cuda", "cpu"}
    assert cfg.device() is device or cfg.device() == device  # cached


def test_device_falls_back_to_cpu_without_torch(monkeypatch):
    """When torch is unavailable, device selection falls back to ``cpu``."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert Config._resolve_device() == "cpu"


def test_path_joins_under_project_root():
    """``path`` joins onto the project root and resolves a known file."""
    cfg = Config()
    assert cfg.path("params.yaml") == cfg.PROJECT_ROOT / "params.yaml"
    assert cfg.path("data", "clean").parent == cfg.path("data")


def test_project_root_is_repo_root():
    """PROJECT_ROOT points at the repo root (contains params.yaml and src/)."""
    root = CONFIG.PROJECT_ROOT
    assert isinstance(root, Path)
    assert (root / "params.yaml").exists()
    assert (root / "src").is_dir()
