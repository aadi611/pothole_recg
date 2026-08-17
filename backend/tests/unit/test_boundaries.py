"""Tests for the module-boundary checker itself.

A guard that cannot fail is not a guard. These assert it actually catches the
violations it exists to prevent (FR-DET-01, NFR-DEV-04).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

_CHECKER = Path(__file__).resolve().parents[3] / "scripts" / "check_boundaries.py"


def _load_checker() -> Any:
    spec = importlib.util.spec_from_file_location("check_boundaries", _CHECKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_boundaries"] = module
    spec.loader.exec_module(module)
    return module


def _imports(source: str) -> list[tuple[str, int]]:
    return _load_checker()._imported_modules(ast.parse(source))


def test_detects_plain_import() -> None:
    assert ("torch", 1) in _imports("import torch")


def test_detects_from_import() -> None:
    assert ("ultralytics", 1) in _imports("from ultralytics import YOLO")


def test_detects_submodule_import() -> None:
    """`import torch.nn` must be caught, not just bare `import torch`."""
    assert ("torch", 1) in _imports("import torch.nn.functional as F")


def test_detects_import_inside_function() -> None:
    """A deferred import is still a dependency.

    Hiding `import torch` inside a function body is the most likely way this
    boundary gets crossed by accident.
    """
    source = "def load():\n    import torch\n    return torch"
    assert ("torch", 2) in _imports(source)


def test_ignores_relative_imports() -> None:
    assert _imports("from . import base") == []
    assert _imports("from ..core import contracts") == []


def test_ignores_module_name_in_comment_or_string() -> None:
    """AST-based, so mentions in prose do not trip it."""
    source = '# we deliberately avoid torch here\nX = "ultralytics"\n'
    assert _imports(source) == []


def test_real_codebase_passes() -> None:
    assert _load_checker().check() == 0
