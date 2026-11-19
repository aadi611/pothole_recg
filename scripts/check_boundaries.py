
"""Module-boundary enforcement (NFR-DEV-04).

Two architectural rules that are easy to state and easy to violate by accident:

1. Only ``roadwatch/detection/`` may import the model library. This is what
   makes the detector swappable (FR-DET-01) and what keeps ``torch`` out of
   the API container.
2. Only ``roadwatch/db/`` may import SQLAlchemy. This keeps persistence
   concerns from leaking into pure logic modules.

Run via ``python scripts/check_boundaries.py``; wired into CI. Uses the AST
rather than grep so that a matching string in a comment or docstring does not
trip it, and so ``import torch`` inside a function body is still caught.

Exits 1 on violation, printing file, line, and the rule broken.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
PACKAGE = BACKEND / "roadwatch"


@dataclass(frozen=True)
class Rule:
    """One boundary: ``modules`` may only be imported under ``allowed_dir``."""

    name: str
    modules: frozenset[str]
    allowed_dir: str
    reason: str


RULES: tuple[Rule, ...] = (
    Rule(
        name="model-library",
        modules=frozenset({"ultralytics", "torch", "torchvision", "tensorflow", "keras"}),
        allowed_dir="detection",
        reason=(
            "Only roadwatch/detection/ may import the model library, so the "
            "Detector implementation stays swappable (FR-DET-01) and torch "
            "stays out of the API container."
        ),
    ),
    Rule(
        name="orm",
        modules=frozenset({"sqlalchemy", "geoalchemy2", "alembic"}),
        allowed_dir="db",
        reason="Only roadwatch/db/ may import SQLAlchemy.",
    ),
)

# training/ trains models; it necessarily touches the model library and is not
# part of the runtime import graph of either app.
EXEMPT_DIRS = frozenset({"training"})


def _root_module(name: str) -> str:
    return name.split(".")[0]


def _imported_modules(tree: ast.AST) -> list[tuple[str, int]]:
    """Every module imported anywhere in the file, with its line number."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((_root_module(a.name), node.lineno) for a in node.names)
        # Relative imports (level > 0) have no external module to check.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((_root_module(node.module), node.lineno))
    return found


def check() -> int:
    if not PACKAGE.is_dir():
        print(f"error: package not found at {PACKAGE}", file=sys.stderr)
        return 1

    violations: list[str] = []

    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        top_dir = rel.parts[0] if len(rel.parts) > 1 else ""

        if top_dir in EXEMPT_DIRS:
            continue

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path}: could not parse: {exc}")
            continue

        for module, lineno in _imported_modules(tree):
            for rule in RULES:
                if module in rule.modules and top_dir != rule.allowed_dir:
                    violations.append(
                        f"{path.relative_to(BACKEND)}:{lineno}: "
                        f"imports '{module}' outside roadwatch/{rule.allowed_dir}/\n"
                        f"    rule [{rule.name}]: {rule.reason}"
                    )

    if violations:
        print("Module boundary violations:\n", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(f"\n{len(violations)} violation(s).", file=sys.stderr)
        return 1

    print(f"Module boundaries OK ({len(RULES)} rules enforced).")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
