"""Assert application layers do not import concrete FLAN-T5 backend."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORBIDDEN_IMPORT_ROOTS = {
    "losses",
    "rewards",
    "training",
    "repositories",
}


def _imports_flan(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "flan_t5" in node.module or node.module.endswith(".flan_t5"):
                return True
            for alias in node.names:
                if alias.name == "FLANT5Backend":
                    return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "flan_t5" in alias.name or alias.name == "FLANT5Backend":
                    return True
    return False


def test_no_flan_imports_outside_backend_adapter():
    offenders = []
    for root_name in FORBIDDEN_IMPORT_ROOTS:
        root = ROOT / root_name
        for path in root.rglob("*.py"):
            if _imports_flan(path):
                # Allow docstring mentions only — AST import check already filters
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"FLAN-T5 imports leaked into: {offenders}"
