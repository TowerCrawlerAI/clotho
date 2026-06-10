"""Regression tests for issue #29 — stdlib import classification.

The bug: `is_stdlib_import` was decided by `"stdlib" in resolved.parts`, i.e. by
the resolved path *text*. When a consumer checked the stdlib repo out into a
directory NOT named `stdlib` (e.g. `_stdlib/`) and pointed `FML_STDLIB_PATH` at
it, the substring check failed, the catalog was never classified as stdlib, and
tree-shake kept the entire catalog (the published floor.lua ballooned).

The fix classifies by the import NAME (`stdlib`), which every branch of
`_resolve_import_path` knows — independent of the resolved directory's name.
"""

from __future__ import annotations

from pathlib import Path

from fml_parser import parse_fml
from fml_parser.parser import _resolve_import_path
from fml_parser.tree_shake import roots_from_floor, tree_shake


# ─── Unit: _resolve_import_path classifies by name, not path text ────────────


def test_env_override_to_non_stdlib_dir_is_still_classified_stdlib(
    monkeypatch, tmp_path
):
    """A stdlib checked out into `_stdlib/` (not `stdlib/`) and reached via
    FML_STDLIB_PATH must still be classified as stdlib — the whole point of #29.
    The resolved path contains NO `stdlib` part, so the old substring check
    would have returned False here."""
    catalog = tmp_path / "_stdlib"
    catalog.mkdir()
    index = catalog / "index.md"
    index.write_text("# Catalog\n", encoding="utf-8")
    monkeypatch.setenv("FML_STDLIB_PATH", str(index))

    resolved, is_stdlib = _resolve_import_path("stdlib", tmp_path)

    assert resolved == index.resolve()
    assert is_stdlib is True
    # Prove the directory is genuinely not named "stdlib" — the old
    # `"stdlib" in resolved.parts` heuristic would have classified this False.
    assert "stdlib" not in resolved.parts


def test_relative_import_is_not_classified_stdlib(tmp_path):
    """A plain relative import (e.g. a floor's own room file) is never stdlib —
    even if its path happens to contain a `stdlib` component."""
    resolved, is_stdlib = _resolve_import_path("rooms/cell.md", tmp_path)
    assert is_stdlib is False

    # And a relative path that DOES contain a "stdlib" component must NOT be
    # misclassified as stdlib (the inverse failure mode the old check had).
    resolved2, is_stdlib2 = _resolve_import_path("stdlib_notes/cell.md", tmp_path)
    assert is_stdlib2 is False


# ─── Integration: catalog is tree-shaken through a non-"stdlib" checkout dir ──


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_catalog_tree_shaken_when_stdlib_dir_not_named_stdlib(monkeypatch, tmp_path):
    """End-to-end (#29): lower a floor whose `[stdlib](stdlib)` import resolves —
    via FML_STDLIB_PATH — to a directory NOT named `stdlib`, and assert the
    unreferenced catalog entity is classified as stdlib and pruned by tree-shake,
    while the floor's own authored entity survives."""
    # Catalog checked out into `_stdlib/` (deliberately NOT named "stdlib").
    _write(
        tmp_path / "_stdlib" / "index.md",
        "# Test Catalog\n\n### orphan_widget\n\n- kind: thing\n\n"
        "> An unreferenced catalog entity that must be tree-shaken away.\n",
    )
    # Floor imports the stdlib by name and declares its own room.
    _write(
        tmp_path / "floor" / "index.md",
        "# Test Floor\n\n[stdlib](stdlib)\n\n### cell\n\n- kind: room\n\n"
        "> A bare stone cell.\n",
    )
    monkeypatch.setenv("FML_STDLIB_PATH", str(tmp_path / "_stdlib" / "index.md"))

    floor_index = tmp_path / "floor" / "index.md"
    floor = parse_fml(floor_index.read_text(encoding="utf-8"), floor_index)

    widget = next(e for e in floor.entities.values() if e.kind == "thing")
    cell = next(e for e in floor.entities.values() if e.kind == "room")

    # The fix: the catalog entity is classified stdlib despite the `_stdlib` dir
    # name. (Under the old substring check this set was empty → regression.)
    assert widget.id in floor.stdlib_entity_ids
    assert cell.id not in floor.stdlib_entity_ids

    # Downstream: tree-shake prunes the unreferenced catalog entity, keeps the
    # authored floor entity. This is the "catalog still tree-shaken" guarantee.
    kept = tree_shake(floor, roots_from_floor(floor))
    assert cell.id in kept
    assert widget.id not in kept
