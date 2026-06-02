"""LFR tree-shake pass — prune unreachable entities before emission.

After the entity store is fully populated (all stdlib imports resolved),
`tree_shake` walks reachability from a set of *root* entity ids and returns
the subset that should survive into the emitted LFR.

## Roots

Every entity declared directly in a floor or party file the user lowered.
NOT roots: stdlib catalog content (monster definitions, spell definitions,
item definitions) that has no authored instance in the floor or party.

## What survives (kept set)

Starting from roots, BFS through:

1. **`kind:` value** — the entity's declared kind.  Follow to the kind
   entity, then walk its parent chain (recursive).  Every kind in the
   ancestor chain is kept.

2. **List-valued property references** — for each list-valued property,
   inspect each element.  If the element is a snake_case identifier that
   resolves to a known entity id in the entity store, mark it reachable.
   Heuristic: lowercase letters/digits/underscores, length ≥ 2.
   Examples: `spells: [fire_bolt, magic_missile]`,
   `inventory: [longsword, leather]`.

3. **Subentities** — H5 children of any kept entity are ALWAYS kept with
   their parent.  Subentities are never pruned independently.

4. **`requires: [id, ...]`** — explicit author opt-in escape hatch.
   Treats each id as a hard reference.  Use to anchor entities referenced
   by dynamic Luau (`engine.create_named_entity("ghoul")`) that static
   analysis cannot see.

## Always kept (never pruned)

- **All verbs** (`kind: verb`) — the player's input vocabulary.
- **Subentities of any kept entity** — as above.
- **Kinds in the parent chain of any kept entity** — so kind resolution
  at lower time still works.

## What gets pruned

Mostly stdlib catalog content: monster definitions with no instantiated
entity, spell definitions not in any kept entity's `spells:` list, item
definitions not in any kept entity's `inventory:` / `equipment:` lists,
etc.

## `requires:` escape hatch

If a Luau trigger calls `engine.create_named_entity("ghoul")`, the parser
cannot statically see that reference.  Add `- requires: [ghoul]` to any
entity on the floor (or the floor itself via a top-level property) to keep
`ghoul` through the tree-shake pass.

The C engine (luau_bindings.cpp) emits a structured JSONL error event when
`engine.create_named_entity` is called with a name not in the LFR:

    {"type":"error","data":{"code":"spawn_failed","message":
     "spawn_failed: entity 'ghoul' not in LFR. Add `- requires: [ghoul]`
      to a root entity to keep it through tree-shake."}}

See also: docs/design/PARSER.md §9 (tree-shake pass) and
docs/design/LFR.md §12 (requires: escape hatch).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import FMLEntity, Floor

# ── Snake-case identifier heuristic ──────────────────────────────────────────
# Matches identifiers that could plausibly be entity ids:
# all lowercase letters, digits, underscores; at least 2 chars; starts with
# a letter or underscore (entity ids must start with a letter per EntityId
# validation, but we're liberal here and let the entity-store lookup filter).
_SNAKE_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]+$")


def _is_snake_case_ident(s: str) -> bool:
    """True if `s` looks like it could be a snake_case entity id."""
    return bool(_SNAKE_IDENT_RE.fullmatch(s))


# ── Core tree-shake pass ──────────────────────────────────────────────────────


def tree_shake(
    floor: "Floor",
    root_ids: set[str],
) -> set[str]:
    """Return the set of entity ids that should survive the prune.

    Parameters
    ----------
    floor:
        The fully-populated Floor (all imports resolved).
    root_ids:
        The *authored* entity ids — those declared in the floor/party files
        the user explicitly lowered.  Stdlib catalog entities have ids too,
        but they are NOT roots unless the author explicitly references them.

    Returns
    -------
    set[str]
        All entity ids that should be emitted.  The caller filters
        ``floor.entities`` to this set before passing it to the emitter.
    """
    entity_store = floor.entities
    kept: set[str] = set()
    worklist: list[str] = []

    # Step 1: always keep all verbs (and their subentities).
    for eid, ent in entity_store.items():
        if ent.kind == "verb":
            kept.add(eid)
            for sub in ent.subentities:
                kept.add(sub.id)

    # Step 2: seed worklist with roots.
    for eid in root_ids:
        if eid not in kept:
            worklist.append(eid)

    # Step 3: BFS reachability.
    while worklist:
        eid = worklist.pop()
        if eid in kept:
            continue
        kept.add(eid)

        ent = entity_store.get(eid)
        if ent is None:
            # Reference to an id that doesn't exist — keep the reference
            # in kept so the caller doesn't try to emit it, but don't crash.
            continue

        # Subentities always ride with parent.
        for sub in ent.subentities:
            if sub.id not in kept:
                kept.add(sub.id)

        # Kind chain — every kind in the ancestor chain is needed for
        # inheritance resolution at lower time.
        _enqueue_kind_chain(floor, ent.kind, kept, worklist)

        # requires: explicit opt-in escape hatch.
        requires = ent.properties.get("requires")
        if requires:
            if isinstance(requires, str):
                requires = [requires]
            if isinstance(requires, list):
                for req in requires:
                    if isinstance(req, str) and req not in kept:
                        worklist.append(req)

        # List-valued property heuristic: if any element looks like a
        # snake_case entity id AND is known in the entity store, follow it.
        for prop_name, prop_value in ent.properties.items():
            if prop_name == "requires":
                continue  # already handled above
            if isinstance(prop_value, list):
                for item in prop_value:
                    if (
                        isinstance(item, str)
                        and _is_snake_case_ident(item)
                        and item in entity_store
                        and item not in kept
                    ):
                        worklist.append(item)

    return kept


def _enqueue_kind_chain(
    floor: "Floor",
    kind_name: str | None,
    kept: set[str],
    worklist: list[str],
) -> None:
    """Walk the kind chain and enqueue each kind id that is an entity."""
    if not kind_name:
        return
    entity_store = floor.entities
    for k in floor.kind_chain(kind_name):
        if k in entity_store and k not in kept:
            worklist.append(k)


# ── Convenience: collect root ids from Floor.entry_point_entity_ids ───────────


def roots_from_floor(floor: "Floor") -> set[str]:
    """Extract the set of root entity ids for tree-shaking.

    Strategy (in preference order):

    1. **stdlib_entity_ids is populated** — the parser tracked which entity ids
       came from stdlib imports.  Roots = all entity ids NOT in stdlib_entity_ids.
       This is the most accurate discriminator.

    2. **entry_point_entity_ids is populated but stdlib_entity_ids is not** —
       fall back to entry_point_entity_ids.  This is conservative for multi-file
       floors (may miss authored entities in sub-files) but never prunes authored
       content that IS in entry_point_entity_ids.

    3. **Neither is populated** — treat every entity as a root.  No pruning
       occurs.  This matches pre-tree-shake conservative behaviour.
    """
    all_ids = set(floor.entities.keys())

    # Case 1: stdlib_entity_ids was populated by the parser.  Use it.
    if floor.stdlib_entity_ids:
        # Roots = everything that did NOT come from stdlib.
        # Note: verbs and kind_definitions in stdlib are always kept by the
        # tree_shake() pass itself (kind:verb is never pruned), so we don't
        # need to exclude them from roots here — the BFS handles it.
        return all_ids - floor.stdlib_entity_ids

    # Case 2: entry_point_entity_ids was set (parser ran in entry-point mode).
    if floor.entry_point_entity_ids is not None:
        # Use entry-point ids directly.  Conservative but safe.
        return set(floor.entry_point_entity_ids)

    # Case 3: no tracking info — all entities are roots, no pruning.
    return all_ids
