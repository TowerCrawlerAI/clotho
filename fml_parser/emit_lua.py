"""FML Floor → Luau LFR emitter.

Single public function: `emit_lua(floor, source_path, version) -> str`.

See `docs/design/LFR.md` for the canonical output format.

Prose template compilation:
  ``_compile_prose(prose_val, prop_key, source_path)`` lowers a ``ProseValue``
  to a Luau ``function(self, ctx) ... end`` closure per ``docs/design/PROSE.md``.

Kind-property inheritance:
  Before emitting an entity's property table, ``_resolve_inherited_properties``
  walks the kind chain (instance → parent kind → grandparent → … → root) and
  merges properties bottom-up so root defaults appear first and instance values
  win.  Triggers follow the same rule: instance trigger for a stage name
  overrides the kind's trigger of the same name; otherwise the kind's trigger
  is used.  See ``docs/design/OBJECT_MODEL.md`` §7 (Parent chain).

Tree-shake pass:
  ``emit_lua`` accepts an optional ``kept`` set of entity ids produced by
  ``parser.tree_shake.tree_shake()``.  When supplied, only entities whose id
  appears in ``kept`` are emitted.  Verbs and kind-chain ancestors are always
  included by the tree-shake pass (see ``docs/design/PARSER.md`` §9).
  When ``kept`` is ``None`` (the default) all entities are emitted — the
  pre-tree-shake conservative behaviour.
"""

from __future__ import annotations

import re
from typing import Any

from .dice_value import LuauCode, ProseValue
from .errors import FmlSyntaxError
from .models import ActionLine, BareLink, FMLEntity, Floor, OutputLine, Predicate, PropertySet, Trigger, TriggerBodyItem, UnderstandDirective

# ─── Section grouping ─────────────────────────────────────────────────────────

# Kinds that carry stdlib metadata, not floor-data — skip them entirely.
_SKIP_KINDS = frozenset(["kind_definition", "verb"])

# Lua identifier pattern — keys that don't match get bracket syntax.
_LUA_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Trigger stage word → LFR prefix, including the InsteadOf special case.
_STAGE_MAP: dict[str, str] = {
    "Test":       "test",
    "InsteadOf":  "instead_of",
    "Before":     "before",
    "On":         "on",
    "After":      "after",
    "Report":     "report",
}


# ─── Kind-property inheritance ────────────────────────────────────────────────


def _resolve_inherited_properties(
    entity: FMLEntity, floor: Floor
) -> tuple[dict[str, Any], list[Trigger]]:
    """Walk the entity's kind chain and return fully-merged (properties, triggers).

    Algorithm (OBJECT_MODEL.md §7 — parent chain only, no applique/mix-in):

    1. Build the chain [instance_kind, parent_kind, grandparent_kind, …].
    2. For each kind in the chain (root first, instance last), look up the
       ``kind_definition`` entity in ``floor.entities`` by its ``name`` property.
    3. Merge properties root→…→parent→instance, instance wins per key.
       - Scalar / function / prose: instance slot replaces parent slot.
       - List: instance list replaces parent list (no concatenation — per spec).
    4. Merge triggers by stage-slot key (``_trigger_slot_key(trigger.name)``):
       instance trigger replaces parent trigger of the same slot.

    Returns the merged (properties_dict, triggers_list) ready to emit.
    The entity's own ``kind`` and ``id`` / ``name`` fields are NOT in properties.
    Skipped properties: ``name``, ``ancestor`` — these are kind-definition
    metadata, not inheritable world properties.
    """
    # Build kind entity lookup: kind name → kind_definition FMLEntity
    kind_defs: dict[str, FMLEntity] = {}
    for ent in floor.entities.values():
        if ent.kind == "kind_definition":
            kname = ent.properties.get("name")
            if isinstance(kname, str):
                kind_defs[kname] = ent

    # Walk the chain from root to leaf so deeper (more-specific) kinds win.
    chain = floor.kind_chain(entity.kind)  # [instance_kind, parent, …, root]
    # Reverse so we merge root → … → parent → instance (instance wins).
    chain_reversed = list(reversed(chain))

    # Skip internal kind-definition metadata keys — not world-entity properties.
    # 'kind' appears in properties because the parser stores it there as well as
    # on entity.kind; we don't want 'kind: kind_definition' bleeding into instances.
    _KD_SKIP = frozenset({"name", "ancestor", "attributes", "kind"})

    merged_props: dict[str, Any] = {}
    # Triggers keyed by slot (e.g. "after:Attack") so instance overrides parent.
    trigger_by_slot: dict[str, Trigger] = {}

    for kind_name in chain_reversed:
        kd = kind_defs.get(kind_name)
        if kd is None:
            continue
        # Merge kind's properties (excluding internal metadata).
        for k, v in kd.properties.items():
            if k not in _KD_SKIP:
                merged_props[k] = v
        # Merge kind's triggers (slot-keyed, later overrides earlier).
        for trigger in kd.triggers:
            slot = _trigger_slot_key(trigger.name)
            trigger_by_slot[slot] = trigger

    # Instance properties win over all kind-chain properties.
    for k, v in entity.properties.items():
        merged_props[k] = v

    # Instance triggers win over kind-chain triggers of the same slot.
    for trigger in entity.triggers:
        slot = _trigger_slot_key(trigger.name)
        trigger_by_slot[slot] = trigger

    # Rebuild trigger list in a stable order: kind-chain triggers first (by
    # their slot), then instance-only triggers appended. Order within each
    # group is preserved from the source.
    seen_slots: set[str] = set()
    merged_triggers: list[Trigger] = []

    # Kind-chain triggers (in reverse-chain order, then slot order)
    for kind_name in chain_reversed:
        kd = kind_defs.get(kind_name)
        if kd is None:
            continue
        for trigger in kd.triggers:
            slot = _trigger_slot_key(trigger.name)
            if slot not in seen_slots:
                # Only add if this is the winning trigger for the slot.
                if trigger_by_slot[slot] is trigger:
                    merged_triggers.append(trigger)
                    seen_slots.add(slot)

    # Instance triggers: add all (already de-dup'd by slot if overriding kind).
    for trigger in entity.triggers:
        slot = _trigger_slot_key(trigger.name)
        if slot not in seen_slots:
            merged_triggers.append(trigger)
            seen_slots.add(slot)

    return merged_props, merged_triggers


# ─── Stage name → table key ───────────────────────────────────────────────────

def _verb_stage_key(trigger_name: str) -> str:
    """'Test Go' → 'test', 'InsteadOf Go' → 'instead_of', 'On Take' → 'on'."""
    parts = trigger_name.split(None, 1)
    if not parts:
        return "on"
    return _STAGE_MAP.get(parts[0], parts[0].lower())


# ─── Graph emitter (F6 — binding-surface LFR) ────────────────────────────────

# Built-in verbs the engine bootstraps (tg_verb_bootstrap): take, drop, put-in, go.
# Re-emitting any of these via define_verb fails on the duplicate name, so skip
# them. The engine names "put-in" with a hyphen, but FML EntityIds are snake_case
# (^[a-z][a-z0-9_]*$), so a content verb intended as put-in is slugged "put_in" —
# match that form. ("put" alone is NOT a built-in and must remain emittable.)
_BUILTIN_VERBS = frozenset(["take", "drop", "put_in", "go"])

# Kinds that are schema metadata, not world instances.
_SCHEMA_KINDS = frozenset(["kind_definition", "verb", "event"])


def emit_lua_graph(
    floor: Floor,
    source_path: str | None = None,
    stdlib_module: bool = False,
) -> str:
    """Emit a DB-init LFR for the LPG engine binding surface.

    Calls ``engine.*`` bindings to populate the graph and register verbs.

    Two modes (mirroring the legacy emit_lua / emit_lua_stdlib_module split):

    * **stdlib module** (``stdlib_module=True``): emit SCHEMA + PROCEDURES only —
      ``define_kind`` + ``define_relation`` + ``define_verb``. No world instances,
      no start actor. This is loaded first in the assembly (``--stdlib``).
    * **floor** (default): emit INSTANCES — ``create_node`` + ``relate`` +
      floor-local ``define_verb`` + ``set_start_actor``. Entities/verbs that came
      from a stdlib import (``floor.stdlib_entity_ids``) are skipped — the stdlib
      module already registered them; the floor only owns its own content.

    Emission order: header → [stdlib: kinds] → relations(map) → create_node →
    relate → define_verb (skip engine built-ins) → [floor: set_start_actor].

    Deterministic: same input → bit-identical output.
    """
    src = source_path or "unknown.md"
    parts: list[str] = []

    # Scope filter: in floor mode, a non-None stdlib_entity_ids means "only emit
    # entities this floor owns" (skip imported stdlib catalog/kinds/verbs). In
    # stdlib-module mode every entity is in scope.
    stdlib_ids = floor.stdlib_entity_ids or set()

    def _in_scope(ent: FMLEntity) -> bool:
        if stdlib_module:
            return True
        return ent.id not in stdlib_ids

    # 1. Header ----------------------------------------------------------------
    lua_path = src.replace(".md", ".lua") if src else "unknown.lua"
    parts.append(f"-- {lua_path}")
    parts.append(f"-- Generated by fml-parser (graph emitter) from {src}")
    hash_val = floor.fml_source_hash or "unknown"
    parts.append(f"-- fml-source-hash: {hash_val}")
    parts.append("")

    # 2. Kinds -----------------------------------------------------------------
    # Only the stdlib module registers kinds; a floor relies on the stdlib it
    # imports (loaded first in the assembly) to have defined them.
    # Emit in section_mappings order when available, else stable declaration order.
    kind_entities: list[FMLEntity] = []
    if not stdlib_module:
        kind_entities = []
    elif floor.section_mappings:
        # Collect kind names from section_mappings (preserves declared order).
        seen_kind_names: set[str] = set()
        kind_name_order: list[str] = []
        for _heading, kind_id in floor.section_mappings:
            if kind_id not in seen_kind_names:
                kind_name_order.append(kind_id)
                seen_kind_names.add(kind_id)
        # Build a lookup: kind-definition entity by its `name` property.
        kdef_by_name: dict[str, FMLEntity] = {}
        for ent in floor.entities.values():
            if ent.kind == "kind_definition":
                kname = ent.properties.get("name")
                if isinstance(kname, str):
                    kdef_by_name[kname] = ent
        # Emit in section-mapping order, then any remaining kind_definition entities.
        seen_ids: set[str] = set()
        for kname in kind_name_order:
            kd = kdef_by_name.get(kname)
            if kd is not None and kd.id not in seen_ids:
                kind_entities.append(kd)
                seen_ids.add(kd.id)
        for ent in floor.entities.values():
            if ent.kind == "kind_definition" and ent.id not in seen_ids:
                kind_entities.append(ent)
                seen_ids.add(ent.id)
    else:
        kind_entities = [e for e in floor.entities.values() if e.kind == "kind_definition"]

    if kind_entities:
        parts.append("-- Kinds")
    for ent in kind_entities:
        kname = ent.properties.get("name")
        if not isinstance(kname, str):
            kname = ent.name
        parts.append(f"engine.define_kind({_lua_string(kname)})")
    if kind_entities:
        parts.append("")

    # 3. Custom relations — emit 'map' if any world entity declares exits -------
    # World instances are a floor concern; in stdlib-module mode there are none.
    world_entities = (
        []
        if stdlib_module
        else [
            e for e in floor.entities.values()
            if e.kind not in _SCHEMA_KINDS and _in_scope(e)
        ]
    )
    need_map_relation = any(
        e.properties.get("exits") or e.properties.get("exit")
        for e in world_entities
    )
    if need_map_relation:
        parts.append("-- Custom relations")
        parts.append(
            'local _rel_map = engine.define_relation("map", { symmetric = true, role = "map" })'
        )
        parts.append("")

    # 4. World entity nodes ---------------------------------------------------
    if world_entities:
        parts.append("-- World entities")
    for ent in world_entities:
        scalar_props = _collect_scalar_props(ent)
        if scalar_props:
            prop_pairs = ", ".join(
                f"{_lua_key(k)} = {_lua_value(v)}" for k, v in scalar_props.items()
            )
            parts.append(
                f"local n_{ent.id} = engine.create_node({{ name = {_lua_string(ent.name)}, {prop_pairs} }})"
            )
        else:
            parts.append(
                f"local n_{ent.id} = engine.create_node({{ name = {_lua_string(ent.name)} }})"
            )
    if world_entities:
        parts.append("")

    # 4b. Descriptions — the entity's prose becomes a "description" property the
    # engine's `look` reads out. (set_prop after create_node so the node exists.)
    desc_lines: list[str] = []
    for ent in world_entities:
        desc = _entity_description(ent)
        if desc:
            desc_lines.append(
                f'engine.set_prop(n_{ent.id}, "description", {_lua_string(desc)})'
            )
    if desc_lines:
        parts.append("-- Descriptions")
        parts.extend(desc_lines)
        parts.append("")

    # 5. Relations (after all nodes exist) ------------------------------------
    world_ids: set[str] = {e.id for e in world_entities}
    relation_lines: list[str] = []

    for ent in world_entities:
        # Containment: at_location / location property → relate("in", ...).
        # (FML uses `at_location` for items in a room; `location` is the older
        # form. First match wins.)
        container = ent.properties.get("at_location") or ent.properties.get("location")
        if isinstance(container, str) and container in world_ids:
            relation_lines.append(
                f'engine.relate("in", n_{ent.id}, n_{container})'
            )

    # Exits → 'map' edges. The relation is symmetric, so emit each unordered
    # room pair once and skip self-loops (avoids duplicate/degenerate edges from
    # rooms that declare reciprocal or self exits).
    seen_map_pairs: set[frozenset[str]] = set()
    for ent in world_entities:
        exits = ent.properties.get("exits") or ent.properties.get("exit")
        if isinstance(exits, dict):
            for _direction, dest_slug in exits.items():
                if not (isinstance(dest_slug, str) and dest_slug in world_ids):
                    continue
                if dest_slug == ent.id:
                    continue  # self-loop
                pair = frozenset((ent.id, dest_slug))
                if pair in seen_map_pairs:
                    continue
                seen_map_pairs.add(pair)
                relation_lines.append(
                    f'engine.relate("map", n_{ent.id}, n_{dest_slug})'
                )

    if relation_lines:
        parts.append("-- Relations")
        parts.extend(relation_lines)
        parts.append("")

    # 6. Verbs ----------------------------------------------------------------
    # stdlib module: all verbs. floor: only floor-local verbs (skip imported).
    verb_entities = [
        e for e in floor.entities.values()
        if e.kind == "verb" and _in_scope(e)
    ]
    # Build a set of understand_directives indexed by verb_id for O(1) lookup.
    ud_by_verb: dict[str, list[str]] = {}
    for directive in floor.understand_directives:
        for phrase in directive.phrases:
            if isinstance(phrase, str) and phrase:
                ud_by_verb.setdefault(directive.verb_id, []).append(phrase)

    for ent in verb_entities:
        verb_name = ent.id
        # Skip built-in verbs.
        if verb_name in _BUILTIN_VERBS:
            continue

        parts.append(f"-- verb: {verb_name}")
        parts.append(f"engine.define_verb({{")
        parts.append(f"    name = {_lua_string(verb_name)},")

        # Grammar fields from properties.
        grammar_keys = ("noun", "noun2", "preposition", "scope", "target_rel", "subject_is_src")
        for key in grammar_keys:
            val = ent.properties.get(key)
            if val is not None:
                parts.append(f"    {_lua_key(key)} = {_lua_value(val)},")

        # Aliases: from props aliases/phrases + understand_directives.
        alias_list: list[str] = []
        for alias_key in ("aliases", "phrases"):
            alias_val = ent.properties.get(alias_key)
            if alias_val is not None:
                if isinstance(alias_val, str):
                    alias_list.append(alias_val)
                elif isinstance(alias_val, list):
                    alias_list.extend(a for a in alias_val if isinstance(a, str))
        alias_list.extend(ud_by_verb.get(verb_name, []))
        if alias_list:
            items_lua = ", ".join(_lua_string(a) for a in alias_list)
            parts.append(f"    aliases = {{ {items_lua} }},")

        # Stages: only from lua/luau script triggers.
        lua_triggers = [
            t for t in ent.triggers
            if t.script is not None and t.script.language in ("lua", "luau")
        ]
        non_lua_triggers = [
            t for t in ent.triggers
            if t.script is None or t.script.language not in ("lua", "luau")
        ]

        if lua_triggers:
            parts.append("    stages = {")
            for trigger in lua_triggers:
                stage_key = _verb_stage_key(trigger.name)
                parts.append(f"        {stage_key} = function(ctx)")
                for line in trigger.script.source.splitlines():  # type: ignore[union-attr]
                    parts.append(f"            {line}" if line.strip() else "")
                parts.append("        end,")
            parts.append("    },")

        parts.append("})")

        # Warn about non-lua trigger bodies.
        for trigger in non_lua_triggers:
            parts.append(
                f"-- WARNING: verb {_lua_string(verb_name)} trigger {_lua_string(trigger.name)}"
                f" has a non-script FML body; not transpiled (lean scope)"
                f" — rewrite as emergent or native lua"
            )

        parts.append("")

    # 7. Start actor (floor mode only — a stdlib module has no player) --------
    if not stdlib_module:
        start_id: str | None = None

        # (a) an explicit start-actor entity named by a floor property.
        for prop_key in ("start_actor", "start", "player"):
            val = floor.properties.get(prop_key)
            if isinstance(val, str) and val in world_ids:
                start_id = val
                break

        # (b) an entity whose kind chain includes 'player'/'pc'.
        if start_id is None:
            for ent in world_entities:
                chain = ent.kind_chain or [ent.kind]
                if any(k in ("player", "pc") for k in chain):
                    start_id = ent.id
                    break

        if start_id is not None:
            parts.append(f"engine.set_start_actor(n_{start_id})")
        else:
            # (c) No player entity exists (the legacy host synthesized one). If
            # the floor names a `start_location`/`start` ROOM, synthesize a
            # player node there and make it the start actor — graph mode has no
            # player bootstrap, so the floor LFR must create the actor.
            start_room = None
            for prop_key in ("start_location", "start", "start_room"):
                val = floor.properties.get(prop_key)
                if isinstance(val, str) and val in world_ids:
                    start_room = val
                    break
            if start_room is not None:
                parts.append("-- Synthesized player (no player entity in source)")
                parts.append('local n__player = engine.create_node({ name = "you" })')
                parts.append(f'engine.relate("in", n__player, n_{start_room})')
                parts.append("engine.set_start_actor(n__player)")
            else:
                parts.append(
                    "-- engine.set_start_actor: no player/start entity determined; set manually"
                )

    parts.append("")
    return "\n".join(parts)


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _flatten_prose_markup(s: str) -> str:
    """Reduce inline Markdown to plain text for an in-game description:
    `[north](#Hall)` → `north`. Leaves other text untouched."""
    return _MD_LINK_RE.sub(r"\1", s)


def _entity_description(ent: FMLEntity) -> str | None:
    """Flatten an entity's prose into a single description string, or None.

    Plain-string prose is used as-is; a ProseValue (duck-typed by its `lines`)
    is joined preserving paragraph breaks. Markdown links are reduced to their
    text. Returns None when there is no prose.
    """
    p = ent.prose
    if isinstance(p, str):
        s = p.strip()
    else:
        lines = getattr(p, "lines", None)
        s = "\n".join(lines).strip() if lines else ""
    if not s:
        return None
    return _flatten_prose_markup(s)


def _collect_scalar_props(ent: FMLEntity) -> dict[str, Any]:
    """Return only scalar (str/int/float/bool) properties, excluding location/exits/exit.

    ``name`` is also skipped: it is emitted explicitly as the create_node name,
    and repeating it would produce a duplicate Lua table key.
    """
    _SKIP = frozenset({"name", "location", "exits", "exit"})
    result: dict[str, Any] = {}
    for k, v in ent.properties.items():
        if k in _SKIP:
            continue
        # Accept only primitive scalars (bool before int, since bool is subclass of int).
        if isinstance(v, (bool, str, float)) or (isinstance(v, int) and not isinstance(v, bool)):
            result[k] = v
    return result


# ─── Public entry points ──────────────────────────────────────────────────────


def emit_lua_stdlib_module(floor: Floor, source_path: str | None = None) -> str:
    """Emit a stdlib FML floor (verbs.md) as a Lua verb-module string.

    Returns a Lua module that ``return``s a table keyed by verb name.
    Each value is a verb descriptor compatible with the C dispatch engine:
    ``{ name, event, aliases, ..., stages = { test=fn, on=fn, ... } }``.

    Also emits ``engine.register_verb_alias`` calls (before the ``return``)
    for each alias listed in a verb's ``aliases`` property, so that single-
    word shortcuts like ``"n"`` and multi-word phrases like ``"get off"``
    resolve correctly at dispatch time.

    Handler indirection: verbs with a ``handler`` property but no triggers of
    their own (e.g. ``north`` → ``go_handler``) delegate to named handler
    functions emitted at module level.  The canonical implementation is the
    entity whose ``handler`` value matches the handler name AND which also
    carries triggers (e.g. the ``go`` entity).

    Only entities whose ``kind == "verb"`` are emitted.
    """
    src = source_path or "unknown.md"
    parts = [f"-- stdlib verb module — generated from {src}"]

    # --- Pass 1: build handler_map ----------------------------------------
    # handler_name → entity that implements it (has triggers + matching handler)
    handler_map: dict[str, "FMLEntity"] = {}
    for entity in floor.entities.values():
        if entity.kind != "verb":
            continue
        if not entity.triggers:
            continue
        h = entity.properties.get("handler", "")
        if h and h not in handler_map:
            handler_map[h] = entity

    # Collect alias registration calls first; emit before the return table.
    alias_calls: list[str] = []
    for entity in floor.entities.values():
        if entity.kind != "verb":
            continue
        aliases = entity.properties.get("aliases")
        if aliases:
            if isinstance(aliases, str):
                aliases = [aliases]
            if isinstance(aliases, list):
                for phrase in aliases:
                    if isinstance(phrase, str) and phrase:
                        alias_calls.append(
                            f"engine.register_verb_alias("
                            f"{_lua_string(phrase)}, {_lua_string(entity.id)})"
                        )
        phrases = entity.properties.get("phrases")
        if phrases:
            if isinstance(phrases, str):
                phrases = [phrases]
            if isinstance(phrases, list):
                for phrase in phrases:
                    if isinstance(phrase, str) and phrase:
                        alias_calls.append(
                            f"engine.register_verb_alias("
                            f"{_lua_string(phrase)}, {_lua_string(entity.id)})"
                        )

    # Understand directives (floor-level; may carry full-command targets like "go north").
    for directive in floor.understand_directives:
        for phrase in directive.phrases:
            if isinstance(phrase, str) and phrase:
                alias_calls.append(
                    f"engine.register_verb_alias("
                    f"{_lua_string(phrase)}, {_lua_string(directive.verb_id)})"
                )

    if alias_calls:
        parts.append("")
        parts.append("-- verb alias registrations")
        parts.extend(alias_calls)

    # Named handler closures: one local per (handler_name, stage_key) pair.
    handler_fn_names: dict[str, dict[str, str]] = {}  # handler_name → {stage_key → local_name}
    for handler_name, impl_entity in handler_map.items():
        stage_fns: dict[str, str] = {}
        for trigger in impl_entity.triggers:
            stage_key = _verb_stage_key(trigger.name)
            local_name = f"_h_{handler_name}_{stage_key}"
            body = _trigger_body(trigger)
            parts.append(f"\nlocal {local_name} = function(ctx)")
            if body:
                for line in body.splitlines():
                    parts.append(f"    {line}" if line.strip() else "")
            parts.append("end")
            stage_fns[stage_key] = local_name
        handler_fn_names[handler_name] = stage_fns

    parts.append("")
    parts.append("return {")

    for entity in floor.entities.values():
        if entity.kind != "verb":
            continue

        parts.append(f"    [{_lua_string(entity.id)}] = {{")
        parts.append(f"        name = {_lua_string(entity.name)},")

        for k, v in entity.properties.items():
            parts.append(f"        {_lua_key(k)} = {_lua_value(v)},")

        handler_name = entity.properties.get("handler", "")
        use_handler_fns = (
            not entity.triggers
            and handler_name
            and handler_name in handler_fn_names
        )

        parts.append("        stages = {")
        if entity.triggers:
            # Verb defines its own stages inline.
            for trigger in entity.triggers:
                stage_key = _verb_stage_key(trigger.name)
                body = _trigger_body(trigger)
                parts.append(f"            {stage_key} = function(ctx)")
                if body:
                    for line in body.splitlines():
                        parts.append(f"                {line}" if line.strip() else "")
                parts.append("            end,")
        elif use_handler_fns:
            # Delegate to named handler closures.
            for stage_key, local_name in handler_fn_names[handler_name].items():
                parts.append(f"            {stage_key} = {local_name},")
        parts.append("        },")
        parts.append("    },")

    parts.append("}")
    return "\n".join(parts)


def emit_lua(
    floor: Floor,
    source_path: str | None = None,
    version: str = "0.1",
    kept: set[str] | None = None,
) -> str:
    """Emit a Floor model as a Luau LFR file string.

    source_path: original .md source path for the header comment.
    version: tower-lower version string for the header comment.
    kept: optional set of entity ids from ``tree_shake()``; when supplied,
        only entities in ``kept`` are emitted.  Pass ``None`` to emit all
        entities (pre-tree-shake conservative behaviour).
    """
    parts: list[str] = []
    _emit_header(parts, floor, source_path, version)
    _emit_floor_table(parts, floor)
    _emit_entity_finder(parts)
    _emit_verb_aliases(parts, floor)

    # Build the section → [entity] map from the floor, respecting section order.
    # Apply tree-shake filter when a kept set is provided.
    section_locals, entity_section = _build_sections(floor, kept=kept)

    for local_name, entities in section_locals:
        if not entities:
            _emit_empty_section(parts, local_name)
        else:
            _emit_section(parts, local_name, entities, floor)

    # Catch-all for kinds that didn't match any section.
    other = entity_section.get(None, [])
    if other:
        parts.append("-- --- Other ---")
        parts.append("local other = {}")
        for entity in other:
            _emit_entity_table(parts, "other", entity, floor)
        _emit_section_triggers(parts, "other", other, floor)
        parts.append("")

    _emit_return(parts, section_locals, other)
    return "\n".join(parts) + "\n"


# ─── Header ───────────────────────────────────────────────────────────────────


def _emit_header(
    parts: list[str],
    floor: Floor,
    source_path: str | None,
    version: str,
) -> None:
    lua_path = source_path.replace(".md", ".lua") if source_path else "unknown.lua"
    parts.append(f"-- {lua_path}")
    src = source_path or "unknown.md"
    parts.append(f"-- Generated by tower-lower {version} from {src}")
    hash_val = floor.fml_source_hash or "unknown"
    parts.append(f"-- fml-source-hash: {hash_val}")
    parts.append("")


# ─── Floor table ─────────────────────────────────────────────────────────────


def _emit_floor_table(parts: list[str], floor: Floor) -> None:
    parts.append("local floor = {")
    parts.append(f"    name = {_lua_string(floor.name)},")
    # Floor-level properties (excluding any that would collide with name/prose).
    for k, v in floor.properties.items():
        if k in ("name", "prose"):
            continue
        parts.append(f"    {_lua_key(k)} = {_lua_value(v)},")
    if floor.prose:
        if isinstance(floor.prose, ProseValue):
            parts.append(f"    prose = {_compile_prose(floor.prose, 'prose')},")
        elif isinstance(floor.prose, LuauCode):
            parts.append(f"    prose = {floor.prose.source},")
        else:
            parts.append(f"    prose = {_lua_string(floor.prose)},")
    if floor.imports:
        items = ", ".join(_lua_string(i) for i in floor.imports)
        parts.append(f"    imports = {{ {items} }},")
    parts.append("    extra = {},")
    parts.append("}")
    parts.append("")


# ─── Entity finder helper ─────────────────────────────────────────────────────


def _emit_entity_finder(parts: list[str]) -> None:
    """Emit a _find_entity(slug) helper that resolves an FML slug to an integer
    entity_id at runtime.  Triggers that reference named entities (BareLink body
    items) use this to obtain the C-side id for engine.call_trigger.

    The helper memoises results in a module-local table so repeat calls are O(1).
    """
    parts.append("-- --- Entity finder (runtime slug → entity_id) ---")
    parts.append("local _eid_cache = {}")
    parts.append("local function _find_entity(slug)")
    parts.append("    if _eid_cache[slug] then return _eid_cache[slug] end")
    parts.append("    local candidates = engine.entities_in_scope(\"global\", 0)")
    parts.append("    for _, eid in ipairs(candidates) do")
    parts.append("        local e = engine.query_entity(eid)")
    parts.append("        if e and e.id == slug then")
    parts.append("            _eid_cache[slug] = eid")
    parts.append("            return eid")
    parts.append("        end")
    parts.append("    end")
    parts.append("    return nil")
    parts.append("end")
    parts.append("")


# ─── Verb alias registration ─────────────────────────────────────────────────


def _emit_verb_aliases(parts: list[str], floor: Floor) -> None:
    """Emit ``engine.register_verb_alias`` calls for every Understand directive.

    One call per phrase in each directive, in declaration order.  The block is
    omitted entirely when there are no directives — keeping the output clean for
    floors that don't use ``**Understand**``.
    """
    if not floor.understand_directives:
        return
    parts.append("-- --- Verb aliases ---")
    for directive in floor.understand_directives:
        for phrase in directive.phrases:
            parts.append(
                f"engine.register_verb_alias({_lua_string(phrase)}, {_lua_string(directive.verb_id)})"
            )
    parts.append("")


# ─── Section building ─────────────────────────────────────────────────────────


def _build_sections(
    floor: Floor,
    kept: set[str] | None = None,
) -> tuple[list[tuple[str, list[FMLEntity]]], dict[str | None, list[FMLEntity]]]:
    """Return (ordered [(local_name, [entity])], {section|None: [entity]}).

    When floor.section_mappings is populated (from stdlib import), entities are
    routed to the declared sections by kind. Without stdlib section mappings,
    all entities land in a single flat ``entities`` table — no guessing.
    Kind definitions and verb declarations are excluded in either case.

    When ``kept`` is provided (from ``tree_shake()``), only entities whose id
    appears in ``kept`` are included in the output sections.  This filters out
    unreachable stdlib catalog content (monsters, spells, items with no authored
    instance) while preserving all verbs and kind-chain ancestors.
    """
    non_skip = [
        e for e in floor.entities.values()
        if e.kind not in _SKIP_KINDS
        and (kept is None or e.id in kept)
    ]

    if not floor.section_mappings:
        # No stdlib: flat table, no hardcoded routing.
        return [("entities", non_skip)], {"entities": non_skip}

    # stdlib-declared sections: route by kind.
    local_order: list[str] = []
    local_kinds: dict[str, list[str]] = {}
    for heading, kind in floor.section_mappings:
        local_name = heading.lower().replace(" ", "_")
        if local_name not in local_kinds:
            local_order.append(local_name)
            local_kinds[local_name] = []
        local_kinds[local_name].append(kind)

    kind_to_local: dict[str, str] = {}
    for local_name, kinds in local_kinds.items():
        for k in kinds:
            if k not in kind_to_local:
                kind_to_local[k] = local_name

    buckets: dict[str | None, list[FMLEntity]] = {}
    for entity in non_skip:
        local = kind_to_local.get(entity.kind)
        buckets.setdefault(local, []).append(entity)

    ordered_sections = [
        (local_name, buckets.get(local_name, []))
        for local_name in local_order
    ]
    return ordered_sections, buckets


# ─── Section emission ─────────────────────────────────────────────────────────


def _emit_empty_section(parts: list[str], local_name: str) -> None:
    label = local_name.replace("_", " ").title()
    parts.append(f"-- --- {label} ---")
    parts.append(f"local {local_name} = {{}}")
    parts.append("")


def _emit_section(
    parts: list[str], local_name: str, entities: list[FMLEntity], floor: Floor
) -> None:
    label = local_name.replace("_", " ").title()
    parts.append(f"-- --- {label} ---")
    parts.append(f"local {local_name} = {{}}")
    for entity in entities:
        parts.append("")
        _emit_entity_table(parts, local_name, entity, floor)
    _emit_section_triggers(parts, local_name, entities, floor)
    parts.append("")


def _emit_section_triggers(
    parts: list[str], local_name: str, entities: list[FMLEntity], floor: Floor
) -> None:
    for entity in entities:
        _, merged_triggers = _resolve_inherited_properties(entity, floor)
        if merged_triggers:
            for trigger in merged_triggers:
                _emit_trigger_attachment(parts, local_name, entity.id, trigger)
        for sub in entity.subentities:
            for trigger in sub.triggers:
                parts.append(
                    f"-- TODO: wire sub-entity trigger {entity.id}/{sub.id}.{trigger.name}"
                )


# ─── Entity table ─────────────────────────────────────────────────────────────


def _emit_entity_table(
    parts: list[str], local_name: str, entity: FMLEntity, floor: Floor
) -> None:
    # Resolve kind-chain inherited properties before emitting.
    # Returns the full flat property set with instance values winning.
    merged_props, _triggers_unused = _resolve_inherited_properties(entity, floor)

    parts.append(f"{local_name}.{entity.id} = {{")
    parts.append(f"    id = {_lua_string(entity.id)},")
    parts.append(f"    kind = {_lua_string(entity.kind)},")
    parts.append(f"    name = {_lua_string(entity.name)},")

    # exits only for rooms, inlined at top level — use merged exits.
    exits = merged_props.get("exits") or merged_props.get("exit")
    if exits and isinstance(exits, dict):
        parts.append(f"    exits = {_lua_value(exits)},")

    if entity.prose:
        if isinstance(entity.prose, ProseValue):
            parts.append(f"    prose = {_compile_prose(entity.prose, 'prose')},")
        elif isinstance(entity.prose, LuauCode):
            # Round-tripped from lua_reader — emit the function literal verbatim.
            parts.append(f"    prose = {entity.prose.source},")
        else:
            parts.append(f"    prose = {_lua_string(entity.prose)},")

    if entity.links:
        items = ", ".join(_lua_string(link) for link in entity.links)
        parts.append(f"    links = {{ {items} }},")

    # properties dict — emit merged (kind-chain + instance), skipping exits.
    remaining_props = {
        k: v for k, v in merged_props.items()
        if k not in ("exits", "exit")
    }
    if remaining_props:
        parts.append("    properties = {")
        for k, v in remaining_props.items():
            parts.append(f"        {_lua_key(k)} = {_lua_prop_value(v)},")
        parts.append("    },")
    else:
        parts.append("    properties = {},")

    # Sub-entities inline.
    if entity.subentities:
        parts.append("    subentities = {")
        for sub in entity.subentities:
            parts.append("        {")
            parts.append(f"            id = {_lua_string(sub.id)},")
            parts.append(f"            kind = {_lua_string(sub.kind)},")
            parts.append(f"            name = {_lua_string(sub.name)},")
            if sub.prose:
                if isinstance(sub.prose, ProseValue):
                    parts.append(f"            prose = {_compile_prose(sub.prose, 'prose')},")
                elif isinstance(sub.prose, LuauCode):
                    parts.append(f"            prose = {sub.prose.source},")
                else:
                    parts.append(f"            prose = {_lua_string(sub.prose)},")
            sub_props = dict(sub.properties)
            if sub_props:
                parts.append("            properties = {")
                for k, v in sub_props.items():
                    parts.append(f"                {_lua_key(k)} = {_lua_prop_value(v)},")
                parts.append("            },")
            else:
                parts.append("            properties = {},")
            parts.append("        },")
        parts.append("    },")

    parts.append("    triggers = {},")
    parts.append("}")


# ─── Trigger attachments ──────────────────────────────────────────────────────


def _trigger_slot_key(name: str) -> str:
    """Convert 'On Enter' / 'InsteadOf Attack' to 'on:Enter' / 'instead_of:Attack'."""
    parts = name.split(None, 1)
    if not parts:
        return "on:Unknown"
    stage_word = parts[0]
    event = parts[1] if len(parts) > 1 else ""
    stage_prefix = _STAGE_MAP.get(stage_word, stage_word.lower())
    return f"{stage_prefix}:{event}" if event else stage_prefix


def _emit_trigger_attachment(
    parts: list[str], local_name: str, entity_id: str, trigger: Trigger
) -> None:
    slot = _trigger_slot_key(trigger.name)
    parts.append(f'{local_name}.{entity_id}.triggers["{slot}"] = function(ctx)')
    guard = _compile_when_guard(trigger.when) if trigger.when else None
    if trigger.when and guard is None:
        # Complex guard that couldn't be compiled — emit a comment so authors
        # know the guard was dropped. The trigger will fire unconditionally.
        parts.append(f"    -- TODO: when guard not compiled: {trigger.when!r}")
    elif guard is not None:
        # guard is a Lua expression that is TRUE when the trigger SHOULD fire.
        # If it evaluates to false, return early (skip the trigger body).
        parts.append(f"    if not ({guard}) then return end")
    body = _trigger_body(trigger)
    if body:
        for line in body.splitlines():
            parts.append(f"    {line}" if line.strip() else "")
    parts.append("end")


# Pattern: flag(identifier)
_FLAG_RE = re.compile(r"^flag\(([A-Za-z_][A-Za-z0-9_]*)\)$")
# Pattern: not flag(identifier)
_NOT_FLAG_RE = re.compile(r"^not\s+flag\(([A-Za-z_][A-Za-z0-9_]*)\)$")


def _compile_when_guard(when: str) -> str | None:
    """Compile a FML 'when:' guard string to a Lua boolean expression.

    Handles simple flag(x) and not flag(x) forms via engine.get_world.
    Returns None for guards that can't be compiled (leaves trigger unconditional
    with a TODO comment — better to fire than to silently not fire).
    """
    s = when.strip()
    m = _NOT_FLAG_RE.match(s)
    if m:
        flag_name = m.group(1)
        return f'engine.get_world({_lua_string(flag_name)}) ~= "true"'
    m = _FLAG_RE.match(s)
    if m:
        flag_name = m.group(1)
        return f'engine.get_world({_lua_string(flag_name)}) == "true"'
    # Complex guard (e.g. time_in_room predicates) — emit a TODO comment
    # and leave the trigger unconditional for now.
    return None


def _trigger_body(trigger: Trigger) -> str:
    if trigger.script is not None:
        if trigger.script.language == "lua" or trigger.script.language == "luau":
            return trigger.script.source
        # Python script from pre-pivot FML
        return "-- NOTE: original trigger was Python; manual Luau translation required"
    if trigger.body:
        return _compile_body_items(trigger.body)
    return ""


def _compile_body_items(items: list[TriggerBodyItem]) -> str:
    return "\n".join(line for item in items for line in _compile_item(item))


def _compile_item(item: TriggerBodyItem) -> list[str]:
    if isinstance(item, PropertySet):
        segments = item.path.rsplit(".", 1)
        if len(segments) == 2:
            entity_path, prop = segments
            return [f"engine.set_property(ctx.{entity_path}.id, {_lua_string(prop)}, {_lua_value(item.value)})"]
        return [f"-- TODO: malformed property path {item.path!r}"]

    if isinstance(item, OutputLine):
        return [f"engine.output({_template_to_lua(item.template)})"]

    if isinstance(item, BareLink):
        # A bare Markdown link in a trigger body is a reference to another entity.
        # Compile to: look up the entity by its FML slug and fire its On Start trigger.
        # Target is typically "#slug" — strip the leading "#" if present.
        slug = item.target.lstrip("#")
        return [
            f"do",
            f"    local _tgt_id = _find_entity({_lua_string(slug)})",
            f"    if _tgt_id then engine.call_trigger(_tgt_id, \"on:Start\", ctx) end",
            f"end",
        ]

    if isinstance(item, ActionLine):
        return _compile_action_line(item)

    return [f"-- TODO: compile {item.kind!r} body item to Luau"]


# Patterns for Set/Clear [label](flag:name) action lines in trigger bodies.
_SET_FLAG_RE = re.compile(r"^Set\s+\[[^\]]*\]\(flag:([A-Za-z_][A-Za-z0-9_]*)\)\s*$")
_CLEAR_FLAG_RE = re.compile(r"^Clear\s+\[[^\]]*\]\(flag:([A-Za-z_][A-Za-z0-9_]*)\)\s*$")
# Apply [status] for N rounds to [entity](#entity_id) — emit a TODO for now.
_APPLY_STATUS_RE = re.compile(r"^Apply\s+")
# Loop through collection: name in collection — compiled by LoopThroughBlock; ActionLine fallback here.
_CLEAR_REACTIONS_RE = re.compile(r"^Clear reactions from")


def _compile_action_line(item: ActionLine) -> list[str]:
    """Compile a Form A action line to Lua.

    Handles:
    - Set [label](flag:name)  → engine.set_world("name", "true")
    - Clear [label](flag:name) → engine.set_world("name", "false")
    - Other lines            → TODO comment (does not crash)
    """
    raw = item.raw.strip()

    m = _SET_FLAG_RE.match(raw)
    if m:
        flag_name = m.group(1)
        return [f"engine.set_world({_lua_string(flag_name)}, \"true\")"]

    m = _CLEAR_FLAG_RE.match(raw)
    if m:
        flag_name = m.group(1)
        return [f"engine.set_world({_lua_string(flag_name)}, \"false\")"]

    return [f"-- TODO: action line {raw!r}"]


_TEMPLATE_TOKEN_RE = re.compile(r"\*([^*]+)\*|`([^`]+)`")


def _template_to_lua(template: str) -> str:
    """Compile output template to a Lua string expression.

    *path*   → ctx.path         (simple dotted path; assumed string, no tostring)
    `expr`   → tostring(expr)   (arbitrary Luau expression; always wrapped)
    """
    lua_parts: list[str] = []
    last_end = 0
    for m in _TEMPLATE_TOKEN_RE.finditer(template):
        literal = template[last_end:m.start()]
        if literal:
            lua_parts.append(_lua_string(literal))
        if m.group(1) is not None:
            # *path* → ctx.path
            lua_parts.append(f"ctx.{m.group(1)}")
        else:
            # `expr` → tostring(expr)
            lua_parts.append(f"tostring({m.group(2)})")
        last_end = m.end()
    trailing = template[last_end:]
    if trailing:
        lua_parts.append(_lua_string(trailing))
    if not lua_parts:
        return _lua_string("")
    if len(lua_parts) == 1:
        return lua_parts[0]
    return " .. ".join(lua_parts)


# ─── Return statement ─────────────────────────────────────────────────────────


def _emit_return(
    parts: list[str],
    section_locals: list[tuple[str, list[FMLEntity]]],
    other: list[FMLEntity],
) -> None:
    parts.append("return {")
    parts.append("    floor      = floor,")
    for local_name, _ in section_locals:
        parts.append(f"    {local_name:<10} = {local_name},")
    if other:
        parts.append("    other      = other,")
    parts.append("}")


# ─── Prose template compilation ──────────────────────────────────────────────

# Backtick segment tokenizer: splits a prose line into literal and `...` parts.
# Matches the outermost backtick pairs (non-greedy).
_PROSE_BACKTICK_RE = re.compile(r"`([^`]*)`")

# Old conditional syntax — hard error post-implementation (spec §8.1).
_OLD_IF_RE = re.compile(r"\[if\b|\[else\]|\[end if\]", re.IGNORECASE)

# Luau keywords / openers that classify a backtick segment as a statement,
# not a value expression.  We check these after ruling out simple expressions.
# Order matters: longer/more-specific first.
_STMT_OPENERS = (
    "if ",
    "elseif ",
    "else",
    "end",
    "for ",
    "while ",
    "repeat",
    "do",
    "local ",
    "return ",
    "break",
    "continue",
    "function ",
)


def _is_prose_statement(segment: str) -> bool:
    """Return True if a backtick segment is a Luau statement/partial chunk.

    Classification per PROSE.md §3.1:
    - If stripped contents match a known statement opener → statement.
    - Otherwise → expression (emit via tostring()).

    No Luau parser dependency — we use keyword prefix matching which is
    sufficient for the FML prose use cases.
    """
    s = segment.strip()
    if not s:
        return False
    for opener in _STMT_OPENERS:
        if s == opener.rstrip() or s.startswith(opener):
            return True
    return False


def _check_old_if_syntax(text: str, context: str) -> None:
    """Raise FmlSyntaxError if text contains old [if X in Y] conditional syntax.

    Per PROSE.md §8.1: after the prose-template implementation lands, this is
    a hard lower-time error.  Authors must use backtick-embedded Luau instead.
    """
    if _OLD_IF_RE.search(text):
        raise FmlSyntaxError(
            f"Old conditional prose syntax '[if ...]' found in {context}. "
            "Migrate to backtick-embedded Luau per docs/design/PROSE.md §8.2. "
            "Example: '[if X in Y]text[end if]' → "
            "'`if engine.entity_at(self.entity_id, \"X\") then` text `end`'"
        )


def _compile_prose(
    prose_val: "ProseValue",
    prop_key: str = "prose",
    source_path: str = "",
) -> str:
    """Lower a ProseValue to a Luau ``function(self, ctx) ... end`` string.

    Implements the lowering algorithm from docs/design/PROSE.md §4:

    1. Split into ``>`` lines (already done — ``prose_val.lines``).
    2. Tokenize each line into literal + backtick segments.
    3. Backtick segment: statement → emit verbatim; expression → ``tostring()``.
    4. Between adjacent ``>`` lines: emit ``s = s .. " "`` (line-join).
    5. Wrap in ``function(self, ctx) local s = "" ... return s end``.
    6. Prepend debug comment (spec §4.3).

    Returns the complete function literal as a string (no trailing newline).
    The caller embeds it at the property slot in the LFR table.
    """
    lines = prose_val.lines
    src_path = source_path or prose_val.source_path or "unknown"
    start_line = prose_val.source_line

    # Hard-error on old conditional syntax in every line.
    for line in lines:
        _check_old_if_syntax(
            line,
            context=f"{src_path}:{start_line} property {prop_key!r}",
        )

    parts: list[str] = []

    # Debug comment: inline with function opener so 'key = function(self, ctx)' is
    # a single token run in the LFR (required by tests and valid Lua).
    parts.append(
        f"function(self, ctx) -- source: {src_path}:{start_line} (prose template start)"
    )
    parts.append("  local s = \"\"")

    for line_idx, line in enumerate(lines):
        # Inter-line join: emit a space between consecutive non-empty > lines.
        # Empty lines are paragraph separators — emit a double-newline instead.
        if line_idx > 0:
            prev_empty = not lines[line_idx - 1].strip()
            curr_empty = not line.strip()
            if prev_empty or curr_empty:
                # Paragraph separator: an empty line between > blocks marks a
                # paragraph break.  Emit "\n\n" when transitioning from an
                # empty separator line back to content (prev_empty + curr not
                # empty).  Skip when the current line is itself empty.
                if prev_empty and not curr_empty:
                    parts.append("  s = s .. \"\\n\\n\" -- paragraph break")
            else:
                parts.append("  s = s .. \" \" -- line join")

        # Skip empty separator lines entirely (no code emitted for them).
        if not line.strip():
            continue

        line_num = start_line + line_idx + 1  # +1 for the property header
        stripped = line.rstrip()

        # Tokenize into alternating literals and backtick segments.
        cursor = 0
        for m in _PROSE_BACKTICK_RE.finditer(stripped):
            # Literal before this backtick segment.
            literal = stripped[cursor : m.start()]
            if literal:
                escaped = literal.replace("\\", "\\\\").replace('"', '\\"')
                parts.append(
                    f"  s = s .. \"{escaped}\" -- source: {src_path}:{line_num}"
                )

            segment = m.group(1)
            if _is_prose_statement(segment):
                # Statement/partial chunk — emit verbatim.
                parts.append(
                    f"  {segment} -- source: {src_path}:{line_num}"
                )
            else:
                # Expression — wrap in tostring().
                parts.append(
                    f"  s = s .. tostring({segment}) -- source: {src_path}:{line_num}"
                )

            cursor = m.end()

        # Trailing literal after the last backtick (or the whole line if no backticks).
        trailing = stripped[cursor:]
        if trailing:
            escaped = trailing.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(
                f"  s = s .. \"{escaped}\" -- source: {src_path}:{line_num}"
            )

    parts.append("  return s")
    parts.append("end")
    return "\n".join(parts)


# ─── Lua value serialisation ──────────────────────────────────────────────────


def _lua_string(s: str) -> str:
    if "\n" in s and "]]" not in s and not s.endswith("]"):
        return f"[[{s}]]"
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s}"'


def _lua_key(k: str) -> str:
    if _LUA_IDENT_RE.fullmatch(k):
        return k
    return f'["{k}"]'


def _lua_value(v: Any) -> str:
    """Serialize an arbitrary Python value to a Lua literal."""
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, LuauCode):
        return v.source
    if isinstance(v, ProseValue):
        return _compile_prose(v)
    if isinstance(v, str):
        return _lua_string(v)
    if isinstance(v, list):
        if not v:
            return "{}"
        items = ", ".join(_lua_value(i) for i in v)
        return f"{{ {items} }}"
    if isinstance(v, dict):
        if not v:
            return "{}"
        pairs = ", ".join(f"{_lua_key(k)} = {_lua_value(val)}" for k, val in v.items())
        return f"{{ {pairs} }}"
    if isinstance(v, Predicate):
        return _lua_string(v.model_dump_json())
    return _lua_string(str(v))


def _lua_prop_value(v: Any) -> str:
    """Like _lua_value but handles Predicate, LuauCode, and ProseValue types.

    - ProseValue: compile to Luau function via _compile_prose().
    - LuauCode: emit the source verbatim (no quoting) so live Luau code
      lands at the property slot in the LFR file.
    - Predicate: serialise as a JSON string (existing behaviour).
    - All other types: delegate to _lua_value.
    """
    if isinstance(v, ProseValue):
        return _compile_prose(v)
    if isinstance(v, LuauCode):
        return v.source
    if isinstance(v, Predicate):
        return _lua_string(v.model_dump_json())
    return _lua_value(v)
