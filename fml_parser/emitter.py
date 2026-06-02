"""LFR → FML reverse emitter.

Emits canonical FML text from a `Floor` model. The output is deterministic:
the same `Floor` always produces byte-identical FML.

See `docs/design/PARSER.md` § 6 and `docs/design/LFR.md` § Reverse emission algorithm
for the canonical ordering rules.
"""

from __future__ import annotations

from typing import Any

from .dice_value import LuauCode, ProseValue, is_dice_expr
from .models import FMLEntity, Floor, Script, Trigger
from .trigger_body_parser import format_trigger_body

# Fallback maps — used ONLY when a Floor carries no stdlib-resolved mappings
# (i.e. parsed in-memory with no source_path, or with a stdlib that omits
# `## Section to Kind` / `## Subsection to Kind` blocks). The authoritative
# source for these mappings is the stdlib FML itself; see stdlib/dnd5e/index.md.
_FALLBACK_KIND_TO_SECTION = [
    ("npc", "People"),
    ("item", "Items"),
    ("room", "Rooms"),
    ("encounter", "Encounters"),
    ("quest", "Quests"),
    ("wandering", "Wandering"),
    ("adjudicator_note", "Adjudicator Notes"),
    ("spell", "Spells"),
]

_FALLBACK_SUBKIND_TO_H4 = [
    ("trait", "Traits"),
    ("action", "Actions"),
    ("reaction", "Reactions"),
    ("legendary_action", "Legendary Actions"),
    ("spell", "Spells"),
    ("wandering_entry", "Entries"),
]


def _reverse_mappings(
    mappings: list[tuple[str, str]],
    fallback: list[tuple[str, str]],
) -> tuple[dict[str, str], list[str]]:
    """Invert a floor's `[(heading, kind), ...]` list into `kind → heading`
    plus the canonical heading-emission order.

    If the floor's mappings are empty, fall back to the hardcoded list above
    (kind → heading order). Insertion order in the returned dict is the
    section-emission order; for the kind→heading map, the FIRST heading per
    kind wins (matches stdlib declaration order).
    """
    pairs = (
        [(kind, heading) for heading, kind in mappings]
        if mappings
        else list(fallback)
    )
    kind_to_heading: dict[str, str] = {}
    heading_order: list[str] = []
    for kind, heading in pairs:
        if heading not in heading_order:
            heading_order.append(heading)
        if kind not in kind_to_heading:
            kind_to_heading[kind] = heading
    return kind_to_heading, heading_order


def emit_fml(floor: Floor, *, consolidated: bool = False) -> str:
    """Emit canonical FML text from a `Floor` model.

    Default mode emits the **entry-point view**: only the entities declared
    in the entry-point document (see `Floor.entry_point_entity_ids`), with
    the original import links preserved at the top. Imported entities are
    skipped — they live in their own source files.

    `consolidated=True` emits the **fully-flattened view**: every entity in
    the in-memory model (entry-point + everything pulled in from imports),
    with no import links — exactly the picture the runtime sees after the
    parser has finished resolving and merging. Round-trippable as standalone
    FML, but loses the per-source-file attribution. Useful for debugging the
    consolidated model (`tower consolidate FILE`).
    """
    parts: list[str] = []

    # H1
    parts.append(f"# {floor.name}")
    parts.append("")

    # Resolve section ordering from the stdlib-derived mappings.
    kind_to_section, section_order = _reverse_mappings(
        floor.section_mappings, _FALLBACK_KIND_TO_SECTION
    )

    if consolidated:
        # === Inline stdlib expansion at the import position ===
        # Source: `# Floor\n[stdlib](stdlib)\n- floor properties...`
        # Consolidated re-emits stdlib content here (right after H1) so a
        # reader sees the stdlib pasted in at the import point, not dumped
        # at the end of the document.
        #
        # Order WITHIN the expansion:
        #   0. Floor-level property list — emitted FIRST, before any H2 section,
        #      so `_extract_top_metadata` (which stops at the first H2) can read
        #      it when the consolidated file is re-parsed by `tower lower`.
        #      Without this, start_location and other floor properties are
        #      buried inside section content and never seen by the parser.
        #   1. Section/Subsection/Reserved-key mappings (define the H2/H4
        #      vocabulary so the parser knows how to bucket subsequent
        #      entities when re-parsing).
        #   2. Stdlib entities grouped by section (verbs, kinds, spells,
        #      monsters, items, features — whatever the stdlib brought in).
        #
        # Partition: entity is "from stdlib" iff its id is NOT in the
        # entry_point_entity_ids set. With one top-level import (`stdlib`)
        # this cleanly separates inlined vs entry-point content. Multi-import
        # provenance would need richer Floor metadata (v1 concern).
        entry_point_ids = set(floor.entry_point_entity_ids or set())
        stdlib_ids = set(floor.entities.keys()) - entry_point_ids

        # Floor-level property list (must precede first H2 for parser visibility).
        if floor.properties:
            parts.append(_emit_property_list(floor.properties))
            parts.append("")

        if floor.section_mappings:
            parts.append("## Section to Kind")
            parts.append("")
            for heading, kind in floor.section_mappings:
                parts.append(f"- {heading}: {kind}")
            parts.append("")
        if floor.subsection_mappings:
            parts.append("## Subsection to Kind")
            parts.append("")
            for heading, kind in floor.subsection_mappings:
                parts.append(f"- {heading}: {kind}")
            parts.append("")
        if floor.reserved_property_keys:
            parts.append("## Reserved Property Keys")
            parts.append("")
            for key, desc in floor.reserved_property_keys:
                parts.append(f"- {key}: {desc}")
            parts.append("")

        stdlib_by_section = _group_by_section(
            floor,
            kind_to_section=kind_to_section,
            restrict_to_ids=stdlib_ids,
        )
        for section_name in section_order:
            entities = stdlib_by_section.get(section_name, [])
            if not entities:
                continue
            parts.append(f"## {section_name}")
            parts.append("")
            for entity in entities:
                parts.append(_emit_entity(entity, floor, heading_level=3))
                parts.append("")
        stdlib_other = stdlib_by_section.get(None, [])
        if stdlib_other:
            parts.append("## Other")
            parts.append("")
            for entity in stdlib_other:
                parts.append(_emit_entity(entity, floor, heading_level=3))
                parts.append("")
    else:
        # Entry-point view: emit the original import links, runtime resolves them.
        for imp in floor.imports:
            parts.append(f"[stdlib]({imp})")
        if floor.imports:
            parts.append("")

    # === Entry-point content (floor's own contribution) ===

    # Floor-level property list (non-consolidated path only; consolidated path
    # emits it earlier, before the first H2 section, so the parser sees it).
    if not consolidated and floor.properties:
        parts.append(_emit_property_list(floor.properties))
        parts.append("")

    # Floor-level intro prose
    if floor.prose:
        parts.append(_emit_prose(floor.prose))
        parts.append("")

    # Group entry-point entities by section in canonical order
    entry_point_restrict = (
        set(floor.entry_point_entity_ids or set()) if consolidated else None
    )
    by_section = _group_by_section(
        floor,
        kind_to_section=kind_to_section,
        consolidated=consolidated,
        restrict_to_ids=entry_point_restrict,
    )
    for section_name in section_order:
        entities = by_section.get(section_name, [])
        if not entities:
            continue
        parts.append(f"## {section_name}")
        parts.append("")
        for entity in entities:
            parts.append(_emit_entity(entity, floor, heading_level=3))
            parts.append("")

    # Any entities whose kind isn't in the canonical section map → trailing
    # "## Other" section, preserved for round-trip.
    other = by_section.get(None, [])
    if other:
        parts.append("## Other")
        parts.append("")
        for entity in other:
            parts.append(_emit_entity(entity, floor, heading_level=3))
            parts.append("")

    # Strip trailing blanks, ensure single trailing newline
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts) + "\n"


# ─── Grouping ─────────────────────────────────────────────────────────────────


def _group_by_section(
    floor: Floor,
    *,
    kind_to_section: dict[str, str],
    consolidated: bool = False,
    restrict_to_ids: set[str] | None = None,
) -> dict[str | None, list[FMLEntity]]:
    """Bucket top-level entities by their canonical section heading.

    `kind_to_section` is the reverse map derived from `floor.section_mappings`
    (or the hardcoded fallback when those are empty). Insertion order within
    each section is preserved from `floor.entities`. Entities whose kind
    doesn't have a canonical section land under None (rendered as `## Other`
    by the emitter).

    Filtering:
    - `restrict_to_ids` (if provided) — only emit entities with these ids;
      overrides `consolidated`. Used by consolidated emission to partition
      stdlib vs entry-point.
    - `consolidated=False` (default), no `restrict_to_ids`: if
      `floor.entry_point_entity_ids` is set, only those entities are emitted;
      imported entities are skipped (they live in their own files).
    - `consolidated=True`, no `restrict_to_ids`: every entity is emitted
      regardless of source.
    """
    out: dict[str | None, list[FMLEntity]] = {}
    if restrict_to_ids is not None:
        allowed = restrict_to_ids
    else:
        allowed = None if consolidated else floor.entry_point_entity_ids
    for entity_id, entity in floor.entities.items():
        if allowed is not None and entity_id not in allowed:
            continue
        section = kind_to_section.get(entity.kind)
        out.setdefault(section, []).append(entity)
    return out


# ─── Entity emission (recursive for sub-entities) ─────────────────────────────


def _emit_entity(entity: FMLEntity, floor: Floor, heading_level: int) -> str:
    """Emit one entity, including its property list, prose, scripts, triggers,
    and sub-entities (recursively).
    """
    hashes = "#" * heading_level
    parts: list[str] = [f"{hashes} {entity.name}", ""]

    if entity.properties:
        parts.append(_emit_property_list(entity.properties))
        parts.append("")

    if entity.prose:
        parts.append(_emit_prose(entity.prose))
        parts.append("")

    # Bare scripts (not under any trigger) — emit as bare fenced code blocks
    for script in entity.scripts:
        parts.append(_emit_script(script))
        parts.append("")

    # Sub-entities, grouped under H4 subsections by stdlib-mapped subkind
    if entity.subentities and heading_level == 3:
        subkind_to_h4, h4_order = _reverse_mappings(
            floor.subsection_mappings, _FALLBACK_SUBKIND_TO_H4
        )
        grouped = _group_subs_by_h4(entity.subentities, subkind_to_h4)
        for h4_label in h4_order:
            subs = grouped.get(h4_label, [])
            if not subs:
                continue
            parts.append(f"#### {h4_label}")
            parts.append("")
            for sub in subs:
                parts.append(_emit_entity(sub, floor, heading_level=5))
                parts.append("")
        # Sub-entities without a canonical H4 → no grouping heading
        ungrouped = grouped.get(None, [])
        for sub in ungrouped:
            parts.append(_emit_entity(sub, floor, heading_level=5))
            parts.append("")
    elif entity.subentities:
        # Sub-entities under H5 parents (no further nesting in v0): just emit
        # them directly at H7+, which CommonMark doesn't support cleanly.
        # Defer this case to v1; warn the author by emitting an HTML comment.
        parts.append(
            "<!-- WARNING: sub-entities nested deeper than H5 are not emitted in v0 -->"
        )

    # H3 triggers: grouped under #### Triggers (new canonical format)
    # H5 triggers: emitted directly as H6 (old format — H4 inside H5 body
    #              terminates the H5 slice, so #### Triggers can't nest there)
    if entity.triggers and heading_level == 3:
        parts.append("#### Triggers")
        parts.append("")
        for trigger in entity.triggers:
            parts.append(_emit_trigger_new_style(trigger))
            parts.append("")
    elif entity.triggers:
        # H5 sub-entity triggers: old-style H6 directly (backward compat)
        for trigger in entity.triggers:
            parts.append(_emit_trigger_legacy(trigger))
            parts.append("")

    # Strip trailing blanks
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


def _group_subs_by_h4(
    subs: list[FMLEntity], subkind_to_h4: dict[str, str]
) -> dict[str | None, list[FMLEntity]]:
    out: dict[str | None, list[FMLEntity]] = {}
    for sub in subs:
        h4_label = subkind_to_h4.get(sub.kind)
        out.setdefault(h4_label, []).append(sub)
    return out


# ─── Triggers and scripts ─────────────────────────────────────────────────────


def _emit_trigger_new_style(trigger: Trigger) -> str:
    """Emit a trigger under a ``#### Triggers`` H4 (v2 canonical format).

    Name is emitted verbatim (PascalCase event name, or humanized slug for
    backward-compat triggers that were stored with slugified names).

    Body:
    - ``- when: <pred>`` if trigger has a when guard
    - Form A body items (bare links, action lines, control flow)
    - Python code block for legacy ``trigger.script`` (with comment)
    """
    name = _display_trigger_name(trigger.name)
    parts = [f"###### {name}", ""]

    body_lines = format_trigger_body(trigger.when, trigger.body)
    if body_lines:
        parts.extend(body_lines)
        # Ensure body block ends cleanly before any script
        if body_lines[-1] != "":
            parts.append("")

    if trigger.script is not None:
        body = trigger.script.source.rstrip("\n")
        lang = trigger.script.language
        parts.append(f"```{lang}\n{body}\n```")

    # Strip trailing blank lines
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


def _emit_trigger_legacy(trigger: Trigger) -> str:
    """Emit a trigger in the old H6-direct format (H5 sub-entity backward compat).

    This format is the original: H6 heading directly under the entity, no
    ``#### Triggers`` grouping.  Used for H5 sub-entity triggers only.
    """
    name = _display_trigger_name(trigger.name)
    if trigger.script is not None:
        body = trigger.script.source.rstrip("\n")
        lang = trigger.script.language
        return f"###### {name}\n\n```{lang}\n{body}\n```"
    # Trigger with no script and no body — emit name only (shouldn't happen in
    # practice for old-style triggers, but handle gracefully)
    return f"###### {name}"


def _emit_script(script: Script) -> str:
    body = script.source.rstrip("\n")
    lang = script.language
    return f"```{lang}\n{body}\n```"


def _display_trigger_name(name: str) -> str:
    """Return the display form of a trigger name.

    - PascalCase names (new-style, e.g. ``OnEnter``) → returned verbatim.
    - Slug names (old-style, e.g. ``on_damaged``) → humanized (``On Damaged``).

    Heuristic: if the name contains ``_`` it was slugified; otherwise verbatim.
    """
    if "_" in name:
        return " ".join(word.capitalize() for word in name.split("_"))
    return name


# ─── Property list emission ───────────────────────────────────────────────────


def _emit_property_list(props: dict[str, Any], indent: int = 0) -> str:
    """Emit a property dict as `- key: value` bullet lines.

    Nested dicts emit as sub-bullets:
        - exits:
          - north: target
          - east: other
    """
    lines: list[str] = []
    prefix = "  " * indent
    for key, value in props.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}- {key}:")
            lines.append(_emit_property_list(value, indent=indent + 1))
        else:
            lines.append(f"{prefix}- {key}: {_emit_scalar(value)}")
    return "\n".join(lines)


def _emit_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, LuauCode):
        # Emit as a backtick-wrapped Luau code fragment so the parser
        # reconstructs it as LuauCode on re-parse.
        return f"`{value.source}`"
    if isinstance(value, str):
        # Quote if the value contains characters that would confuse the parser
        # (specifically: leading/trailing whitespace, or starts with a digit but
        # isn't actually numeric — rare). Keep most strings unquoted for
        # readability.
        if value != value.strip():
            return f'"{value}"'
        # Quote fraction-like values (e.g. "1/4") so they survive round-trip
        # as strings rather than being mis-parsed.
        if "/" in value and any(c.isdigit() for c in value):
            return f'"{value}"'
        # Quote dice-expression strings so they survive round-trip as strings
        # and aren't re-evaluated as dice on the next parse pass.
        if is_dice_expr(value):
            return f'"{value}"'
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        # Bracket-list syntax: `[a, b, c]`. The parser recognizes this as a
        # list; bare comma-joined strings keep string semantics. Items go
        # through _emit_scalar individually so quoted/numeric items survive.
        return "[" + ", ".join(_emit_scalar(v) for v in value) + "]"
    return str(value)


# ─── Prose emission ───────────────────────────────────────────────────────────


def _emit_prose(prose: Any) -> str:
    """Wrap each paragraph in a `>` blockquote, preserving line breaks within
    a paragraph as line breaks within the blockquote.

    Accepts either a plain ``str``, a ``ProseValue``, or a ``LuauCode``.
    - ``ProseValue``: ``> `` prefix already present in ``__str__``; emit directly.
    - ``LuauCode``: opaque round-tripped function literal; emit as a Luau
      code fence (not re-convertible to FML prose form).
    - Plain ``str``: wrap each paragraph in ``> `` blockquote lines.
    """
    if isinstance(prose, ProseValue):
        # Reconstruct FML blockquote form: prefix each non-empty line with "> ",
        # use ">" alone for empty lines (paragraph separators).
        lines = prose.lines
        fml_lines: list[str] = []
        for i, line in enumerate(lines):
            if line:
                fml_lines.append(f"> {line}")
            else:
                # Empty line = paragraph separator — emit blank line between blockquotes.
                # Don't emit a bare ">" for paragraph separators; FML uses blank lines.
                if fml_lines:
                    fml_lines.append("")
        return "\n".join(fml_lines)
    if isinstance(prose, LuauCode):
        # Round-tripped from LFR — emit as a Luau code block (tower unlower
        # cannot reconstruct the original FML template from compiled Luau).
        return f"```luau\n{prose.source}\n```"
    prose_str = str(prose) if not isinstance(prose, str) else prose
    paragraphs = [p for p in prose_str.split("\n\n") if p.strip()]
    blocks: list[str] = []
    for p in paragraphs:
        lines = p.splitlines()
        block = "\n".join(f"> {line}" if line else ">" for line in lines)
        blocks.append(block)
    return "\n\n".join(blocks)
