"""FML → LFR parser.

This module implements the parser specified in `docs/design/PARSER.md`. v0.3
scope covers the document structure end-to-end:

- single H1 (document title)
- top-level imports (link-shaped, expanded inline with include-once dedup)
- top-level property list and intro prose
- H2 sections (convention only; preserved for round-trip)
- H3 entities with property lists, prose, scripts, triggers, and H5 sub-entities
- H4 organizational subsections (convention; preserved)
- H5 sub-entities (full FMLEntity recursion)
- H6 triggers (named event handlers with code-block bodies)

The parser is deterministic. Given the same FML text + source path + the
same import-resolution outcome, it produces a byte-identical `Floor` model.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

from .dice_value import ProseValue, is_dice_expr, parse_dice_value
from .errors import FmlImportError, FmlSyntaxError
from .models import EntityId, FMLEntity, Floor, Script, Trigger, UnderstandDirective
from .slugify import slugify
from .trigger_body_parser import parse_trigger_body

_md = MarkdownIt("commonmark", {"breaks": False, "html": True})


# ─── Import context (include-once across recursive parses) ────────────────────


@dataclass
class _KindMaps:
    """Composite kind-resolution maps accumulated from imported stdlib documents.

    Stdlib documents declare `## Section to Kind` and `## Subsection to Kind`
    blocks; the bullet lists below them populate these maps. The composite
    overrides the parser's hardcoded fallback at resolution time.

    Each dict is keyed by the *canonical heading text* (preserving the casing
    the stdlib author wrote — `People`, not `people`). Lookup at resolution
    time normalizes both sides to lowercase. Insertion order is the canonical
    emission order (and is preserved across `update` for re-declarations).
    """

    section_to_kind: dict[str, str] = field(default_factory=dict)
    subsection_to_subkind: dict[str, str] = field(default_factory=dict)

    def merge(self, other: _KindMaps) -> None:
        _ordered_merge(self.section_to_kind, other.section_to_kind)
        _ordered_merge(self.subsection_to_subkind, other.subsection_to_subkind)


def _ordered_merge(target: dict[str, str], updates: dict[str, str]) -> None:
    """Merge `updates` into `target` with case-insensitive key matching.

    Existing entries (matched case-insensitively on the heading text) have
    their VALUE updated in place — the original key (and thus its insertion
    position) is preserved, so a higher-layer override doesn't reorder the
    section. New entries append at the end.
    """
    lowered = {k.lower(): k for k in target}
    for new_key, new_val in updates.items():
        existing_key = lowered.get(new_key.lower())
        if existing_key is not None:
            target[existing_key] = new_val
        else:
            target[new_key] = new_val
            lowered[new_key.lower()] = new_key


@dataclass
class _ImportContext:
    """Tracks the import dictionary across a parser invocation.

    `seen` holds canonicalized paths whose entities have already been merged
    into the entry-point floor. `in_progress` holds paths currently being
    parsed (used for cycle detection). `kind_maps` accumulates the composite
    section→kind and subsection→subkind mappings from imported stdlib docs.
    `reserved_keys` accumulates the `## Reserved Property Keys` declarations.
    """

    seen: set[Path] = field(default_factory=set)
    in_progress: set[Path] = field(default_factory=set)
    kind_maps: _KindMaps = field(default_factory=_KindMaps)
    reserved_keys: dict[str, str] = field(default_factory=dict)

_KV_RE = re.compile(r"^(?P<key>[a-z_][a-z0-9_-]*)\s*:\s*(?P<value>.*)$")
_UNDERSTAND_RE = re.compile(
    r'^\*\*[Uu]nderstand\*\*\s+"([^"]+)"\s+as\s+"([^"]+)"\s*$'
)
# Sentinel returned by _parse_list_item_as_property when a list item is an
# Understand directive (not a key:value pair but also not prose — it should
# not cause the surrounding property list to be abandoned).
_UNDERSTAND_ITEM: tuple[str, None] = ("__understand__", None)

# Prose prefix: property values starting with "> " are prose-typed.
_PROSE_VALUE_PREFIX = "> "

# Old conditional-prose syntax — hard error at lower time (PROSE.md §8.1).
_OLD_IF_RE = re.compile(r"\[if\b|\[else\]|\[end if\]", re.IGNORECASE)

# Kind-map keys are section/subsection NAMES; they may contain spaces and use
# either lowercase or title case (`- People: npc` or `- people: npc`). The
# original-case form is preserved for emission; the lookup form is lowercased.
# Values are bare identifiers (a single kind name).
_KIND_MAP_RE = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9 _]*[A-Za-z0-9])\s*:\s*(?P<value>[a-z_][a-z0-9_]*)$"
)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# ─── Public entry point ───────────────────────────────────────────────────────


def parse_fml(text: str, source_path: Path | None = None) -> Floor:
    """Parse FML text into a `Floor` model.

    Resolves all imports recursively (see `docs/design/FML.md` §6 and
    `docs/design/PARSER.md` §5.7): each canonicalized import path is processed
    exactly once across the invocation; re-references are no-ops; cycles
    raise `FmlImportError`. Same-id entities across the import tree merge
    per `docs/design/FML.md` §6.2 via `Floor.__iadd__`.
    """
    ctx = _ImportContext()
    if source_path is not None:
        ctx.in_progress.add(Path(source_path).resolve())
    return _parse_with_imports(text, source_path, ctx, is_entry_point=True)


def _parse_with_imports(
    text: str,
    source_path: Path | None,
    ctx: _ImportContext,
    *,
    is_entry_point: bool,
) -> Floor:
    """Internal recursive parser. Shares the import context across recursion."""
    tokens = _md.parse(text)
    h1_text = _find_single_h1(tokens, source_path)

    floor_properties, floor_prose, imports = _extract_top_metadata(tokens)

    floor = Floor(
        name=h1_text,
        properties=floor_properties if is_entry_point else {},
        prose=floor_prose if is_entry_point else "",
        imports=imports if is_entry_point else [],
        fml_source_hash=(
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            if is_entry_point
            else None
        ),
    )

    # Phase 1: process imports FIRST so the composite kind-resolution maps
    # are populated before we resolve this document's own entity kinds.
    # When source_path is None (in-memory parse), imports are recorded but
    # not loaded — safe default. Same with cycle detection: missing files
    # raise from inside `_load_import`.
    if imports and source_path is not None:
        base_dir = source_path.parent
        for imp_link in imports:
            _load_import(imp_link, base_dir, floor, ctx)

    # Phase 2: extract any `## Section to Kind` / `## Subsection to Kind`
    # blocks from THIS document into the composite kind maps. (Stdlib
    # documents declare these; entry-point floors typically don't.) Also
    # accumulate `## Reserved Property Keys` declarations.
    local_maps = _extract_local_kind_maps(tokens)
    ctx.kind_maps.merge(local_maps)
    for key, desc in _extract_local_reserved_keys(tokens).items():
        # First declaration wins on the description text (stdlib's intent),
        # but later layers may add new keys. Don't override existing
        # descriptions silently — they're authored, not user-provided.
        ctx.reserved_keys.setdefault(key, desc)

    # Phase 3: walk the H2 sections and extract H3 entities from THIS doc,
    # using the now-composite kind maps for resolution.  Also collect any
    # Understand directives from entity property lists.
    understand_from_entities: list[UnderstandDirective] = []
    for section_name, entity_blocks in _iter_sections_and_entities(tokens):
        if _is_kind_map_section(section_name):
            # The kind map H2 sections don't contain entities — already
            # consumed in Phase 2 above.
            continue
        for entity_name, entity_tokens in entity_blocks:
            entity = _parse_entity(
                entity_name,
                entity_tokens,
                default_kind=_resolve_section_kind(section_name, ctx.kind_maps),
                heading_level=3,
                kind_maps=ctx.kind_maps,
            )
            floor += entity
            understand_from_entities.extend(
                _collect_understand_directives(entity_tokens)
            )

    # Collect Understand directives from the floor-level property list.
    start = _index_after_h1(tokens)
    end = _next_heading_at_or_above(tokens, start, 2)
    floor_level_directives = _collect_understand_directives(tokens[start:end])

    # Prepend this doc's own directives; import-accumulated ones already sit
    # in floor.understand_directives from the _load_import calls in Phase 1.
    floor.understand_directives = (
        floor_level_directives + understand_from_entities + floor.understand_directives
    )

    # Snapshot entry-point's own entity ids — entities just added in Phase 3
    # are the entry-point's; imports merged in Phase 1 already populated the
    # floor.entities dict. So entry-point ids are the diff between current
    # entities and those that existed before Phase 3.
    if is_entry_point:
        # The simplest correct snapshot: post-import entity ids that came
        # from THIS document. Imports contribute their own entities; this
        # document's H3s contribute the rest.
        floor.entry_point_entity_ids = _collect_local_entity_ids(tokens)

        # Snapshot the resolved kind maps onto the floor. The emitter uses
        # these for grouping/ordering, so the floor is self-describing — the
        # stdlib content drives layout, not hardcoded constants in emitter.py.
        floor.section_mappings = list(ctx.kind_maps.section_to_kind.items())
        floor.subsection_mappings = list(ctx.kind_maps.subsection_to_subkind.items())
        floor.reserved_property_keys = list(ctx.reserved_keys.items())

        # Populate kind_chain on every entity now that all kind_definitions
        # from the import tree are present. This is what makes
        # `entity.is_a("actor")` work for downstream consumers.
        floor.populate_kind_chains()

    return floor


def _collect_local_entity_ids(tokens: list[Token]) -> set[EntityId]:
    """Walk THIS document's tokens and return the set of slugified entity ids
    declared directly in it (not from imports)."""
    out: set[EntityId] = set()
    for section_name, entity_blocks in _iter_sections_and_entities(tokens):
        if _is_kind_map_section(section_name):
            continue
        for entity_name, _ in entity_blocks:
            out.add(slugify(entity_name))
    return out


def _is_kind_map_section(section_name: str) -> bool:
    """A reserved H2 section name whose body is metadata (kind maps, reserved
    property keys, etc.) rather than a container of H3 entities."""
    norm = section_name.strip().lower()
    return norm in (
        "section to kind",
        "subsection to kind",
        "reserved property keys",
    )


def _extract_local_kind_maps(tokens: list[Token]) -> _KindMaps:
    """Pull `## Section to Kind` and `## Subsection to Kind` H2 sections and
    interpret their first bullet list as a kind-map.

    Kind-map bullets allow whitespace in keys (`adjudicator notes: adjudicator_note`),
    unlike regular property lists whose keys are strict identifiers.
    """
    maps = _KindMaps()
    for section_name, section_tokens in _iter_h2_section_token_ranges(tokens):
        norm = section_name.strip().lower()
        pairs = _extract_first_kind_map_list(section_tokens)
        if norm == "section to kind":
            for key, value in pairs.items():
                # Preserve the original casing of the heading so the emitter
                # has the canonical text to write (`## People`, not `## people`).
                maps.section_to_kind[key.strip()] = value
        elif norm == "subsection to kind":
            for key, value in pairs.items():
                maps.subsection_to_subkind[key.strip()] = value
    return maps


def _extract_local_reserved_keys(tokens: list[Token]) -> dict[str, str]:
    """Pull a `## Reserved Property Keys` H2 section and parse its bullet list.

    Each bullet is `- key_name: free-form description`. Keys are strict
    identifiers; descriptions are anything up to end-of-line. Returns
    `{key_name: description}` preserving declaration order (dict-insertion).
    """
    out: dict[str, str] = {}
    for section_name, section_tokens in _iter_h2_section_token_ranges(tokens):
        if section_name.strip().lower() != "reserved property keys":
            continue
        i = 0
        while i < len(section_tokens):
            if (
                section_tokens[i].type == "bullet_list_open"
                and section_tokens[i].level == 0
            ):
                for item_tokens in _list_items(section_tokens, i):
                    for tok in item_tokens:
                        if tok.type == "inline":
                            m = _KV_RE.match(tok.content.strip())
                            if m is not None:
                                out[m.group("key")] = m.group("value").strip()
                            break
                break
            i += 1
    return out


def _extract_first_kind_map_list(region: list[Token]) -> dict[str, str]:
    """Parse the first top-level bullet list as a kind-map (key may contain spaces)."""
    i = 0
    while i < len(region):
        if region[i].type == "bullet_list_open" and region[i].level == 0:
            return _parse_bullet_list_as_kind_map(region, i)
        i += 1
    return {}


def _parse_bullet_list_as_kind_map(
    region: list[Token], start: int
) -> dict[str, str]:
    """Parse a bullet_list as `<section name>: <kind>` pairs."""
    out: dict[str, str] = {}
    for item_tokens in _list_items(region, start):
        for tok in item_tokens:
            if tok.type == "inline":
                m = _KIND_MAP_RE.match(tok.content.strip())
                if m is not None:
                    out[m.group("key")] = m.group("value")
                break
    return out


def _normalize_section_key(key: str) -> str:
    """Section/subsection map keys are stored lowercased to match the
    case-insensitive lookup at resolution time."""
    return key.strip().lower().replace("_", " ")


def _iter_h2_section_token_ranges(
    tokens: list[Token],
) -> Iterator[tuple[str, list[Token]]]:
    """Yield (section_name, section_body_tokens) for each H2 in the document."""
    i = _first_h2_index(tokens)
    while i < len(tokens):
        if _heading_level(tokens[i]) == 2:
            section_name = _inline_text_after(tokens, i)
            section_end = _next_heading_at_or_above(tokens, i + 3, 2)
            yield section_name, tokens[i + 3 : section_end]
            i = section_end
        else:
            i += 1


def _resolve_section_kind(section_name: str, kind_maps: _KindMaps) -> str:
    """Stdlib-driven section → kind resolution.

    With no stdlib imported, every entity defaults to kind `entity` — the
    root of the kind hierarchy. Authors who want anything more specific
    must either import a stdlib that registers section bindings, or set
    `- kind: <name>` explicitly on the entity.

    Map keys are stored in their original casing for round-trip emission;
    lookup is case-insensitive on the heading text.
    """
    return _lookup_ci(kind_maps.section_to_kind, section_name) or "entity"


def _resolve_subsection_kind(h4_label: str | None, kind_maps: _KindMaps) -> str:
    if h4_label is None:
        return "entity"
    return _lookup_ci(kind_maps.subsection_to_subkind, h4_label) or "entity"


def _lookup_ci(d: dict[str, str], key: str) -> str | None:
    """Case-insensitive dict lookup that also tolerates `_` vs ` ` in keys."""
    norm = key.strip().lower().replace("_", " ")
    for k, v in d.items():
        if k.strip().lower().replace("_", " ") == norm:
            return v
    return None


def _load_import(
    link_target: str, base_dir: Path, floor: Floor, ctx: _ImportContext
) -> None:
    """Load one imported document, with include-once dedup + cycle detection.

    Skips if already seen; raises `FmlImportError` on cycle or missing file.
    On success, merges the imported document's entities into `floor`.

    Resolution order for `link_target`:
    1. **Named import** — if the target is a bare identifier (no slash, no `.`),
       it is looked up in `_NAMED_IMPORTS` and resolved against the project
       root (discovered by walking up from `base_dir` for a marker file).
    2. **Relative path** — otherwise resolved against `base_dir`.

    H1-entity promotion (the "single-entity file" pattern):
    If the imported document has no H3 entities of its own (they may come from
    the document's own sub-imports, but the document itself declares none as
    H3 headings), AND the document's H1 body (the region from after the H1 to
    the first H2 or end of document, after stripping import-link paragraphs)
    is non-empty, the H1 is treated as an entity declaration.  Its body is
    parsed with the same rules as an H3 entity body.  This supports the
    author convention of one file = one H1-top-entity, used throughout
    data/sample/ for modular entity files.
    """
    resolved = _resolve_import_path(link_target, base_dir)

    if resolved in ctx.seen:
        # Already processed — no-op (the entities are already merged via the
        # first inclusion). The link is preserved on `floor.imports` for
        # round-trip.
        return

    if resolved in ctx.in_progress:
        raise FmlImportError(
            f"import cycle detected at {resolved} "
            f"(in-progress: {[str(p) for p in ctx.in_progress]})"
        )

    if not resolved.exists():
        raise FmlImportError(
            f"imported file not found: {resolved} (from link {link_target!r})"
        )
    if not resolved.is_file():
        raise FmlImportError(
            f"imported path is not a file: {resolved} (from link {link_target!r})"
        )

    ctx.in_progress.add(resolved)
    try:
        sub_text = resolved.read_text(encoding="utf-8")
        sub_floor = _parse_with_imports(
            sub_text, resolved, ctx, is_entry_point=False
        )
        # Determine whether this import is from the stdlib (path contains
        # "stdlib/").  Stdlib-sourced entities are tracked for tree-shake
        # purposes — they are pruneable catalog content unless referenced.
        is_stdlib_import = "stdlib" in resolved.parts

        # Merge entities and understand_directives from the imported doc into
        # the parent floor. The imported doc's H1/intro prose/top-level
        # properties are metadata of the imported doc — discarded here.
        for entity in sub_floor.entities.values():
            floor += entity
            if is_stdlib_import:
                floor.stdlib_entity_ids.add(entity.id)
        floor.understand_directives.extend(sub_floor.understand_directives)

        # Propagate stdlib_entity_ids from sub_floor (they may have come from
        # transitive stdlib imports within the stdlib itself, e.g. dnd5e
        # importing core).
        if is_stdlib_import:
            floor.stdlib_entity_ids.update(sub_floor.stdlib_entity_ids)

        # H1-entity promotion: if the imported document declares no H3
        # entities of its own, try to synthesize an entity from the H1 body.
        # This handles the "one file = one H1-top-entity" convention used by
        # modular FML files (e.g. data/sample/people/skull_king.md).
        tokens = _md.parse(sub_text)
        h1_entity = _try_synthesize_h1_entity(tokens, sub_floor.name, floor, ctx)
        if h1_entity is not None:
            floor += h1_entity
            if is_stdlib_import:
                floor.stdlib_entity_ids.add(h1_entity.id)
    finally:
        ctx.in_progress.discard(resolved)
        ctx.seen.add(resolved)


# ─── H1-entity promotion (single-entity file convention) ─────────────────────


def _has_local_h3_entities(tokens: list[Token]) -> bool:
    """Return True if the document has at least one H3 entity declared directly
    (not from imports).  Import-link paragraphs don't count.
    """
    for section_name, entity_blocks in _iter_sections_and_entities(tokens):
        if _is_kind_map_section(section_name):
            continue
        for _name, _toks in entity_blocks:
            return True
    return False


def _h1_body_region(tokens: list[Token]) -> list[Token]:
    """Return the token slice between the H1 and the first H2 (or end of doc),
    with import-link paragraphs already stripped out.

    This region is the candidate body for a synthesized H1 entity.
    """
    start = _index_after_h1(tokens)
    end = _next_heading_at_or_above(tokens, start, 2)
    region = tokens[start:end]
    # Strip import-link paragraphs (they're not entity content).
    discarded: list[str] = []
    region = _consume_import_paragraphs(region, discarded)
    return region


def _region_has_entity_content(region: list[Token]) -> bool:
    """Return True if `region` has at least one token that constitutes entity
    content in FML: a property list (bullet list), a blockquote (entity prose),
    a child heading (H4/H5/H6), or a fenced code block.

    Bare paragraphs (ordinary plain-text paragraphs without the ``>`` blockquote
    prefix) are NOT considered entity content — they appear in library/stdlib
    description files but not in authored entity files.  This keeps the H1
    promotion from synthesizing spurious entities out of stdlib index files
    whose descriptions are written as plain prose.
    """
    for tok in region:
        if tok.type == "bullet_list_open":
            return True
        if tok.type == "blockquote_open":
            return True
        lvl = _heading_level(tok)
        if lvl is not None and lvl >= 4:
            return True
        if tok.type == "fence":
            return True
    return False


def _resolve_kind_by_attributes(
    properties: dict,
    floor: Floor,
    kind_maps: _KindMaps,
) -> str:
    """Attribute-heuristic kind resolver for H1-promoted entities.

    Resolution order:
    1. Explicit ``- kind: <name>`` property — always wins.
    2. Discriminating-attribute priority list — a hardcoded set of
       (kind, {key, ...}) pairs where each key is unique to that kind
       in the standard sample.  The first kind whose ANY key is present
       in the entity's properties wins.  Ordered from most-specific to
       least-specific; evaluated before the stdlib scan to avoid false
       positives from shared generic attributes like ``type``.
    3. Stdlib attribute scan — walk the floor's kind_definition entities;
       require at least ``_ATTR_MATCH_THRESHOLD`` overlapping keys so that
       single generic matches (e.g. ``type`` matching both npc and item) are
       suppressed.
    4. Fallback to ``"entity"``.
    """
    # 1. Explicit kind property.
    if "kind" in properties:
        return str(properties["kind"])

    entity_keys = set(properties.keys()) - {"kind"}

    # 2. Discriminating-attribute priority list.
    # Keys here are UNIQUE to their kind in standard FML usage — unlikely to
    # appear on entities of other kinds.  Authors who use these keys on an
    # entity of a different kind should set ``- kind:`` explicitly.
    _DISCRIMINATING_ATTRS: list[tuple[str, frozenset[str]]] = [
        # NPC: D&D 5e stat-block attributes that only creatures have.
        ("npc",       frozenset({"ac", "hp", "cr", "str", "dex", "con",
                                 "int", "wis", "cha"})),
        # Room: directional exits are exclusive to rooms in this system.
        ("room",      frozenset({"exits"})),
        # Encounter: combatants list is exclusive to encounters.
        ("encounter", frozenset({"combatants"})),
        # Quest: completion predicate is exclusive to quests.
        ("quest",     frozenset({"complete", "discovery"})),
        # Wandering table: distinguished by table_name.
        ("wandering", frozenset({"table_name"})),
        # Item: slot / value / weight are item-specific.
        ("item",      frozenset({"slot", "value", "weight", "portable"})),
        # Spell: school is spell-specific.
        ("spell",     frozenset({"school", "save_dc", "concentration"})),
    ]
    for kind_name, disc_keys in _DISCRIMINATING_ATTRS:
        if entity_keys & disc_keys:
            return kind_name

    # 3. Stdlib attribute scan with multi-key threshold.
    _ATTR_MATCH_THRESHOLD = 2  # require at least this many overlapping keys
    if entity_keys:
        kind_signatures: list[tuple[str, set[str]]] = []
        for ent in floor.entities.values():
            if ent.kind != "kind_definition":
                continue
            kind_name = ent.properties.get("name")
            attrs = ent.properties.get("attributes")
            if isinstance(kind_name, str) and isinstance(attrs, dict) and attrs:
                kind_signatures.append((kind_name, set(attrs.keys())))
        # Reversed so dnd5e kinds (registered later) are tried before core.
        for kind_name, attr_keys in reversed(kind_signatures):
            if len(entity_keys & attr_keys) >= _ATTR_MATCH_THRESHOLD:
                return kind_name

    return "entity"


def _try_synthesize_h1_entity(
    tokens: list[Token],
    h1_text: str,
    floor: Floor,
    ctx: _ImportContext,
) -> FMLEntity | None:
    """Attempt to synthesize an entity from the H1 of an imported document.

    Returns a ``FMLEntity`` if the document follows the "one file = one
    H1-top-entity" convention, otherwise None.

    Conditions for synthesis (ALL must hold):
    1. The document has no imports of its own.  Stdlib index files and other
       library/router files re-export things via imports — they are not
       single-entity files.  A file with sub-imports is treated as a
       collection, not an entity.
    2. The document has no H3 entities declared directly (not from sub-imports).
       Files that contain H3 entity blocks (e.g. the dnd5e stdlib's kind
       definitions) are collection files.
    3. The H1 body region (after stripping import paragraphs) has entity
       content: a bullet property list, a blockquote (entity prose), a child
       heading (H4/H5/H6), or a fenced code block.  Empty files and files
       with only plain prose paragraphs do not qualify.

    The synthesized entity's kind is resolved via ``_resolve_kind_by_attributes``:
    explicit ``- kind:`` wins, then a discriminating-attribute priority list,
    then a stdlib attribute overlap scan, then ``"entity"`` fallback.
    """
    # Gate 1: document must have no imports of its own.
    # Library/router files (stdlib/index.md, core/index.md, etc.) all have
    # sub-imports; leaf entity files (skull_king.md, the_sigil.md, etc.) don't.
    _, _, sub_imports = _extract_top_metadata(tokens)
    if sub_imports:
        return None

    # Gate 2: no local H3 entities — this is a leaf entity file.
    if _has_local_h3_entities(tokens):
        return None

    # Gate 3: body region must have actual entity content.
    body = _h1_body_region(tokens)
    if not _region_has_entity_content(body):
        return None

    # Parse the body as an entity body using the same machinery as H3 entities.
    # Use heading_level=3 so that H4/H5 children are handled correctly.
    default_kind = _resolve_section_kind("", ctx.kind_maps)  # no section → "entity"
    entity = _parse_entity(
        h1_text,
        body,
        default_kind=default_kind,
        heading_level=3,
        kind_maps=ctx.kind_maps,
    )

    # Re-resolve kind with the attribute heuristic (the stdlib is already
    # loaded into `floor` at this point because stdlib imports come first).
    better_kind = _resolve_kind_by_attributes(entity.properties, floor, ctx.kind_maps)
    if better_kind != entity.kind:
        entity = entity.model_copy(update={"kind": better_kind})

    return entity


# ─── Token helpers (small layer over markdown-it-py's flat stream) ────────────


def _heading_level(tok: Token) -> int | None:
    """Return the level for heading_open tokens; None for others."""
    if tok.type == "heading_open" and tok.tag.startswith("h"):
        try:
            return int(tok.tag[1:])
        except ValueError:
            return None
    return None


def _inline_text_after(tokens: list[Token], idx: int) -> str:
    """For a heading_open at tokens[idx], return the heading's inline text."""
    if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline":
        return tokens[idx + 1].content.strip()
    return ""


def _next_heading_at_or_above(tokens: list[Token], start: int, level: int) -> int:
    """Index of the next heading_open with tag-level <= `level` at or after `start`.

    Returns len(tokens) if none found.
    """
    i = start
    while i < len(tokens):
        lvl = _heading_level(tokens[i])
        if lvl is not None and lvl <= level:
            return i
        i += 1
    return len(tokens)


# ─── Top-level extraction ─────────────────────────────────────────────────────


def _find_single_h1(tokens: list[Token], source_path: Path | None) -> str:
    """Locate the single H1 heading and return its text."""
    h1_texts: list[str] = []
    for i, tok in enumerate(tokens):
        if _heading_level(tok) == 1:
            h1_texts.append(_inline_text_after(tokens, i))
    if not h1_texts:
        raise FmlSyntaxError(
            f"FML document {source_path or '<text>'} has no H1 heading"
        )
    if len(h1_texts) > 1:
        raise FmlSyntaxError(
            f"FML document {source_path or '<text>'} has multiple H1 headings: {h1_texts}"
        )
    return h1_texts[0]


def _extract_top_metadata(
    tokens: list[Token],
) -> tuple[dict[str, Any], str, list[str]]:
    """Extract the floor's leading property list, intro prose, and imports.

    Walks from after the H1 until the first H2 (or document end). Inside that
    window: collect a leading property list, collect blockquote prose, and
    collect import-shaped links (Markdown links that appear as standalone
    paragraphs with no surrounding prose).
    """
    properties: dict[str, Any] = {}
    prose_pieces: list[str] = []
    imports: list[str] = []

    start = _index_after_h1(tokens)
    end = _next_heading_at_or_above(tokens, start, 2)

    region = tokens[start:end]

    # Identify and consume import-link paragraphs (paragraph_open + inline
    # whose content is exactly one Markdown link + paragraph_close). These
    # come immediately after the H1, before any non-link content.
    region = _consume_import_paragraphs(region, imports)

    # Now extract the leading property list (the first bullet list whose items
    # all match `- key: value`) and any prose blockquotes.
    properties.update(_extract_first_property_list(region))
    prose_pvs = _collect_prose_blockquotes(region)
    prose_pieces.extend(pv.lines[0] if len(pv.lines) == 1 else " ".join(pv.lines)
                        for pv in prose_pvs)

    prose = "\n\n".join(prose_pieces)
    return properties, prose, imports


def _index_after_h1(tokens: list[Token]) -> int:
    """Return the index just past the H1's heading_close."""
    for i, tok in enumerate(tokens):
        if _heading_level(tok) == 1:
            # Skip heading_open, inline, heading_close — 3 tokens
            return i + 3
    return 0


def _consume_import_paragraphs(region: list[Token], out_imports: list[str]) -> list[Token]:
    """Strip leading "import-link" paragraphs from region and record their targets.

    An import paragraph is a paragraph whose entire inline content (after
    whitespace normalization) is a sequence of one or more Markdown links
    `[label](target)` separated by whitespace/newlines. Each link's target
    becomes an import. Multiple import paragraphs in a row are all consumed
    before a non-import paragraph stops the consumption window.
    """
    kept: list[Token] = []
    i = 0
    consuming = True
    while i < len(region):
        tok = region[i]
        if (
            consuming
            and tok.type == "paragraph_open"
            and i + 2 < len(region)
            and region[i + 1].type == "inline"
            and region[i + 2].type == "paragraph_close"
        ):
            content = region[i + 1].content.strip()
            link_targets = _parse_as_only_links(content)
            if link_targets:
                out_imports.extend(link_targets)
                i += 3
                continue
            # First non-import paragraph stops the consumption window.
            consuming = False
        kept.append(tok)
        i += 1
    return kept


def _parse_as_only_links(content: str) -> list[str] | None:
    """If `content` is one-or-more `[label](target)` separated by whitespace
    (with nothing else), return [target, ...]. Otherwise return None.
    """
    remainder = content
    targets: list[str] = []
    while remainder.strip():
        remainder = remainder.lstrip()
        m = _LINK_RE.match(remainder)
        if m is None:
            return None
        targets.append(m.group(2))
        remainder = remainder[m.end():]
    return targets if targets else None


# ─── Property list extraction ─────────────────────────────────────────────────


def _extract_first_property_list(region: list[Token]) -> dict[str, Any]:
    """Return key:value pairs from the first bullet list in `region`.

    A list qualifies as a property list only if every item matches `- key: value`.
    Otherwise treated as prose and skipped here.
    """
    # Locate the first top-level bullet_list_open in this region.
    i = 0
    while i < len(region):
        if region[i].type == "bullet_list_open" and region[i].level == 0:
            return _parse_bullet_list_as_props(region, i)
        i += 1
    return {}


def _collect_understand_directives(region: list[Token]) -> list[UnderstandDirective]:
    """Scan all top-level bullet lists in ``region`` for Understand directives.

    Returns one ``UnderstandDirective`` per matching item across all lists in
    the region (floor-level and entity-level property lists may both contain
    Understand lines).
    """
    directives: list[UnderstandDirective] = []
    i = 0
    while i < len(region):
        tok = region[i]
        if tok.type == "bullet_list_open" and tok.level == 0:
            for item_tokens in _list_items(region, i):
                for t in item_tokens:
                    if t.type == "inline":
                        m = _UNDERSTAND_RE.match(t.content.strip())
                        if m:
                            phrase_str, verb_id = m.group(1), m.group(2)
                            phrases = [p.strip() for p in phrase_str.split("/") if p.strip()]
                            directives.append(
                                UnderstandDirective(phrases=phrases, verb_id=verb_id)
                            )
                        break
            # Skip past the close of this list
            depth = 0
            while i < len(region):
                if region[i].type == "bullet_list_open":
                    depth += 1
                elif region[i].type == "bullet_list_close":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            continue
        i += 1
    return directives


def _parse_bullet_list_as_props(
    region: list[Token], start: int
) -> dict[str, Any]:
    """Parse a bullet_list (starting at `start`) as a property dict.

    Supports nested bullet lists as sub-dicts: `- exits:\\n  - north: target`
    becomes `{"exits": {"north": "target"}}`.

    Returns {} if any top-level item fails the key:value pattern (non-strict:
    the list is treated as prose).
    """
    out: dict[str, Any] = {}
    items = _list_items(region, start)
    for item_tokens in items:
        parsed = _parse_list_item_as_property(item_tokens)
        if parsed is None:
            # Non-strict: a list that isn't entirely properties is prose.
            return {}
        if parsed is _UNDERSTAND_ITEM:
            # Understand directive — not a key:value property, but valid in a
            # property list.  Skip here; collected by _collect_understand_directives.
            continue
        key, value = parsed
        out[key] = value
    return out


def _list_items(region: list[Token], list_open_idx: int) -> list[list[Token]]:
    """Return the token-slices for each list_item in the bullet_list at
    `list_open_idx`. Each slice is the tokens *between* the item's
    list_item_open/list_item_close, exclusive.
    """
    items: list[list[Token]] = []
    depth = 0
    item_start: int | None = None
    item_depth: int | None = None
    i = list_open_idx
    while i < len(region):
        tok = region[i]
        if tok.type == "bullet_list_open":
            depth += 1
        elif tok.type == "bullet_list_close":
            depth -= 1
            if depth == 0:
                return items
        elif tok.type == "list_item_open" and depth == 1:
            item_start = i + 1
            item_depth = depth
        elif tok.type == "list_item_close" and depth == 1 and item_start is not None:
            items.append(region[item_start:i])
            item_start = None
        i += 1
    return items


def _parse_list_item_as_property(item_tokens: list[Token]) -> tuple[str, Any] | None:
    """Parse a single list_item's tokens. Returns (key, value) or None.

    Returns the module-level sentinel ``_UNDERSTAND_ITEM`` when the item
    matches the Understand directive pattern — the caller should skip it
    (don't add to the property dict) but should not treat the whole list
    as prose.
    """
    # Find the first inline whose content begins with `key: ...`.
    key: str | None = None
    raw_value: str | None = None
    inline_idx: int | None = None
    for i, tok in enumerate(item_tokens):
        if tok.type == "inline":
            text = tok.content.strip()
            # Understand directives look like **Understand** "..." as verb_id
            # They are not key:value pairs — skip them without breaking the list.
            if _UNDERSTAND_RE.match(text):
                return _UNDERSTAND_ITEM
            m = _KV_RE.match(text)
            if not m:
                return None
            key = m.group("key")
            raw_value = m.group("value").strip()
            inline_idx = i
            break
    if key is None:
        return None

    # Prose-typed property: value starts with "> " (single-line form).
    # Per PROSE.md §2.1: store as ProseValue for lowering to a Luau function.
    raw = raw_value or ""
    if raw.startswith(_PROSE_VALUE_PREFIX):
        prose_line = raw[len(_PROSE_VALUE_PREFIX):]
        # Hard error on old conditional syntax (PROSE.md §8.1).
        if _OLD_IF_RE.search(prose_line):
            raise FmlSyntaxError(
                f"Old conditional prose syntax '[if ...]' found in property {key!r}. "
                "Migrate to backtick-embedded Luau per docs/design/PROSE.md §8.2."
            )
        return key, ProseValue(lines=(prose_line,))

    # Check for a nested bullet list following the inline → sub-dict value.
    has_nested = any(
        t.type == "bullet_list_open"
        for t in item_tokens[(inline_idx or 0) + 1 :]
    )
    if has_nested:
        # Find the nested list start
        nested_start = next(
            i
            for i, t in enumerate(item_tokens)
            if t.type == "bullet_list_open" and i > (inline_idx or 0)
        )
        nested = _parse_bullet_list_as_props(item_tokens, nested_start)
        # If the inline had a non-empty value AND there's a nested list,
        # prefer the nested dict (the inline value is just the key label).
        # Otherwise return whatever we got from the nested parse.
        if nested:
            return key, nested
        # Fall through to scalar handling if the nested list wasn't all properties.

    # Hard error on old conditional syntax in any plain string property value
    # (PROSE.md §8.1 — the old syntax must not appear anywhere).
    if _OLD_IF_RE.search(raw):
        raise FmlSyntaxError(
            f"Old conditional prose syntax '[if ...]' found in property {key!r}. "
            "Migrate to backtick-embedded Luau per docs/design/PROSE.md §8.2."
        )

    return key, _parse_scalar(raw)


def _parse_scalar(s: str) -> Any:
    """Parse a property value as int / float / bool / quoted string / list /
    bare string — or a dice expression / Luau code fragment.

    Six-form dice/Luau contract (per docs/design/PARSER.md §5.8 and task #46):

        Form 1 — bare dice:            ``3d6+2``
            Resolved at parse time (SeededRng seed 0); stored as int literal.

        Form 2 — backtick function:    `` `function(self, ctx) ... end` ``
            Stored as LuauCode verbatim.

        Form 3 — backtick literal:     `` `13` ``
            Strip backticks; parse inner value as a scalar.

        Form 4 — bare literal:         ``13``
            Stored as int/float/bool/str (existing behaviour, unchanged).

        Form 5 — backtick dice thunk:  `` `3d6+2` ``
            Wrapped in ``function(self, ctx) return engine.roll("3d6+2") end``;
            stored as LuauCode.

        Form 6 — backtick invoked:     `` `(3d6+2)()` ``
            Resolved at parse time; stored as int literal.

    List syntax is ``[a, b, c]`` (square brackets, comma-separated). Items are
    themselves parsed by ``_parse_scalar`` (so quoted, numeric, and bool items
    are supported). Plain comma-separated strings without brackets stay
    strings — ``damage_immunities: poison, necrotic`` is intentionally a
    single string per the canonical 5e stat-block convention.
    """
    s = s.strip()

    # ── Backtick forms (Forms 2, 3, 5, 6) ────────────────────────────────────
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        return parse_dice_value(s)

    # ── Bare dice (Form 1) ────────────────────────────────────────────────────
    if is_dice_expr(s):
        return parse_dice_value(s)

    # ── Forms 3 & 4 (existing scalar logic) ──────────────────────────────────
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
    # Markdown link: [text](#anchor) or [text](target) → slugified entity ID.
    m_link = _LINK_RE.fullmatch(s)
    if m_link:
        target = m_link.group(2)
        anchor = target.lstrip("#").replace("%20", " ")
        return slugify(anchor)
    if _is_bracket_list(s):
        return _parse_bracket_list(s[1:-1])
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _is_bracket_list(s: str) -> bool:
    """True only if ``s`` is a balanced ``[...]`` expression where the opening
    ``[`` at index 0 closes at the final character.

    A value like ``[if X in Y]Z[end if]`` starts with ``[`` and ends with ``]``
    but its first ``]`` closes mid-string — the trailing ``[...]`` is a
    separate run, not a nested item. Treating such a value as a list strips
    the outer brackets and corrupts embedded DSL tags, so we reject it here.
    """
    if len(s) < 2 or not (s.startswith("[") and s.endswith("]")):
        return False
    depth = 0
    in_quote = False
    prev = ""
    last_idx = len(s) - 1
    for i, ch in enumerate(s):
        if ch == '"' and prev != "\\":
            in_quote = not in_quote
            prev = ch
            continue
        prev = ch
        if in_quote:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and i != last_idx:
                return False
            if depth < 0:
                return False
    return depth == 0


def _parse_bracket_list(body: str) -> list[Any]:
    """Split a bracket-list body by commas at the top level (respecting nested
    brackets and double-quoted strings), then `_parse_scalar` each item."""
    if not body.strip():
        return []
    items: list[str] = []
    current: list[str] = []
    depth = 0
    in_quote = False
    for ch in body:
        if ch == '"' and (not current or current[-1] != "\\"):
            in_quote = not in_quote
            current.append(ch)
        elif in_quote:
            current.append(ch)
        elif ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current or items:
        items.append("".join(current))
    return [_parse_scalar(item) for item in items]


# ─── Prose extraction ─────────────────────────────────────────────────────────


def _collect_prose_blockquotes(region: list[Token]) -> list[ProseValue]:
    """Return all prose paragraphs from top-level blockquotes in `region`.

    Returns a list of ProseValue objects (one per blockquote paragraph).
    Each ProseValue carries the lines of that paragraph.

    Old conditional syntax ([if X in Y]) in blockquote prose is a hard
    FmlSyntaxError (PROSE.md §8.1).
    """
    paragraphs: list[ProseValue] = []
    i = 0
    while i < len(region):
        tok = region[i]
        if tok.type == "blockquote_open" and tok.level == 0:
            lines, next_i = _blockquote_lines(region, i)
            if lines:
                for line in lines:
                    if _OLD_IF_RE.search(line):
                        raise FmlSyntaxError(
                            "Old conditional prose syntax '[if ...]' found in blockquote. "
                            "Migrate to backtick-embedded Luau per docs/design/PROSE.md §8.2."
                        )
                paragraphs.append(ProseValue(lines=tuple(lines)))
            i = next_i
            continue
        i += 1
    return paragraphs


def _blockquote_lines(region: list[Token], start: int) -> tuple[list[str], int]:
    """Capture prose lines from a blockquote starting at `start`.

    Returns (lines, index-past-blockquote-close).  Each inline token inside the
    blockquote becomes a separate line.  Within a single inline token, newlines
    split the content into multiple lines (markdown-it folds soft-wrapped lines
    into one inline with embedded newlines).
    """
    lines: list[str] = []
    depth = 0
    i = start
    while i < len(region):
        tok = region[i]
        if tok.type == "blockquote_open":
            depth += 1
        elif tok.type == "blockquote_close":
            depth -= 1
            if depth == 0:
                return lines, i + 1
        elif tok.type == "inline" and depth >= 1:
            for sub in tok.content.splitlines():
                stripped = sub.rstrip()
                if stripped:
                    lines.append(stripped)
        i += 1
    return lines, i


def _blockquote_text(region: list[Token], start: int) -> tuple[str, int]:
    """Capture the prose text inside a blockquote starting at `start`.

    Returns (text, index-past-blockquote-close). Joins multiple inline
    paragraphs inside one blockquote with newlines.

    Legacy helper — still used by _extract_top_metadata for floor-level prose.
    Entity blockquote prose now goes through _blockquote_lines instead.
    """
    pieces: list[str] = []
    depth = 0
    i = start
    while i < len(region):
        tok = region[i]
        if tok.type == "blockquote_open":
            depth += 1
        elif tok.type == "blockquote_close":
            depth -= 1
            if depth == 0:
                return "\n".join(pieces).strip(), i + 1
        elif tok.type == "inline" and depth >= 1:
            pieces.append(tok.content)
        i += 1
    return "\n".join(pieces).strip(), i


def _merge_prose_values(pieces: list[ProseValue]) -> "ProseValue | str":
    """Merge multiple ProseValue paragraphs into a single ProseValue.

    Multiple blockquote paragraphs (adjacent ``> ...`` blocks) are joined with
    a single empty-string line as a paragraph separator.  Returns empty string
    when there are no pieces, for backward compat with the ``if entity.prose``
    check in the emitter.
    """
    if not pieces:
        return ""
    if len(pieces) == 1:
        return pieces[0]
    # Merge all lines from all pieces.
    all_lines: list[str] = []
    for i, pv in enumerate(pieces):
        if i > 0:
            # Empty line marks a paragraph break within the prose function.
            all_lines.append("")
        all_lines.extend(pv.lines)
    return ProseValue(lines=tuple(all_lines))


def _extract_links_from_prose(prose: str) -> list[str]:
    """Find slugified entity ids in markdown links inside prose."""
    out: list[str] = []
    for m in _LINK_RE.finditer(prose):
        target = m.group(2)
        if target.startswith("#"):
            out.append(slugify(target[1:].replace("%20", " ")))
    return list(dict.fromkeys(out))  # dedupe preserving order


# ─── H2 sections and H3 entities ──────────────────────────────────────────────


def _iter_sections_and_entities(
    tokens: list[Token],
) -> Iterator[tuple[str, Iterator[tuple[str, list[Token]]]]]:
    """Yield (section_name, entities_iter) for each H2 in the document.

    Each entity is (entity_name, entity_tokens) where entity_tokens spans the
    H3's heading_close to the next H3-or-higher heading.

    H3 entities that appear BEFORE any H2 (or in a document with no H2 at all)
    are emitted under a synthetic empty section name; their default kind is
    resolved against the (possibly empty) stdlib map, falling through to the
    root kind `entity` when nothing matches.
    """
    first_h2 = _first_h2_index(tokens)

    # Pre-H2 entities: any H3 between the H1 (or start of doc) and the first H2.
    # Find the start: just after the H1 if present, else 0.
    h1_end = _index_after_h1(tokens) if any(_heading_level(t) == 1 for t in tokens) else 0
    if first_h2 > h1_end:
        # Check whether there are H3s in this pre-H2 range
        bare_entities = list(_iter_entities(tokens, h1_end, first_h2))
        if bare_entities:
            yield "", iter(bare_entities)

    i = first_h2
    while i < len(tokens):
        if _heading_level(tokens[i]) == 2:
            section_name = _inline_text_after(tokens, i)
            # Find the section end (next H2 or end of doc).
            section_end = _next_heading_at_or_above(tokens, i + 3, 2)
            yield section_name, _iter_entities(tokens, i + 3, section_end)
            i = section_end
        else:
            i += 1


def _first_h2_index(tokens: list[Token]) -> int:
    for i, tok in enumerate(tokens):
        if _heading_level(tok) == 2:
            return i
    return len(tokens)


def _iter_entities(
    tokens: list[Token], start: int, end: int
) -> Iterator[tuple[str, list[Token]]]:
    """Yield (entity_name, body_tokens) for each H3 between [start, end)."""
    i = start
    while i < end:
        if _heading_level(tokens[i]) == 3:
            entity_name = _inline_text_after(tokens, i)
            entity_body_start = i + 3
            # Find next H3-or-higher within this section
            entity_end = _next_heading_at_or_above(tokens, entity_body_start, 3)
            entity_end = min(entity_end, end)
            yield entity_name, tokens[entity_body_start:entity_end]
            i = entity_end
        else:
            i += 1


# ─── Entity parsing (recursive for sub-entities) ──────────────────────────────


def _parse_entity(
    name: str,
    tokens: list[Token],
    default_kind: str,
    heading_level: int,
    kind_maps: _KindMaps | None = None,
) -> FMLEntity:
    """Parse a single entity (H3 or H5) from its body tokens.

    The "own" body of an entity is everything BEFORE its first child heading:
    - For H3 parents, "own" stops at the first H4/H5/H6.
    - For H5 parents, "own" stops at the first H6.
    Anything after that boundary is sub-entity or trigger material, parsed
    separately.
    """
    # Find the boundary between this entity's own content and its children.
    own_end = _next_child_heading_index(tokens, heading_level)
    own_body = tokens[:own_end]

    # Property list (first bullet list in the entity's OWN body)
    properties = _extract_first_property_list(own_body)

    # Prose blockquotes (only from the entity's OWN body).
    # Returns list[ProseValue]; merge all paragraphs into a single ProseValue
    # by concatenating lines (paragraph break = empty line between).
    prose_pieces = _collect_prose_blockquotes(own_body)
    prose = _merge_prose_values(prose_pieces)
    # Extract links from the plain-text representation of prose for the link graph.
    prose_plain = " ".join(
        line for pv in prose_pieces for line in pv.lines
    ) if prose_pieces else ""
    links = _extract_links_from_prose(prose_plain)

    # Bare scripts (fenced code blocks in the entity's OWN body that aren't
    # under any H6 trigger). Triggers from the parent's own body are H6s in
    # the own_body region; v0 H6s inside an H3's own body are rare but valid.
    own_triggers = list(_parse_h6_triggers(own_body))
    # Scripts in own_body that aren't claimed by any of those H6s
    scripts = list(_parse_bare_scripts(own_body))

    # Sub-entities and triggers from the remainder (after own_end).
    # _parse_entity_children handles #### Triggers H4 detection and routes
    # H6 headings appropriately — either as new-style triggers (verbatim name)
    # or old-style sub-entity triggers (slugified), plus H5 sub-entities.
    subentities: list[FMLEntity] = []
    if heading_level == 3:
        remainder = tokens[own_end:]
        child_triggers, subentities = _parse_entity_children(
            remainder, parent_heading_level=3, kind_maps=kind_maps
        )
        own_triggers.extend(child_triggers)

    # For H5 entities, any H6 triggers in the remainder belong to them.
    # Pass the remainder through _parse_entity_children so it handles the
    # #### Triggers H4 detection for H5 sub-entities too.
    if heading_level == 5:
        remainder_5 = tokens[own_end:]
        child_triggers_5, _ = _parse_entity_children(
            remainder_5, parent_heading_level=5, kind_maps=kind_maps
        )
        own_triggers.extend(child_triggers_5)

    # Determine kind — if `properties` has explicit `kind`, that wins
    kind = str(properties.get("kind", default_kind))

    entity_id = slugify(name)

    return FMLEntity(
        id=entity_id,
        name=name,
        kind=kind,
        properties=properties,
        prose=prose,
        links=links,
        subentities=subentities,
        scripts=scripts,
        triggers=own_triggers,
    )


def _next_child_heading_index(tokens: list[Token], parent_level: int) -> int:
    """Find the index of the first heading_open that starts a new sub-entity
    container (NOT a trigger).

    - For H3 parent (level=3): boundary is H4 or H5. H6 in own_body still
      belongs to the H3 (it's a trigger on the parent, not a sub-entity).
    - For H5 parent (level=5): boundary is H6 — and H6s belong to the H5 too,
      so we don't actually use this for H5; in practice we pass parent_level=5
      and the function looks for H6 as the trigger boundary.
    """
    if parent_level == 3:
        boundary_levels = (4, 5)
    else:
        boundary_levels = (6,)
    for i, tok in enumerate(tokens):
        lvl = _heading_level(tok)
        if lvl is not None and lvl in boundary_levels:
            return i
    return len(tokens)


def _index_of_first_heading_at(tokens: list[Token], level: int) -> int:
    """Return the index of the first heading_open at exactly `level`."""
    for i, tok in enumerate(tokens):
        if _heading_level(tok) == level:
            return i
    return len(tokens)


def _parse_bare_scripts(tokens: list[Token]) -> Iterator[Script]:
    """Fenced code blocks that aren't preceded by an H6 in `tokens`.

    Walk tokens; for each H6 heading, skip its associated code block. The
    remaining top-level fenced code blocks are bare scripts.
    """
    claimed_indices: set[int] = set()
    i = 0
    while i < len(tokens):
        if _heading_level(tokens[i]) == 6:
            # The next fence (if any) is claimed by this H6
            next_h6 = _next_heading_at_or_above(tokens, i + 3, 6)
            for j in range(i + 3, next_h6):
                if tokens[j].type == "fence":
                    claimed_indices.add(j)
                    break
            i = next_h6
            continue
        i += 1
    for idx, tok in enumerate(tokens):
        if tok.type == "fence" and idx not in claimed_indices:
            yield Script(source=tok.content.rstrip("\n"))


def _parse_entity_children(
    tokens: list[Token],
    parent_heading_level: int,
    kind_maps: _KindMaps | None = None,
) -> tuple[list[Trigger], list[FMLEntity]]:
    """Walk the body of an H3 or H5 entity after its own-body boundary and
    return (triggers, sub_entities).

    Detects ``#### Triggers`` H4 subsections:
    - H6 headings under a ``Triggers`` H4 → new-style trigger declarations on the
      parent entity.  The H6 heading text is the trigger name **verbatim** (not
      slugified).
    - H6 headings NOT under a ``Triggers`` H4 (old-style / backward compat) →
      trigger declarations with **slugified** name and a deprecation warning.
    - H5 headings under any non-Triggers H4 (or before any H4) → sub-entities.

    For H5 parents (parent_heading_level=5), the token slice passed here already
    contains only the H5's body beyond own_end, which is only H6s (triggers).
    """
    triggers: list[Trigger] = []
    subentities: list[FMLEntity] = []
    current_h4_label: str | None = None
    in_triggers_section = False

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        lvl = _heading_level(tok)

        if lvl == 4:
            current_h4_label = _inline_text_after(tokens, i)
            in_triggers_section = (
                current_h4_label.strip().lower() == "triggers"
            )
            i += 3
            continue

        if lvl == 6:
            trigger_name_raw = _inline_text_after(tokens, i)
            body_start = i + 3
            next_h6 = _next_heading_at_or_above(tokens, body_start, 6)
            trigger_tokens = tokens[body_start:next_h6]

            if in_triggers_section:
                # New-style: name is verbatim (PascalCase event name).
                trigger = _parse_new_style_trigger(trigger_name_raw, trigger_tokens)
            else:
                # Old-style: slugify the heading (backward compat).
                warnings.warn(
                    f"FML trigger {trigger_name_raw!r}: H6 triggers should be "
                    "declared under '#### Triggers'. Slugifying name for backward "
                    "compatibility.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                trigger = _parse_old_style_trigger(trigger_name_raw, trigger_tokens)

            triggers.append(trigger)
            i = next_h6
            continue

        if lvl == 5 and not in_triggers_section:
            sub_name = _inline_text_after(tokens, i)
            body_start = i + 3
            body_end = _next_heading_at_or_above(tokens, body_start, 5)
            sub_body = tokens[body_start:body_end]
            default_kind = _resolve_subsection_kind(
                current_h4_label, kind_maps or _KindMaps()
            )
            subentities.append(
                _parse_entity(
                    sub_name,
                    sub_body,
                    default_kind=default_kind,
                    heading_level=5,
                    kind_maps=kind_maps,
                )
            )
            i = body_end
            continue

        i += 1

    return triggers, subentities


def _parse_new_style_trigger(name_raw: str, tokens: list[Token]) -> Trigger:
    """Parse an H6 trigger under a ``#### Triggers`` H4 (v2 format).

    Body is a sequence of text lines extracted from the token stream:
    - ``- when: <pred>`` → trigger guard
    - Bare links, Form A blocks, action lines → body items

    Luau code blocks (```lua / ```luau) are stored in trigger.script without
    a warning — they are the correct stdlib implementation language. Python
    code blocks (```python or unlabeled) are deprecated and emit a warning.
    """
    lines = _extract_trigger_body_lines(tokens)
    script: Script | None = None

    fence = _first_fence(tokens)
    if fence is not None:
        lang, source = fence
        if lang in ("lua", "luau"):
            script = Script(language=lang, source=source)  # type: ignore[arg-type]
        else:
            warnings.warn(
                f"FML trigger {name_raw!r}: Python code blocks in trigger bodies are "
                "deprecated. Use Form A control flow and action vocabulary instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            script = Script(language="python", source=source)
        # Remove fence lines so they don't double-parse as body text.
        lines = [l for l in lines if not l.startswith("```")]

    when, body = parse_trigger_body(lines)
    return Trigger(name=name_raw, when=when, body=body, script=script)


def _parse_old_style_trigger(name_raw: str, tokens: list[Token]) -> Trigger:
    """Parse a legacy H6 trigger NOT under a ``#### Triggers`` H4.

    Slugifies the name and stores the first Python code block as the script.
    """
    script_source = ""
    fence_source = _first_fence_source(tokens)
    if fence_source is not None:
        script_source = fence_source
    return Trigger(
        name=_canonical_event_name(name_raw),
        script=Script(source=script_source),
    )


def _first_fence_source(tokens: list[Token]) -> str | None:
    """Return the source of the first fenced code block in `tokens`, or None."""
    for tok in tokens:
        if tok.type == "fence":
            return tok.content.rstrip("\n")
    return None


def _first_fence(tokens: list[Token]) -> tuple[str, str] | None:
    """Return (language, source) of the first fenced code block, or None."""
    for tok in tokens:
        if tok.type == "fence":
            return (tok.info.strip().lower(), tok.content.rstrip("\n"))
    return None


def _extract_trigger_body_lines(tokens: list[Token]) -> list[str]:
    """Extract non-fence text from trigger body tokens as raw lines.

    Blockquote-prefixed lines are returned with a leading ``> `` so the trigger
    body parser can recognise them as OutputLine nodes.
    """
    lines: list[str] = []
    blockquote_depth = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "blockquote_open":
            blockquote_depth += 1
        elif tok.type == "blockquote_close":
            blockquote_depth -= 1
        elif tok.type == "inline":
            prefix = "> " if blockquote_depth > 0 else ""
            for line in tok.content.splitlines():
                lines.append(f"{prefix}{line}")
        i += 1
    return lines


def _parse_subentities(
    tokens: list[Token], kind_maps: _KindMaps | None = None
) -> Iterator[FMLEntity]:
    """Yield H5 sub-entities found within `tokens` (H3 body).

    Sub-entities are grouped under H4 organizational headings; H4 itself
    produces no LFR object but informs the stdlib resolver for kind defaults.

    NOTE: This function is preserved for legacy call-sites that only need
    sub-entities. New code uses ``_parse_entity_children`` which also returns
    triggers.
    """
    _, subentities = _parse_entity_children(tokens, parent_heading_level=3, kind_maps=kind_maps)
    yield from subentities


def _parse_h6_triggers(tokens: list[Token]) -> Iterator[Trigger]:
    """Walk `tokens` and yield Triggers for each H6 found.

    This is the legacy path: called for H6s in ``own_body`` (before any H4/H5).
    These are old-style triggers directly under an entity with no ``#### Triggers``
    H4 wrapper — slugified names, Python script bodies.

    H6s in the remainder (after H4s) are handled by ``_parse_entity_children``.
    """
    i = 0
    while i < len(tokens):
        if _heading_level(tokens[i]) == 6:
            trigger_name = _inline_text_after(tokens, i)
            body_start = i + 3
            next_h6 = _next_heading_at_or_above(tokens, body_start, 6)
            trigger_tokens = tokens[body_start:next_h6]
            script_source = _first_fence_source(trigger_tokens) or ""
            yield Trigger(
                name=_canonical_event_name(trigger_name),
                script=Script(source=script_source),
            )
            i = next_h6
            continue
        i += 1


def _parse_top_level_scripts(
    tokens: list[Token], parent_heading_level: int
) -> Iterator[Script]:
    """Yield code-block scripts at the entity's top level (not under any H6).

    Fenced code blocks that appear AFTER an H6 are claimed by that trigger;
    those before any H6 (and not nested in any sub-entity) are bare scripts.
    """
    # Stop walking when we hit any heading >= the entity's children boundary.
    boundary_level = 5 if parent_heading_level == 3 else 6
    end_idx = _next_heading_at_or_above(tokens, 0, boundary_level)
    for j in range(end_idx):
        tok = tokens[j]
        if tok.type == "fence":
            source = tok.content.rstrip("\n")
            yield Script(source=source)


def _canonical_event_name(heading: str) -> str:
    """Convert an H6 heading like 'On Damaged' or 'on cast' to 'on_damaged' /
    'on_cast' (snake_case)."""
    return slugify(heading)


# ─── Named-import resolution ──────────────────────────────────────────────────


_NAMED_IMPORTS: dict[str, str] = {
    # Convention: `[stdlib](stdlib)` from anywhere in the project resolves to
    # this canonical entry point, which re-exports the default rule system.
    "stdlib": "data/stdlib/index.md",
}

_PROJECT_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git", "CLAUDE.md")


def _resolve_import_path(link_target: str, base_dir: Path) -> Path:
    """Resolve an import link target to an absolute filesystem path.

    See `_load_import` for the resolution order. Named imports fall back to
    a path-style resolution if the project root cannot be discovered, so an
    error is raised at file-open time (with the more useful message) rather
    than here.
    """
    if "/" not in link_target and "." not in link_target:
        named = _NAMED_IMPORTS.get(link_target)
        if named is not None:
            project_root = _find_project_root(base_dir)
            if project_root is not None:
                return (project_root / named).resolve()
    return (base_dir / link_target).resolve()


def _find_project_root(start: Path) -> Path | None:
    """Walk up from `start` looking for a project-root marker file/dir."""
    current = start.resolve()
    while True:
        for marker in _PROJECT_ROOT_MARKERS:
            if (current / marker).exists():
                return current
        if current.parent == current:
            return None
        current = current.parent


