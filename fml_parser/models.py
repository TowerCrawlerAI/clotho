"""Pydantic models for the Lowered Floor Representation (LFR).

See `docs/design/LFR.md` for the full specification. The model hierarchy mirrors
the FML entity hierarchy: a Floor contains entities; entities can contain
sub-entities (recursive); entities carry properties, prose, links, scripts,
and triggers.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field as dc_field
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
)

from .errors import FmlValidationError

# ─── EntityId ──────────────────────────────────────────────────────────────────

_ENTITY_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _check_entity_id(s: str) -> str:
    if not _ENTITY_ID_RE.fullmatch(s):
        raise ValueError(
            f"EntityId must be snake_case ASCII (got {s!r}); "
            "derive from a heading via slugify()"
        )
    return s


EntityId = Annotated[str, AfterValidator(_check_entity_id)]


# ─── Scripts and triggers ─────────────────────────────────────────────────────


class Script(BaseModel):
    """Inline code attached to an entity, from an FML fenced code block."""

    model_config = ConfigDict(extra="forbid")

    language: Literal["python", "lua", "luau"] = "python"
    source: str
    line_offset: int = 0


# ─── TriggerBody AST types ────────────────────────────────────────────────────


class BareLink(BaseModel):
    """A bare Markdown link action: [Text](target)"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["bare_link"] = "bare_link"
    text: str
    target: str


class ActionLine(BaseModel):
    """A trigger action line: Verb [noun](ref) ..."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["action"] = "action"
    verb: str
    raw: str


class BreakStmt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["break"] = "break"


class ContinueStmt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["continue"] = "continue"


class ElseIfClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str
    body: list["TriggerBodyItem"] = Field(default_factory=list)


class IfBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["if"] = "if"
    condition: str
    body: list["TriggerBodyItem"] = Field(default_factory=list)
    else_if_clauses: list[ElseIfClause] = Field(default_factory=list)
    else_body: list["TriggerBodyItem"] | None = None


class WhileBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["while"] = "while"
    condition: str
    body: list["TriggerBodyItem"] = Field(default_factory=list)


class LoopNBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["loop_n"] = "loop_n"
    count: int
    body: list["TriggerBodyItem"] = Field(default_factory=list)


class LoopThroughBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["loop_through"] = "loop_through"
    collection: str
    body: list["TriggerBodyItem"] = Field(default_factory=list)


class LoopUntilBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["loop_until"] = "loop_until"
    condition: str
    max_iterations: int = 10000
    body: list["TriggerBodyItem"] = Field(default_factory=list)


class PropertySet(BaseModel):
    """Native FML property setter: - *entity.prop*: value"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["property_set"] = "property_set"
    path: str
    value: Any


class OutputLine(BaseModel):
    """Native FML output line: > text with *path* substitutions"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["output_line"] = "output_line"
    template: str


TriggerBodyItem = Annotated[
    BareLink
    | ActionLine
    | BreakStmt
    | ContinueStmt
    | IfBlock
    | WhileBlock
    | LoopNBlock
    | LoopThroughBlock
    | LoopUntilBlock
    | PropertySet
    | OutputLine,
    Field(discriminator="kind"),
]

ElseIfClause.model_rebuild()
IfBlock.model_rebuild()
WhileBlock.model_rebuild()
LoopNBlock.model_rebuild()
LoopThroughBlock.model_rebuild()
LoopUntilBlock.model_rebuild()


class Trigger(BaseModel):
    """An H6-named event handler attached to its parent entity."""

    model_config = ConfigDict(extra="forbid")

    name: str
    when: str | None = None
    body: list[TriggerBodyItem] = Field(default_factory=list)
    script: Script | None = None


# ─── Predicates (composable, serializable) ────────────────────────────────────


class Predicate(BaseModel):
    """Base class for completion / discovery predicates.

    Subclasses are discriminated by the `op` field. Composes with
    `&`, `|`, `~`.
    """

    model_config = ConfigDict(extra="forbid")

    op: str

    def __and__(self, other: Predicate) -> And:
        return And(left=self, right=other)

    def __or__(self, other: Predicate) -> Or:
        return Or(left=self, right=other)

    def __invert__(self) -> Not:
        return Not(inner=self)


class Killed(Predicate):
    op: Literal["killed"] = "killed"
    target: str


class Flag(Predicate):
    op: Literal["flag"] = "flag"
    name: str


class Found(Predicate):
    op: Literal["found"] = "found"
    target: str


class DialogueSuccess(Predicate):
    op: Literal["dialogue_success"] = "dialogue_success"
    target: str


class Entered(Predicate):
    op: Literal["entered"] = "entered"
    target: str


class And(Predicate):
    op: Literal["and"] = "and"
    left: Predicate
    right: Predicate


class Or(Predicate):
    op: Literal["or"] = "or"
    left: Predicate
    right: Predicate


class Not(Predicate):
    op: Literal["not"] = "not"
    inner: Predicate


class Has(Predicate):
    op: Literal["has"] = "has"
    actor: str
    item: str


class At(Predicate):
    op: Literal["at"] = "at"
    entity: str
    place: str


class Examined(Predicate):
    op: Literal["examined"] = "examined"
    noun: str
    by: str = "player"


class Visited(Predicate):
    op: Literal["visited"] = "visited"
    place: str


# ─── Entities ─────────────────────────────────────────────────────────────────


class FMLEntity(BaseModel):
    """An entity in the world.

    Uniform shape across all kinds — `npc`, `item`, `room`, `encounter`, `quest`,
    `wandering`, `wandering_entry`, `adjudicator_note`, `trait`, `action`,
    `reaction`, `legendary_action`, `spell`, etc. Kind-specific data lives in
    `properties` and is validated against the stdlib's per-kind schema.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: EntityId
    name: str
    kind: str
    properties: dict[str, Any] = Field(default_factory=dict)
    # prose is str for plain text, ProseValue for compiled prose templates.
    # Use Any so Pydantic doesn't reject the ProseValue dataclass.
    prose: Any = ""
    links: list[EntityId] = Field(default_factory=list)
    subentities: list[FMLEntity] = Field(default_factory=list)
    scripts: list[Script] = Field(default_factory=list)
    triggers: list[Trigger] = Field(default_factory=list)
    # Stdlib-resolved ancestor chain, computed at parse time from the
    # kind_definition entities' `ancestor` fields. Includes `self.kind` as
    # the first element and walks up to the root. Empty when no stdlib is
    # available or kind is unknown — consumers should treat this as
    # advisory, not load-bearing.
    kind_chain: list[str] = Field(default_factory=list)

    def walk(self) -> Iterator[FMLEntity]:
        """Pre-order DFS over self and all descendants."""
        yield self
        for child in self.subentities:
            yield from child.walk()

    def is_a(self, kind_name: str) -> bool:
        """True if `kind_name` appears anywhere in this entity's kind chain.

        Use this for kind-hierarchy queries: `entity.is_a("actor")` returns
        True for an NPC (since `npc.ancestor == actor`). Falls back to a
        literal kind comparison when the kind_chain wasn't resolved.
        """
        if self.kind_chain:
            return kind_name in self.kind_chain
        return self.kind == kind_name

    # ── Convenience accessors over `properties` for reserved keys ─────────────
    # (These replace v0.2's dedicated fields. Same ergonomic API.)

    @property
    def hidden_notes(self) -> list[str]:
        """Return the entity's `hidden` property as a list of strings."""
        return _coerce_str_list(self.properties.get("hidden"))

    @property
    def adjudicator_notes(self) -> list[str]:
        """Return the entity's `adjudicator` property as a list of strings."""
        return _coerce_str_list(self.properties.get("adjudicator"))

    @property
    def complete(self) -> Predicate | None:
        """Parse the entity's `complete` property as a Predicate (Quest entities)."""
        return _parse_pred_property(self.properties.get("complete"))

    @property
    def discovery(self) -> Predicate | None:
        """Parse the entity's `discovery` property as a Predicate (Quest entities)."""
        return _parse_pred_property(self.properties.get("discovery"))


def _is_prose_value(v: Any) -> bool:
    """Duck-type check for ProseValue.

    Using `isinstance(v, ProseValue)` fails in worktree isolation scenarios
    where the same class is imported from two different module copies — the
    identity check returns False even though the objects are structurally
    identical.  Checking by attribute presence is robust to this.
    """
    return hasattr(v, "lines") and hasattr(v, "source_path") and not isinstance(v, str)


def _merge_prose(a: Any, b: Any) -> Any:
    """Merge two prose values for the same entity declared in multiple files.

    When either value is a ProseValue, they cannot be concatenated as strings.
    In that case, the later declaration (b) wins — same as new-wins semantics
    for property values.  When both are plain strings, join with a blank line.
    """
    if not a and not b:
        return ""
    if not a:
        return b
    if not b:
        return a
    # If either is a ProseValue (or duck-type equivalent), later wins.
    if _is_prose_value(a) or _is_prose_value(b):
        return b
    # Both plain strings: append with separator.
    return a + "\n\n" + b


def _merge_entities(existing: FMLEntity, new: FMLEntity) -> FMLEntity:
    """Deep-merge two declarations of the same entity (per docs/design/FML.md §6.2).

    - properties: deep-merge dicts; new wins on per-key conflict.
    - prose: append new prose with a blank-line separator.
    - links: union, preserving order (existing first, then new not in existing).
    - subentities: merge by id (recurse), then append new ids.
    - scripts, triggers: append in source order (existing then new).
    - name, kind: new wins.
    """
    merged_props = _deep_merge_dicts(existing.properties, new.properties)
    merged_prose = _merge_prose(existing.prose, new.prose)
    merged_links = list(existing.links) + [
        link for link in new.links if link not in existing.links
    ]
    merged_subs = _merge_subentities(existing.subentities, new.subentities)
    merged_scripts = list(existing.scripts) + list(new.scripts)
    merged_triggers = list(existing.triggers) + list(new.triggers)

    return FMLEntity(
        id=existing.id,
        name=new.name or existing.name,
        kind=new.kind or existing.kind,
        properties=merged_props,
        prose=merged_prose,
        links=merged_links,
        subentities=merged_subs,
        scripts=merged_scripts,
        triggers=merged_triggers,
    )


def _deep_merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge: keys in `b` win; if both values are dicts, recurse.
    Lists in `b` replace lists in `a` (not concatenated) — list-valued
    properties are atomic.
    """
    out = dict(a)
    for k, v_new in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v_new, dict):
            out[k] = _deep_merge_dicts(out[k], v_new)
        else:
            out[k] = v_new
    return out


def _merge_subentities(
    existing: list[FMLEntity], new: list[FMLEntity]
) -> list[FMLEntity]:
    """Merge two sub-entity lists. Same-id entities deep-merge; new ids append."""
    by_id = {sub.id: sub for sub in existing}
    order = [sub.id for sub in existing]
    for new_sub in new:
        if new_sub.id in by_id:
            by_id[new_sub.id] = _merge_entities(by_id[new_sub.id], new_sub)
        else:
            by_id[new_sub.id] = new_sub
            order.append(new_sub.id)
    return [by_id[i] for i in order]


def _coerce_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _parse_pred_property(v: Any) -> Predicate | None:
    if v is None:
        return None
    if isinstance(v, Predicate):
        return v
    if isinstance(v, str):
        # Import locally to avoid a circular import with predicate_lang
        from .predicate_lang import parse_predicate

        return parse_predicate(v)
    return None


# Forward-reference resolution for recursive subentities
FMLEntity.model_rebuild()


# ─── Understand directives ────────────────────────────────────────────────────


@dataclass
class UnderstandDirective:
    """A parsed ``**Understand** "phrase/phrase" as verb_id`` directive.

    Each ``phrases`` entry is an alternative command string that the engine
    should treat as an alias for ``verb_id``.  The emitter turns each phrase
    into a ``engine.register_verb_alias(phrase, verb_id)`` call in the LFR.

    Directives may appear in the floor-level property list or in any entity's
    property list; in either case they are collected on the owning ``Floor``
    rather than stored as entity properties.
    """

    phrases: list[str]
    verb_id: str


# ─── Floor ────────────────────────────────────────────────────────────────────


class ProseChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    entity_id: EntityId
    entity_kind: str
    prose: Any  # str or ProseValue
    distance: int


class ContextBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: EntityId
    chunks: list[ProseChunk]

    def render(self) -> str:
        """Format as a single string block suitable for prompt injection."""
        from .dice_value import ProseValue  # local import to avoid circular

        def _prose_str(p: Any) -> str:
            if isinstance(p, ProseValue):
                return " ".join(p.lines)
            return str(p) if p else ""

        return "\n\n".join(
            f"--- {c.entity_kind} {c.entity_id} ---\n{_prose_str(c.prose)}"
            for c in self.chunks
        )


class Floor(BaseModel):
    """A single floor of the tower.

    Holds the floor's metadata and the flat dict of all top-level entities.
    Sub-entities are reachable via `walk()` from their parents but not in the
    top-level dict directly.

    `entry_point_entity_ids`, when non-None, is the set of entity ids declared
    directly in the entry-point document (NOT in any imported document). The
    reverse emitter uses this to filter — only entry-point entities are emitted,
    imported entities are left to their own source files.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    # prose is str for plain text, ProseValue for compiled prose templates.
    prose: Any = ""
    imports: list[str] = Field(default_factory=list)
    entities: dict[EntityId, FMLEntity] = Field(default_factory=dict)
    fml_source_hash: str | None = None
    entry_point_entity_ids: set[EntityId] | None = None
    understand_directives: list[UnderstandDirective] = Field(default_factory=list)

    # ── stdlib-derived metadata ──────────────────────────────────────────────
    # Section/subsection mappings resolved from the entire transitive import
    # tree. Each entry: (canonical heading text, kind id). Insertion order is
    # preserved and is also the canonical emission order (the order the stdlib
    # declared the bindings). When the same heading is re-declared by a later
    # layer (e.g. dnd5e overriding core's `People → actor` with `People → npc`),
    # the position stays at the first-layer declaration and only the kind is
    # updated, so the layered model produces a stable section order.
    #
    # The lists are populated by the parser from `## Section to Kind` and
    # `## Subsection to Kind` blocks in imported stdlib documents. Empty when
    # no stdlib was imported, in which case the emitter falls back to a
    # minimal hardcoded set.
    section_mappings: list[tuple[str, str]] = Field(default_factory=list)
    subsection_mappings: list[tuple[str, str]] = Field(default_factory=list)

    # Reserved property keys declared by the stdlib's `## Reserved Property
    # Keys` block. Each entry: (key_name, one-line description). Informational
    # in v0.1 — the parser doesn't validate; consumers may consult to know
    # which keys carry engine meaning vs which are author-defined extensions.
    reserved_property_keys: list[tuple[str, str]] = Field(default_factory=list)

    # Entity ids that came from stdlib imports (paths containing "stdlib/").
    # Populated by the parser for each entity loaded from a stdlib file.
    # Used by `tree_shake.roots_from_floor` to distinguish stdlib catalog
    # content (pruneable) from authored floor/party content (roots).
    # An empty set means "no stdlib entities tracked" — treated as unknown.
    stdlib_entity_ids: set[str] = Field(default_factory=set)

    # ── accumulation ──────────────────────────────────────────────────────────
    def __iadd__(self, entity: FMLEntity) -> Self:
        existing = self.entities.get(entity.id)
        if existing is None:
            self.entities[entity.id] = entity
        else:
            # Re-declaration merge per docs/design/FML.md §6.2: same id across
            # multiple imported docs (or same doc) is the same entity; later
            # properties win per key; prose appends; links union; sub-entities
            # merge by id; scripts/triggers append.
            self.entities[entity.id] = _merge_entities(existing, entity)
        return self

    # ── stdlib introspection ──────────────────────────────────────────────────
    def kind_hierarchy(self) -> dict[str, str | None]:
        """Build the kind → ancestor map from this floor's kind_definition entities.

        Returns a dict where each key is a kind name and each value is its
        parent kind, or None for root-of-hierarchy kinds. Walks
        `self.entities` looking for entities with kind=="kind_definition";
        reads each one's `name` and `ancestor` properties.
        """
        out: dict[str, str | None] = {}
        for entity in self.entities.values():
            if entity.kind != "kind_definition":
                continue
            name = entity.properties.get("name")
            if not isinstance(name, str):
                continue
            ancestor = entity.properties.get("ancestor")
            if ancestor in (None, "null", "None", ""):
                out[name] = None
            elif isinstance(ancestor, str):
                out[name] = ancestor
        return out

    def kind_chain(self, kind_name: str) -> list[str]:
        """Walk a kind's ancestor chain: returns [kind, parent, grandparent, …].

        Bottoms out at a root kind (whose ancestor is None) or at an unknown
        kind. Cycles are guarded against. The kind itself is always first;
        return value is never empty if `kind_name` is non-empty.
        """
        hierarchy = self.kind_hierarchy()
        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = kind_name
        while current and current not in seen:
            chain.append(current)
            seen.add(current)
            current = hierarchy.get(current)
        return chain

    def attributes_for(self, kind_name: str) -> dict[str, Any]:
        """Return the composite attribute schema for `kind_name`.

        Walks the ancestor chain from root downward, accumulating the
        `attributes` dict declared on each kind_definition. A child's
        declaration overrides a parent's for the same key (later in the
        chain wins per-key — this is the natural override semantics).

        v0.1: informational only. The runtime can introspect to know what
        fields an entity of kind `kind_name` is expected to carry; no
        validation is enforced. Returns `{}` if the kind has no chain or
        no kind_definition declares attributes.
        """
        chain = self.kind_chain(kind_name)
        if not chain:
            return {}
        merged: dict[str, Any] = {}
        # Walk root → leaf so deeper kinds override shallower ones.
        for k in reversed(chain):
            kd = self.entities.get(k)
            if kd is None or kd.kind != "kind_definition":
                continue
            attrs = kd.properties.get("attributes")
            if isinstance(attrs, dict):
                merged.update(attrs)
        return merged

    def populate_kind_chains(self) -> None:
        """Set `kind_chain` on every entity (recursively) from this floor's hierarchy.

        Called by the parser at the end of parsing; safe to re-call.
        """
        cache: dict[str, list[str]] = {}
        for entity in self.all_entities():
            chain = cache.get(entity.kind)
            if chain is None:
                chain = self.kind_chain(entity.kind)
                cache[entity.kind] = chain
            entity.kind_chain = chain

    # ── lookup ────────────────────────────────────────────────────────────────
    def by_kind(self, kind: str) -> list[FMLEntity]:
        return [e for e in self.entities.values() if e.kind == kind]

    def by_ancestor(self, kind: str) -> list[FMLEntity]:
        """Return every entity whose `kind_chain` contains `kind`.

        Use this for kind-hierarchy queries — `floor.by_ancestor("actor")`
        returns every NPC, PC, monster, etc. that derives from actor.
        """
        return [e for e in self.entities.values() if e.is_a(kind)]

    def event_names(self) -> list[str]:
        """Return the canonical event names declared by the stdlib.

        Reads from entities of kind `event`. Triggers (H6 handlers) may
        reference names outside this list — the vocabulary is extensible —
        but the runtime can warn on unknown events as a debugging aid.
        """
        names: list[str] = []
        for e in self.entities.values():
            if not e.is_a("event"):
                continue
            n = e.properties.get("name")
            if isinstance(n, str):
                names.append(n)
            else:
                names.append(e.id)
        return names

    def verbs(self) -> list[FMLEntity]:
        """Return entities of kind `verb` — the dispatcher's input vocabulary.

        Higher layers (`stdlib/dnd5e/`, modules) can add verbs; the runtime
        builds its dispatch table from the union of every layer's
        declarations after import resolution.
        """
        return [e for e in self.entities.values() if e.is_a("verb")]

    def find(self, entity_id: EntityId) -> FMLEntity | None:
        """Look up an entity by id, including nested sub-entities."""
        for top in self.entities.values():
            for ent in top.walk():
                if ent.id == entity_id:
                    return ent
        return None

    def all_entities(self) -> Iterator[FMLEntity]:
        """Pre-order DFS over every entity in the floor."""
        for top in self.entities.values():
            yield from top.walk()

    # ── context graph ─────────────────────────────────────────────────────────
    def context_for(self, root: EntityId, depth: int = 2) -> ContextBundle:
        """BFS over the link graph; assembles AI-priming prose.

        See `docs/design/LFR.md` § Context graph.
        """
        chunks: list[ProseChunk] = []
        visited: set[EntityId] = set()
        frontier: deque[tuple[EntityId, int]] = deque([(root, 0)])

        while frontier:
            eid, dist = frontier.popleft()
            if eid in visited or dist > depth:
                continue
            visited.add(eid)
            entity = self.find(eid)
            if entity is None:
                continue
            chunks.append(
                ProseChunk(
                    entity_id=eid,
                    entity_kind=entity.kind,
                    prose=entity.prose,
                    distance=dist,
                )
            )
            for next_id in entity.links:
                if next_id not in visited:
                    frontier.append((next_id, dist + 1))

        return ContextBundle(root=root, chunks=chunks)
