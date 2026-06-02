"""LuaReader — reconstruct a Floor model from a Luau LFR table-literal file.

Only the subset of Luau syntax emitted by ``emit_lua.py`` is supported.
The file structure is:

    -- <header comments>

    local floor = { name = "...", ... }

    local <section> = {}
    <section>.<id> = { id = "...", kind = "...", ... }
    <section>.<id>.triggers["<slot>"] = function(ctx)
        <opaque luau body>
    end

    return { floor = floor, <section> = <section>, ... }

The reader:
 1. Tokenises the source.
 2. Walks the token stream statement-by-statement.
 3. Reconstructs a ``Floor`` + ``FMLEntity`` graph.
 4. Stores trigger function bodies as opaque ``Script(language='luau', ...)``
    on the corresponding entity trigger.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..dice_value import LuauCode
from ..models import FMLEntity, Floor, Script, Trigger
from .tokenizer import TK, Token, TokenizeError, tokenize

# ── Reverse-map the trigger slot key back to a trigger name ──────────────────

_STAGE_REVERSE: dict[str, str] = {
    "test":       "Test",
    "instead_of": "InsteadOf",
    "before":     "Before",
    "on":         "On",
    "after":      "After",
    "report":     "Report",
}


def _slot_to_trigger_name(slot: str) -> str:
    """Convert ``'on:Enter'`` → ``'On Enter'``, ``'instead_of:Attack'`` → ``'InsteadOf Attack'``."""
    if ":" in slot:
        prefix, event = slot.split(":", 1)
        stage = _STAGE_REVERSE.get(prefix, prefix.title())
        return f"{stage} {event}" if event else stage
    return slot


# ── Parse error ───────────────────────────────────────────────────────────────


class LuaReadError(ValueError):
    pass


# ── Token stream cursor ───────────────────────────────────────────────────────


class _Cursor:
    """Thin wrapper around a token list with a single look-ahead."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    @property
    def current(self) -> Token:
        return self._tokens[self._pos]

    def peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]  # EOF
        return self._tokens[idx]

    def advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.kind != TK.EOF:
            self._pos += 1
        return tok

    def expect(self, kind: TK) -> Token:
        tok = self.advance()
        if tok.kind != kind:
            raise LuaReadError(
                f"Expected {kind.name} but got {tok.kind.name} {tok.value!r} at line {tok.line}"
            )
        return tok

    def match(self, *kinds: TK) -> bool:
        return self.current.kind in kinds

    def consume_if(self, kind: TK) -> bool:
        if self.current.kind == kind:
            self.advance()
            return True
        return False


# ── Value parser ─────────────────────────────────────────────────────────────


def _consume_function_literal(cur: _Cursor) -> LuauCode:
    """Parse and consume a ``function(...) ... end`` literal.

    Returns a ``LuauCode`` containing the full function source text.
    Used by ``_parse_value`` to handle function-valued properties in the
    LFR (e.g. ``prose = function(self, ctx) ... end``).

    Tracks nesting depth so inner ``function``/``end`` pairs don't
    prematurely close the outer function.
    """
    start_line = cur.current.line
    tokens: list[Token] = []

    # Consume the opening 'function' keyword.
    fn_tok = cur.expect(TK.FUNCTION)
    tokens.append(fn_tok)

    # Optional function name (for named function form: ``function name(...)``).
    if cur.match(TK.IDENT):
        name_tok = cur.current
        tokens.append(name_tok)
        cur.advance()

    # Consume the parameter list: '(' ... ')'.
    lparen = cur.expect(TK.LPAREN)
    tokens.append(lparen)
    while not cur.match(TK.RPAREN, TK.EOF):
        tokens.append(cur.current)
        cur.advance()
    if cur.match(TK.RPAREN):
        tokens.append(cur.current)
        cur.advance()

    # Consume the body tokens until the matching 'end'.
    # Track all block-opener keywords/identifiers, not just 'function'.
    _BLOCK_OPENER_IDENTS = frozenset({"if", "for", "while", "do"})
    depth = 1
    while depth > 0 and not cur.match(TK.EOF):
        tok = cur.current
        if tok.kind == TK.FUNCTION:
            depth += 1
        elif tok.kind == TK.IDENT and tok.value in _BLOCK_OPENER_IDENTS:
            depth += 1
        elif tok.kind == TK.END:
            depth -= 1
            if depth == 0:
                tokens.append(tok)
                cur.advance()
                break
        tokens.append(tok)
        cur.advance()

    source = _tokens_to_source(tokens)
    return LuauCode(source=source)


def _parse_value(cur: _Cursor) -> Any:
    """Parse a Lua value: string, number, bool, nil, table, or function literal."""
    tok = cur.current
    if tok.kind == TK.STRING:
        cur.advance()
        return tok.value
    if tok.kind == TK.NUMBER:
        cur.advance()
        return tok.value
    if tok.kind == TK.TRUE:
        cur.advance()
        return True
    if tok.kind == TK.FALSE:
        cur.advance()
        return False
    if tok.kind == TK.NIL:
        cur.advance()
        return None
    if tok.kind == TK.LBRACE:
        return _parse_table(cur)
    if tok.kind == TK.FUNCTION:
        # Function literal — e.g. ``prose = function(self, ctx) ... end``.
        # Return as LuauCode so the emitter can round-trip it correctly.
        return _consume_function_literal(cur)
    # Bare identifier as value (e.g. ``return { floor = floor, people = people }``).
    # Treat as a symbolic reference — return the name as a string sentinel.
    if tok.kind == TK.IDENT:
        cur.advance()
        return tok.value
    raise LuaReadError(f"Unexpected token {tok.kind.name} {tok.value!r} at line {tok.line}")


def _parse_table(cur: _Cursor) -> Any:
    """Parse ``{ ... }`` — either a list or a record, depending on first element."""
    cur.expect(TK.LBRACE)
    # Empty table → always a dict (matches emitter's ``{}``)
    if cur.match(TK.RBRACE):
        cur.advance()
        return {}

    # Look ahead: if the first token is a key-like thing followed by '='
    # → record.  Otherwise → list.
    first = cur.peek(0)
    second = cur.peek(1)

    if _is_key_start(first) and second.kind == TK.EQUALS:
        return _parse_record(cur)
    # Bracket key:  ["key"] = value  → also a record
    if first.kind == TK.LBRACKET:
        return _parse_record(cur)
    return _parse_list(cur)


def _is_key_start(tok: Token) -> bool:
    return tok.kind in (TK.IDENT, TK.STRING)


def _parse_record(cur: _Cursor) -> dict[str, Any]:
    """Parse ``{ key = value, ... }`` — Lua field-list record."""
    out: dict[str, Any] = {}
    while not cur.match(TK.RBRACE, TK.EOF):
        key = _parse_key(cur)
        cur.expect(TK.EQUALS)
        val = _parse_value(cur)
        out[key] = val
        cur.consume_if(TK.COMMA)
        cur.consume_if(TK.SEMICOLON)
    cur.expect(TK.RBRACE)
    return out


def _parse_list(cur: _Cursor) -> list[Any]:
    """Parse ``{ val, val, ... }`` — Lua array-style table."""
    out: list[Any] = []
    while not cur.match(TK.RBRACE, TK.EOF):
        out.append(_parse_value(cur))
        cur.consume_if(TK.COMMA)
        cur.consume_if(TK.SEMICOLON)
    cur.expect(TK.RBRACE)
    return out


def _parse_key(cur: _Cursor) -> str:
    """Parse a Lua table key: bare ident, or ``["string"]``."""
    tok = cur.current
    if tok.kind == TK.IDENT:
        cur.advance()
        return tok.value  # type: ignore[return-value]
    if tok.kind == TK.LBRACKET:
        cur.advance()
        inner = cur.expect(TK.STRING)
        cur.expect(TK.RBRACKET)
        return str(inner.value)
    raise LuaReadError(f"Expected table key at line {tok.line}")


# ── Statement-level parser ────────────────────────────────────────────────────

# Matches:  section.entity_id = {
_ENTITY_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([a-z][a-z0-9_]*)$")

# Matches:  section.entity_id.triggers["slot"]
_TRIGGER_ASSIGN_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\.([a-z][a-z0-9_]*)\.triggers$"
)


class _LFRParser:
    """Stateful parser that walks the token stream statement-by-statement."""

    def __init__(self, source: str) -> None:
        tokens = tokenize(source)
        self._cur = _Cursor(tokens)
        # Collected state
        self._floor_table: dict[str, Any] = {}
        # section_name → {entity_id → raw dict}
        self._sections: dict[str, dict[str, dict[str, Any]]] = {}
        # (section, entity_id, slot) → trigger body source text
        self._trigger_bodies: dict[tuple[str, str, str], str] = {}
        # Preserves the original source text so we can extract function bodies
        self._source = source

    # ── Top-level parse ───────────────────────────────────────────────────────

    def parse(self) -> Floor:
        while not self._cur.match(TK.EOF):
            self._parse_statement()
        return self._build_floor()

    def _parse_statement(self) -> None:
        tok = self._cur.current

        # local <ident> = { ... }
        if tok.kind == TK.LOCAL:
            self._parse_local()
            return

        # return { ... }
        if tok.kind == TK.RETURN:
            self._parse_return()
            return

        # Assignment: section.id = { ... }
        # or trigger: section.id.triggers["slot"] = function(ctx) ... end
        if tok.kind == TK.IDENT:
            self._parse_assignment()
            return

        # Anything else (stray comments already stripped by tokenizer) → skip
        self._cur.advance()

    # ── local <name> = { ... } ────────────────────────────────────────────────

    def _parse_local(self) -> None:
        self._cur.expect(TK.LOCAL)

        # Handle ``local function name(...) ... end`` (a helper function
        # declaration used by the emitter, e.g. ``_find_entity``).
        # These are not section tables — skip the entire declaration.
        if self._cur.match(TK.FUNCTION):
            self._parse_function_body()
            return

        name_tok = self._cur.expect(TK.IDENT)
        name = str(name_tok.value)
        self._cur.expect(TK.EQUALS)

        if self._cur.match(TK.LBRACE):
            tbl = _parse_table(self._cur)
            if name == "floor":
                self._floor_table = tbl if isinstance(tbl, dict) else {}
            else:
                # Section local; ensure it exists in the registry
                if name not in self._sections:
                    self._sections[name] = {}
        else:
            # Could be something unexpected — skip value
            self._skip_value()

    # ── return { ... } ────────────────────────────────────────────────────────

    def _parse_return(self) -> None:
        self._cur.expect(TK.RETURN)
        # The return table just names locals — we've already parsed them.
        if self._cur.match(TK.LBRACE):
            _parse_table(self._cur)  # consume; ignore content

    # ── Assignment: lhs = rhs ─────────────────────────────────────────────────

    def _parse_assignment(self) -> None:
        # Collect the LHS path  (ident { .ident | [string] }*)
        lhs_parts, lhs_bracket_key = self._collect_lhs()

        # If the next token is '(' the "assignment" is actually a function-call
        # statement (e.g. ``engine.register_verb_alias(...)``) — not an
        # assignment.  Skip the argument list and return without consuming '='.
        if self._cur.match(TK.LPAREN):
            # Skip the parenthesised argument list, handling nested parens.
            self._cur.advance()  # consume '('
            depth = 1
            while depth > 0 and not self._cur.match(TK.EOF):
                if self._cur.current.kind == TK.LPAREN:
                    depth += 1
                elif self._cur.current.kind == TK.RPAREN:
                    depth -= 1
                    if depth == 0:
                        self._cur.advance()  # consume closing ')'
                        break
                self._cur.advance()
            return

        self._cur.expect(TK.EQUALS)

        # Trigger assignment: section.id.triggers["slot"] = function(ctx) ... end
        if (
            len(lhs_parts) == 3
            and lhs_parts[2] == "triggers"
            and lhs_bracket_key is not None
        ):
            section, entity_id, _ = lhs_parts
            slot = lhs_bracket_key
            body_src = self._parse_function_body()
            self._trigger_bodies[(section, entity_id, slot)] = body_src
            return

        # Entity assignment: section.id = { ... }
        if len(lhs_parts) == 2 and self._cur.match(TK.LBRACE):
            section, entity_id = lhs_parts
            tbl = _parse_table(self._cur)
            if isinstance(tbl, dict):
                if section not in self._sections:
                    self._sections[section] = {}
                self._sections[section][entity_id] = tbl
            return

        # Unknown or complex assignment — skip the RHS
        self._skip_value()

    def _collect_lhs(self) -> tuple[list[str], str | None]:
        """Parse a dotted path and return (parts, bracket_key).

        ``section.id.triggers["on:Enter"]``
            → (["section", "id", "triggers"], "on:Enter")
        ``section.id``
            → (["section", "id"], None)
        """
        parts: list[str] = []
        bracket_key: str | None = None

        # First ident
        first = self._cur.expect(TK.IDENT)
        parts.append(str(first.value))

        while True:
            if self._cur.match(TK.DOT):
                self._cur.advance()
                ident = self._cur.expect(TK.IDENT)
                parts.append(str(ident.value))
            elif self._cur.match(TK.LBRACKET):
                self._cur.advance()
                key_tok = self._cur.expect(TK.STRING)
                self._cur.expect(TK.RBRACKET)
                bracket_key = str(key_tok.value)
                break
            else:
                break

        return parts, bracket_key

    # ── Function body extraction ──────────────────────────────────────────────

    def _parse_function_body(self) -> str:
        """Parse ``function(ctx) ... end`` or ``function name(ctx) ... end``.

        Handles both anonymous function literals (``function(...)`` as table
        values or trigger RHS) and named function declarations
        (``local function name(...) ... end``).

        We track brace/end depth because the body may contain nested
        ``function``s and ``end``s.  The raw source between the outer
        ``function(...)`` and matching ``end`` is returned so that the
        trigger body is available for round-trip fidelity if needed.
        """
        cur = self._cur
        cur.expect(TK.FUNCTION)
        # Optional function name (for ``local function name(...)`` form).
        if cur.match(TK.IDENT):
            cur.advance()
        cur.expect(TK.LPAREN)
        # Consume parameter list up to RPAREN (may be 'ctx' or empty)
        while not cur.match(TK.RPAREN, TK.EOF):
            cur.advance()
        cur.expect(TK.RPAREN)

        # Now collect tokens until the matching 'end', tracking nesting.
        # We also need to reconstruct the source text.
        #
        # Block openers that require a matching 'end' in Luau:
        #   TK.FUNCTION (keyword)
        #   IDENT 'if', 'for', 'while', 'do'  (tokenized as IDENT, not keywords)
        # Block closers: TK.END (keyword)
        # 'repeat' is closed by 'until' (no 'end'), so we don't count it.
        # Track block-opening keywords that each require a matching 'end'.
        # NOTE: 'do' is intentionally excluded — when 'do' appears in a
        # 'for ... do' or 'while ... do' construct, the enclosing 'for'/'while'
        # already accounts for the depth.  A bare 'do ... end' block would need
        # tracking, but the LFR emitter never generates them.  Including 'do'
        # causes a double-count for 'for'/'while' loops.
        _BLOCK_OPENER_IDENTS = frozenset({"if", "for", "while"})

        body_tokens: list[Token] = []
        depth = 1
        while depth > 0 and not cur.match(TK.EOF):
            tok = cur.current
            if tok.kind == TK.FUNCTION:
                depth += 1
            elif tok.kind == TK.IDENT and tok.value in _BLOCK_OPENER_IDENTS:
                depth += 1
            elif tok.kind == TK.END:
                depth -= 1
                if depth == 0:
                    cur.advance()
                    break
            body_tokens.append(tok)
            cur.advance()

        # Reconstruct approximate source from tokens
        return _tokens_to_source(body_tokens)

    # ── Skip unknown RHS ──────────────────────────────────────────────────────

    def _skip_value(self) -> None:
        """Skip over an unexpected RHS value without crashing."""
        tok = self._cur.current
        if tok.kind == TK.LBRACE:
            _parse_table(self._cur)
        elif tok.kind == TK.FUNCTION:
            self._parse_function_body()
        elif tok.kind in (TK.STRING, TK.NUMBER, TK.TRUE, TK.FALSE, TK.NIL, TK.IDENT):
            self._cur.advance()
        # else skip one

    # ── Floor construction ────────────────────────────────────────────────────

    def _build_floor(self) -> Floor:
        ft = self._floor_table
        name = str(ft.get("name", ""))
        # prose may be a plain string or a LuauCode (from a compiled prose template).
        # LuauCode prose is round-tripped as-is; the emitter handles both types.
        _raw_prose = ft.get("prose")
        prose: Any = _raw_prose if isinstance(_raw_prose, LuauCode) else (str(_raw_prose) if _raw_prose is not None else "")
        imports_raw = ft.get("imports", {})
        if isinstance(imports_raw, list):
            imports = [str(i) for i in imports_raw]
        elif isinstance(imports_raw, dict):
            # Emitter always emits as a list, but handle dict form defensively
            imports = list(imports_raw.values())
        else:
            imports = []

        floor_props: dict[str, Any] = {}
        for k, v in ft.items():
            if k in ("name", "prose", "imports", "extra"):
                continue
            floor_props[k] = v

        floor = Floor(
            name=name,
            prose=prose,
            imports=imports,
            properties=floor_props,
        )

        for section_name, entity_map in self._sections.items():
            for entity_id, raw in entity_map.items():
                entity = self._build_entity(entity_id, raw, section_name)
                floor += entity

        # Attach triggers to their entities
        for (section, entity_id, slot), body_src in self._trigger_bodies.items():
            entity = floor.entities.get(entity_id)
            if entity is None:
                continue
            trigger_name = _slot_to_trigger_name(slot)
            trigger = Trigger(
                name=trigger_name,
                script=Script(language="luau", source=body_src),
            )
            entity.triggers.append(trigger)

        return floor

    def _build_entity(
        self, entity_id: str, raw: dict[str, Any], section_name: str
    ) -> FMLEntity:
        eid = str(raw.get("id", entity_id))
        kind = str(raw.get("kind", ""))
        name = str(raw.get("name", ""))
        # prose may be a plain string or a LuauCode (from a compiled prose template).
        # LuauCode prose round-trips through the emitter unchanged.
        _raw_entity_prose = raw.get("prose")
        prose: Any = _raw_entity_prose if isinstance(_raw_entity_prose, LuauCode) else (str(_raw_entity_prose) if _raw_entity_prose is not None else "")

        # links: { "id1", "id2" } → emitted as list
        links_raw = raw.get("links", {})
        if isinstance(links_raw, list):
            links = [str(l) for l in links_raw]
        elif isinstance(links_raw, dict):
            links = list(links_raw.values())
        else:
            links = []

        # properties: base dict from the 'properties' key
        props: dict[str, Any] = {}
        raw_props = raw.get("properties", {})
        if isinstance(raw_props, dict):
            props.update(raw_props)

        # exits are inlined at top level in the emitter (not inside properties)
        if "exits" in raw and "exits" not in props:
            props["exits"] = raw["exits"]

        # Sub-entities
        subentities: list[FMLEntity] = []
        raw_subs = raw.get("subentities", {})
        if isinstance(raw_subs, list):
            for sub_raw in raw_subs:
                if not isinstance(sub_raw, dict):
                    continue
                sub_id = str(sub_raw.get("id", ""))
                if not sub_id:
                    continue
                sub_entity = FMLEntity(
                    id=sub_id,
                    kind=str(sub_raw.get("kind", "")),
                    name=str(sub_raw.get("name", "")),
                    prose=sub_raw["prose"] if isinstance(sub_raw.get("prose"), LuauCode) else (str(sub_raw.get("prose", "")) if sub_raw.get("prose") else ""),
                    properties=dict(sub_raw.get("properties", {}))
                    if isinstance(sub_raw.get("properties"), dict)
                    else {},
                )
                subentities.append(sub_entity)

        return FMLEntity(
            id=eid,
            kind=kind,
            name=name,
            prose=prose,
            links=links,
            properties=props,
            subentities=subentities,
        )


# ── Token → source reconstruction ─────────────────────────────────────────────


def _tokens_to_source(tokens: list[Token]) -> str:
    """Reconstruct approximate Luau source from a token sequence.

    This is intentionally lossy regarding whitespace — the goal is to
    preserve the semantic content of the trigger body so it can be
    re-emitted as a ``luau`` script block without crashing.
    """
    parts: list[str] = []
    _tok_to_str = _make_tok_str_map()
    for tok in tokens:
        parts.append(_tok_to_str(tok))
    return " ".join(parts)


def _make_tok_str_map():
    """Return a closure that maps a single Token to its string representation."""
    _kind_map: dict[TK, str] = {
        TK.LBRACE: "{",
        TK.RBRACE: "}",
        TK.LBRACKET: "[",
        TK.RBRACKET: "]",
        TK.LPAREN: "(",
        TK.RPAREN: ")",
        TK.COMMA: ",",
        TK.EQUALS: "=",
        TK.DOT: ".",
        TK.COLON: ":",
        TK.SEMICOLON: ";",
        TK.TRUE: "true",
        TK.FALSE: "false",
        TK.NIL: "nil",
        TK.LOCAL: "local",
        TK.RETURN: "return",
        TK.FUNCTION: "function",
        TK.END: "end",
        TK.EOF: "",
    }

    def _tok_str(tok: Token) -> str:
        if tok.kind == TK.STRING:
            # Re-emit as a Lua double-quoted string
            s = str(tok.value)
            s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'"{s}"'
        if tok.kind == TK.NUMBER:
            return repr(tok.value) if isinstance(tok.value, float) else str(tok.value)
        if tok.kind == TK.IDENT:
            return str(tok.value)
        return _kind_map.get(tok.kind, "")

    return _tok_str


# ── Public LuaReader class ────────────────────────────────────────────────────


class LuaReader:
    """Static reader for Luau LFR files produced by ``emit_lua.py``.

    Usage::

        floor = LuaReader().read("path/to/floor.lua")
        floor = LuaReader().read_text(lua_source)
    """

    def read(self, path: str | Path) -> Floor:
        """Read a Luau LFR file at *path* and return a ``Floor`` model."""
        source = Path(path).read_text(encoding="utf-8")
        return self.read_text(source)

    def read_text(self, source: str) -> Floor:
        """Parse a Luau LFR string and return a ``Floor`` model.

        Raises ``LuaReadError`` on parse failures.
        """
        try:
            parser = _LFRParser(source)
            return parser.parse()
        except (TokenizeError, LuaReadError):
            raise
        except Exception as exc:
            raise LuaReadError(f"Failed to read LFR: {exc}") from exc
