"""Dice expression detection and FML property-value lowering.

Implements the six FML property-value forms for dice/Luau values:

    Form 1 (bare dice):      str: 3d6+2
        → resolve at parse time via SeededRng(seed=0); store integer literal.

    Form 2 (backtick function): str: `function(self, ctx) ... end`
        → store Luau code verbatim as a LuauCode(source).

    Form 3 (backtick literal):  str: `13`
        → strip backticks; parse inner value as a scalar; store the result.

    Form 4 (bare literal):      str: 13
        → store as literal (existing _parse_scalar behavior, unchanged).

    Form 5 (backtick dice thunk): str: `3d6+2`
        → wrap in a Luau function that calls engine.roll("3d6+2"); store LuauCode.

    Form 6 (backtick invoked):  str: `(3d6+2)()`
        → expression is invoked at parse time; evaluate the dice expr and store
          integer literal.  The `()` suffix signals parse-time evaluation.

Single rule:
    bare form   → parse-time evaluation (dice) or literal (non-dice).
    backtick    → Luau code (thunk) by default; immediate via explicit `()`.

The `DICE_RE` regex matches the internal grammar of TowerAI's C engine dice
parser (dice.h) — intentionally minimal for v0.1:

    count? 'd' sides modifier?
    count    := [1-9][0-9]{0,3}   (optional; 1-1000)
    sides    := [1-9][0-9]{0,8}   (1-10^9)
    modifier := [+-][0-9]{1,9}    (optional)

This is the exact grammar accepted by `tower_dice_eval` in `src/engine/src/dice.c`.
Do not extend the regex to accept kh/kl/reroll — those belong in Luau stdlib.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


# ─── Dice expression regex ────────────────────────────────────────────────────

# Matches the full dice expression string (anchored). Case-insensitive so `D`
# is accepted alongside `d`.
DICE_RE = re.compile(
    r"^"
    r"(?P<count>[1-9][0-9]{0,3})?"   # optional count (1-9999, but engine caps at 1000)
    r"[dD]"                            # 'd' separator
    r"(?P<sides>[1-9][0-9]{0,8})"     # sides (1-9999999999)
    r"(?P<mod>[+-][0-9]{1,9})?"        # optional modifier
    r"$"
)


def is_dice_expr(s: str) -> bool:
    """Return True if `s` matches the engine's dice grammar exactly."""
    return DICE_RE.fullmatch(s.strip()) is not None


# ─── LuauCode value wrapper ───────────────────────────────────────────────────

@dataclass(frozen=True)
class LuauCode:
    """A property value that is Luau source code, not a scalar.

    The emitter recognises this type in `_lua_prop_value` and emits the source
    verbatim (no quoting), so the resulting LFR file contains live Luau code at
    the property slot.

    `source` must be a complete, syntactically valid Luau expression or function
    literal — the emitter does not validate it.
    """

    source: str

    def __repr__(self) -> str:
        return f"LuauCode({self.source!r})"


@dataclass(frozen=True)
class ProseValue:
    """A prose-typed property value — the raw prose text before lowering.

    Prose values are demarcated by the ``> `` prefix in FML (single-line form)
    or the ``: |`` + ``> `` multi-line block form.  They lower to Luau
    ``function(self, ctx) ... end`` closures via ``_compile_prose()`` in
    ``emit_lua.py``.

    ``lines`` holds the prose lines in source order, each stripped of its
    leading ``> `` prefix.  For a single-line property (``- key: > text``)
    there is exactly one line.

    ``source_path`` and ``source_line`` carry the FML location for debug
    comments in the emitted Luau.
    """

    lines: tuple[str, ...]
    source_path: str = ""
    source_line: int = 0

    def __repr__(self) -> str:
        return f"ProseValue({self.lines!r})"

    def __str__(self) -> str:
        """Return the prose content as a plain string.

        Lines are joined with newlines. Empty lines (paragraph separators)
        are preserved as blank lines between paragraphs.

        This makes existing ``str(prose)`` consumers (pydantic ``str`` field
        coercion, chronicler/narrator prompt injection) transparently work —
        they receive the prose TEXT, not the FML-formatted ``> `` prefixed
        version. The result is what the player would read.

        Callers needing the FML source form (for round-trip via ``tower
        unlower``) should use the ``_emit_prose`` function in
        ``parser.emitter``, which handles ``ProseValue`` directly.
        """
        return "\n".join(self.lines)

    def __contains__(self, item: object) -> bool:
        """Support ``'substring' in prose_value`` by delegating to ``str(self)``.

        Existing tests that do ``assert "text" in entity.prose`` continue to
        work transparently — they check against the raw FML source text
        (with ``> `` prefixes included).
        """
        return item in str(self)

    def strip(self, chars: str | None = None) -> str:
        """Return ``str(self).strip(chars)``.

        Allows existing code that calls ``entity.prose.strip()`` to work
        without modification — returns the FML source text (``> line`` form)
        with leading/trailing whitespace stripped.
        """
        return str(self).strip(chars)


# ─── Parse-time dice evaluation ──────────────────────────────────────────────

def _eval_dice_expr(expr: str, rng: random.Random) -> int:
    """Parse and evaluate a dice expression using a Python RNG.

    Implements the same grammar as TowerAI's C dice parser (dice.h) so the
    Python host can evaluate bare dice at parse time without shelling out to
    the engine.  This is only used for Form 1 and Form 6.

    Raises ValueError on any parse failure (consistent with _parse_scalar).
    """
    m = DICE_RE.fullmatch(expr.strip())
    if m is None:
        raise ValueError(f"Not a dice expression: {expr!r}")

    count = int(m.group("count") or "1")
    sides = int(m.group("sides"))
    mod_str = m.group("mod") or "0"
    modifier = int(mod_str) if mod_str else 0

    if count <= 0 or sides <= 0:
        raise ValueError(f"Invalid dice expression: {expr!r}")
    if count > 1000:
        raise ValueError(f"Dice count exceeds cap (1000): {expr!r}")
    if sides > 1_000_000_000:
        raise ValueError(f"Die sides exceeds cap: {expr!r}")

    total = sum(rng.randint(1, sides) for _ in range(count)) + modifier
    return total


# ─── Scalar fallback (avoids circular import with parser.py) ─────────────────

def _parse_scalar_inner(s: str) -> object:
    """Minimal scalar parser for backtick inner values (Forms 3 and 6 fallback).

    Handles the common cases without importing parser._parse_scalar, avoiding
    a circular dependency.  Supports: quoted strings, bool, int, float, bare str.
    Does NOT handle bracket-lists or Markdown links — those don't appear inside
    backtick values in practice.
    """
    s = s.strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
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


# ─── Backtick value classifier ────────────────────────────────────────────────

def _strip_backticks(s: str) -> str:
    """Return the content between the outer backticks."""
    return s[1:-1]


def _is_invoked(inner: str) -> bool:
    """True if `inner` ends with `()`, meaning parse-time evaluation is requested."""
    stripped = inner.strip()
    return stripped.endswith("()")


def _unwrap_invoked(inner: str) -> str:
    """Strip the trailing `()` and optional outer parentheses from an invoked expression.

    ``(3d6+2)()`` → unwrap ``()`` → ``(3d6+2)`` → strip outer parens → ``3d6+2``.
    ``3d6+2()``   → unwrap ``()`` → ``3d6+2`` (already clean).
    """
    stripped = inner.strip()
    assert stripped.endswith("()")
    expr = stripped[:-2].strip()
    # Strip one layer of outer parentheses if present.
    if expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1].strip()
    return expr


def _is_function_literal(inner: str) -> bool:
    """True if `inner` begins with `function`."""
    return inner.strip().startswith("function")


# ─── Public API ──────────────────────────────────────────────────────────────

def parse_dice_value(raw: str, rng: random.Random | None = None) -> object:
    """Parse a raw property-value string under the six-form dice/Luau contract.

    `raw` is the text AFTER `key:` in an FML property bullet, already stripped
    of leading/trailing whitespace.

    `rng` is the parse-time randomness source.  Callers that want deterministic
    parse-time dice should pass a seeded `random.Random(seed)`.  If `rng` is
    None, a default `random.Random(0)` is used so bare dice are still resolved.

    Returns one of:
        - `int`      — a resolved literal (Forms 1, 4 literal int, Form 6)
        - `float`    — a resolved float literal (Form 4 float)
        - `bool`     — True or False (Form 4)
        - `str`      — a bare string literal (Form 4)
        - `LuauCode` — Luau source code (Forms 2, 5)

    This function does NOT handle nested lists or dicts — those are handled
    by `_parse_bracket_list` in parser.py and passed through unchanged.
    """
    if rng is None:
        rng = random.Random(0)

    s = raw.strip()

    # ── Backtick forms (Forms 2, 3, 5, 6) ────────────────────────────────────
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        inner = _strip_backticks(s)

        # Form 6: backtick + explicit invocation `(expr)()`
        if _is_invoked(inner):
            expr_body = _unwrap_invoked(inner)
            if is_dice_expr(expr_body):
                return _eval_dice_expr(expr_body, rng)
            # Non-dice invoked: parse inner as scalar.
            return _parse_scalar_inner(expr_body)

        # Form 2: backtick function literal
        if _is_function_literal(inner):
            return LuauCode(source=inner)

        # Form 5: backtick dice expression → Luau thunk
        if is_dice_expr(inner):
            return LuauCode(
                source=f'function(self, ctx) return engine.roll("{inner}") end'
            )

        # Form 3: backtick literal → parse inner as scalar
        return _parse_scalar_inner(inner)

    # ── Bare dice (Form 1) ────────────────────────────────────────────────────
    if is_dice_expr(s):
        return _eval_dice_expr(s, rng)

    # ── Form 4: bare literal — caller handles via _parse_scalar ──────────────
    # This path is reached when parse_dice_value is called directly in tests;
    # in production, _parse_scalar delegates dice/backtick detection here and
    # handles bare literals itself.
    return _parse_scalar_inner(s)
