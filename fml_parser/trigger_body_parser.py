"""Form A control-flow parser for FML trigger bodies.

Parses the body of an H6 trigger block into a list of `TriggerBodyItem` AST
nodes.  The body is a sequence of raw text lines (already extracted from the
markdown token stream before this module is called).

Supported body content:

- ``- when: <predicate>`` — sets the trigger guard; consumed before body parse.
- Bare Markdown links: ``[text](target)`` on a line by itself → ``BareLink``.
- Form A block keywords (bold, case-sensitive):

    ``**If** <cond>:``       → ``IfBlock``
    ``**Else if** <cond>:``  → ``ElseIfClause`` (chained onto current IfBlock)
    ``**Else:**``            → else branch of current IfBlock
    ``**End.**``             → closes innermost open block
    ``**While** <cond>:``   → ``WhileBlock``
    ``**Loop** N times:``   → ``LoopNBlock``
    ``**Loop through** <coll>:`` → ``LoopThroughBlock``
    ``**Loop until** <cond> (max N):`` → ``LoopUntilBlock``
    ``**Break.**``           → ``BreakStmt``
    ``**Continue.**``        → ``ContinueStmt``

- Any other non-empty line → ``ActionLine(verb=first_word, raw=full_line)``.

Empty lines are silently skipped.  Indentation (2 spaces per nesting level) is
used only for readability and is stripped before keyword matching.

See ``docs/design/FML_EXTENSIONS.md`` §4 for the full spec.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from .models import (
    ActionLine,
    BareLink,
    BreakStmt,
    ContinueStmt,
    ElseIfClause,
    IfBlock,
    LoopNBlock,
    LoopThroughBlock,
    LoopUntilBlock,
    OutputLine,
    PropertySet,
    TriggerBodyItem,
    WhileBlock,
)

# ─── Regexes ──────────────────────────────────────────────────────────────────

_BARE_LINK_RE = re.compile(r"^\[([^\]]*)\]\(([^)]+)\)\s*$")
# Match `- when: <pred>` (raw line from source) OR `when: <pred>` (inline token
# content extracted by markdown-it after stripping the bullet prefix)
_WHEN_RE = re.compile(r"^-?\s*when:\s*(.+)$")

# Form A keyword patterns — match after stripping leading whitespace
_IF_RE = re.compile(r"^\*\*If\*\*\s+(.+):\s*$")
_ELSEIF_RE = re.compile(r"^\*\*Else if\*\*\s+(.+):\s*$")
_ELSE_RE = re.compile(r"^\*\*Else:\*\*\s*$")
_END_RE = re.compile(r"^\*\*End\.\*\*\s*$")
_WHILE_RE = re.compile(r"^\*\*While\*\*\s+(.+):\s*$")
_LOOP_N_RE = re.compile(r"^\*\*Loop\*\*\s+(\d+)\s+times:\s*$")
_LOOP_THROUGH_RE = re.compile(r"^\*\*Loop through\*\*\s+(.+):\s*$")
_LOOP_UNTIL_RE = re.compile(r"^\*\*Loop until\*\*\s+(.+?)(?:\s+\(max\s+(\d+)\))?:\s*$")
_BREAK_RE = re.compile(r"^\*\*Break\.\*\*\s*$")
_CONTINUE_RE = re.compile(r"^\*\*Continue\.\*\*\s*$")

# Native FML syntax patterns
# Property setter: *entity.prop*: value  (extracted from `- *entity.prop*: value` list items)
_PROPERTY_SET_RE = re.compile(r"^\*([^*]+)\*:\s*(.+)$")
# Output line: > text with optional *path* substitutions
_OUTPUT_LINE_RE = re.compile(r"^>\s+(.+)$")


# ─── Value parsing helpers ────────────────────────────────────────────────────


def _parse_value(raw: str) -> Any:
    """Parse a FML property-setter value string to a Python value."""
    s = raw.strip()
    if s == "true":
        return True
    if s == "false":
        return False
    if s in ("nil", "null", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s


def _format_value(v: Any) -> str:
    """Format a Python value back to FML property-setter syntax."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return "nil"
    if isinstance(v, str):
        return v
    return str(v)


# ─── Public API ───────────────────────────────────────────────────────────────


def parse_trigger_body(
    lines: list[str],
) -> tuple[str | None, list[TriggerBodyItem]]:
    """Parse a list of raw text lines into (when_guard, body_items).

    ``when_guard`` is the raw predicate string from ``- when: <pred>`` if
    present, else ``None``.  The ``- when:`` line is consumed and does not
    appear in ``body_items``.

    ``body_items`` is the parsed sequence of ``TriggerBodyItem`` nodes.
    """
    # Extract optional `- when:` guard first (may appear before body items)
    when_guard: str | None = None
    remaining: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and when_guard is None:
            m = _WHEN_RE.match(stripped)
            if m:
                when_guard = m.group(1).strip()
                continue
        remaining.append(line)

    body = _parse_body_lines(remaining)
    return when_guard, body


def format_trigger_body(
    when: str | None,
    body: list[TriggerBodyItem],
    indent: int = 0,
) -> list[str]:
    """Emit trigger body lines for the emitter.

    Returns a list of strings (no trailing newlines).  ``indent`` controls the
    base indentation level for Form A blocks (0 = top-level trigger body).
    """
    lines: list[str] = []
    if when is not None:
        lines.append(f"- when: {when}")
        lines.append("")
    for item in body:
        lines.extend(_emit_item(item, indent=indent))
    return lines


# ─── Private: parsing ─────────────────────────────────────────────────────────


def _parse_body_lines(lines: list[str]) -> list[TriggerBodyItem]:
    """Top-level Form A parser: returns a flat list of top-level body items.

    Uses a block-stack to handle nesting.  The stack tracks open block frames;
    each frame has a target list to append items into and a kind tag for
    validation (``"if"``, ``"while"``, ``"loop_n"``, ``"loop_through"``,
    ``"loop_until"``).
    """
    # Stack entries: (kind_tag, block_object, target_list_ref, elseif_list_ref)
    # For IfBlock: target_list_ref starts as the block's body; transitions to
    # else_body when **Else:** is encountered.
    # elseif_list_ref: the block's else_if_clauses list (only meaningful for if).
    stack: list[_StackFrame] = []
    result: list[TriggerBodyItem] = []

    def _current_target() -> list[TriggerBodyItem]:
        if stack:
            return stack[-1].target
        return result

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Form A keywords — check before bare-link/action to avoid ambiguity
        m: Any

        m = _IF_RE.match(stripped)
        if m:
            block = IfBlock(condition=m.group(1).strip())
            _current_target().append(block)
            stack.append(_StackFrame("if", block, block.body, block.else_if_clauses))
            continue

        m = _ELSEIF_RE.match(stripped)
        if m:
            if not stack or stack[-1].kind != "if":
                warnings.warn(
                    f"FML trigger: **Else if** outside an **If** block (line: {stripped!r})",
                    stacklevel=2,
                )
                continue
            frame = stack[-1]
            clause = ElseIfClause(condition=m.group(1).strip())
            frame.block.else_if_clauses.append(clause)  # type: ignore[union-attr]
            frame.target = clause.body
            continue

        m = _ELSE_RE.match(stripped)
        if m:
            if not stack or stack[-1].kind != "if":
                warnings.warn(
                    f"FML trigger: **Else:** outside an **If** block (line: {stripped!r})",
                    stacklevel=2,
                )
                continue
            frame = stack[-1]
            frame.block.else_body = []  # type: ignore[union-attr]
            frame.target = frame.block.else_body  # type: ignore[union-attr]
            continue

        m = _END_RE.match(stripped)
        if m:
            if not stack:
                warnings.warn(
                    "FML trigger: **End.** with no open block",
                    stacklevel=2,
                )
            else:
                stack.pop()
            continue

        m = _WHILE_RE.match(stripped)
        if m:
            block = WhileBlock(condition=m.group(1).strip())
            _current_target().append(block)
            stack.append(_StackFrame("while", block, block.body, None))
            continue

        m = _LOOP_N_RE.match(stripped)
        if m:
            block = LoopNBlock(count=int(m.group(1)))
            _current_target().append(block)
            stack.append(_StackFrame("loop_n", block, block.body, None))
            continue

        m = _LOOP_THROUGH_RE.match(stripped)
        if m:
            block = LoopThroughBlock(collection=m.group(1).strip())
            _current_target().append(block)
            stack.append(_StackFrame("loop_through", block, block.body, None))
            continue

        m = _LOOP_UNTIL_RE.match(stripped)
        if m:
            condition = m.group(1).strip()
            max_iter = int(m.group(2)) if m.group(2) else 10000
            block = LoopUntilBlock(condition=condition, max_iterations=max_iter)
            _current_target().append(block)
            stack.append(_StackFrame("loop_until", block, block.body, None))
            continue

        m = _BREAK_RE.match(stripped)
        if m:
            _current_target().append(BreakStmt())
            continue

        m = _CONTINUE_RE.match(stripped)
        if m:
            _current_target().append(ContinueStmt())
            continue

        # Output line: > text with optional *path* substitutions
        m = _OUTPUT_LINE_RE.match(stripped)
        if m:
            _current_target().append(OutputLine(template=m.group(1)))
            continue

        # Property setter: *entity.prop*: value  (from `- *entity.prop*: value` list items)
        m = _PROPERTY_SET_RE.match(stripped)
        if m:
            _current_target().append(PropertySet(path=m.group(1), value=_parse_value(m.group(2))))
            continue

        # Bare Markdown link: [text](target)
        m = _BARE_LINK_RE.match(stripped)
        if m:
            _current_target().append(BareLink(text=m.group(1), target=m.group(2)))
            continue

        # Anything else is a plain action line
        if stripped:
            first_word = stripped.split()[0]
            _current_target().append(ActionLine(verb=first_word, raw=stripped))

    if stack:
        warnings.warn(
            f"FML trigger: {len(stack)} unclosed Form A block(s) at end of trigger body",
            stacklevel=2,
        )

    return result


class _StackFrame:
    """Mutable frame for a single open Form A block."""

    __slots__ = ("kind", "block", "target", "elseif_clauses")

    def __init__(
        self,
        kind: str,
        block: Any,
        target: list[TriggerBodyItem],
        elseif_clauses: Any,
    ) -> None:
        self.kind = kind
        self.block = block
        self.target = target
        self.elseif_clauses = elseif_clauses


# ─── Private: emission ────────────────────────────────────────────────────────


def _emit_item(item: TriggerBodyItem, indent: int) -> list[str]:
    """Emit one TriggerBodyItem as a list of lines (no trailing newline)."""
    prefix = "  " * indent

    if item.kind == "output_line":
        return [f"{prefix}> {item.template}"]  # type: ignore[union-attr]

    if item.kind == "property_set":
        return [f"{prefix}- *{item.path}*: {_format_value(item.value)}"]  # type: ignore[union-attr]

    if item.kind == "bare_link":
        return [f"{prefix}[{item.text}]({item.target})"]  # type: ignore[union-attr]

    if item.kind == "action":
        return [f"{prefix}{item.raw}"]  # type: ignore[union-attr]

    if item.kind == "break":
        return [f"{prefix}**Break.**"]

    if item.kind == "continue":
        return [f"{prefix}**Continue.**"]

    if item.kind == "if":
        return _emit_if(item, indent)  # type: ignore[arg-type]

    if item.kind == "while":
        return _emit_while(item, indent)  # type: ignore[arg-type]

    if item.kind == "loop_n":
        return _emit_loop_n(item, indent)  # type: ignore[arg-type]

    if item.kind == "loop_through":
        return _emit_loop_through(item, indent)  # type: ignore[arg-type]

    if item.kind == "loop_until":
        return _emit_loop_until(item, indent)  # type: ignore[arg-type]

    # Fallback
    return [f"{prefix}{item!r}"]


def _emit_if(block: IfBlock, indent: int) -> list[str]:
    prefix = "  " * indent
    lines = [f"{prefix}**If** {block.condition}:"]
    for child in block.body:
        lines.extend(_emit_item(child, indent + 1))
    for clause in block.else_if_clauses:
        lines.append(f"{prefix}**Else if** {clause.condition}:")
        for child in clause.body:
            lines.extend(_emit_item(child, indent + 1))
    if block.else_body is not None:
        lines.append(f"{prefix}**Else:**")
        for child in block.else_body:
            lines.extend(_emit_item(child, indent + 1))
    lines.append(f"{prefix}**End.**")
    return lines


def _emit_while(block: WhileBlock, indent: int) -> list[str]:
    prefix = "  " * indent
    lines = [f"{prefix}**While** {block.condition}:"]
    for child in block.body:
        lines.extend(_emit_item(child, indent + 1))
    lines.append(f"{prefix}**End.**")
    return lines


def _emit_loop_n(block: LoopNBlock, indent: int) -> list[str]:
    prefix = "  " * indent
    lines = [f"{prefix}**Loop** {block.count} times:"]
    for child in block.body:
        lines.extend(_emit_item(child, indent + 1))
    lines.append(f"{prefix}**End.**")
    return lines


def _emit_loop_through(block: LoopThroughBlock, indent: int) -> list[str]:
    prefix = "  " * indent
    lines = [f"{prefix}**Loop through** {block.collection}:"]
    for child in block.body:
        lines.extend(_emit_item(child, indent + 1))
    lines.append(f"{prefix}**End.**")
    return lines


def _emit_loop_until(block: LoopUntilBlock, indent: int) -> list[str]:
    prefix = "  " * indent
    if block.max_iterations != 10000:
        header = f"{prefix}**Loop until** {block.condition} (max {block.max_iterations}):"
    else:
        header = f"{prefix}**Loop until** {block.condition}:"
    lines = [header]
    for child in block.body:
        lines.extend(_emit_item(child, indent + 1))
    lines.append(f"{prefix}**End.**")
    return lines
