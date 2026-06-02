"""FML ⇄ LFR translation pipeline.

Public API:

- `parse_fml(text, source_path)` — FML text → `Floor` model.
- `emit_fml(floor)` — `Floor` model → canonical FML text.
- `slugify(heading)` — convert a heading to an `EntityId`.

See `docs/design/PARSER.md` and `docs/design/LFR.md`.

Note (post-pivot, Decisions.md §13): the Python LFR emitter / loader
(`lfr_emitter.py`, `loader.py`, `bundle.py`, `bootstrap_template.py`) has
been deleted. LFR now targets Luau (`.lua`). The new emitter lands in
ticket P1; the engine loads `.lua` LFR directly via FFI in P2+.
"""

from __future__ import annotations

from .emitter import emit_fml
from .errors import (
    FloorLoadError,
    FmlEmitError,
    FmlError,
    FmlImportError,
    FmlParseError,
    FmlPredicateError,
    FmlStdlibError,
    FmlSyntaxError,
    FmlValidationError,
)
from .models import (
    ActionLine,
    And,
    At,
    BareLink,
    BreakStmt,
    ContextBundle,
    ContinueStmt,
    DialogueSuccess,
    ElseIfClause,
    EntityId,
    Entered,
    Examined,
    FMLEntity,
    Flag,
    Floor,
    Found,
    Has,
    IfBlock,
    Killed,
    LoopNBlock,
    LoopThroughBlock,
    LoopUntilBlock,
    Not,
    Or,
    Predicate,
    ProseChunk,
    Script,
    Trigger,
    TriggerBodyItem,
    UnderstandDirective,
    Visited,
    WhileBlock,
)
from .parser import parse_fml
from .slugify import slugify

__all__ = [
    # Parsing / emission
    "parse_fml",
    "emit_fml",
    "slugify",
    # Models
    "EntityId",
    "Script",
    "Trigger",
    "TriggerBodyItem",
    "BareLink",
    "ActionLine",
    "BreakStmt",
    "ContinueStmt",
    "ElseIfClause",
    "IfBlock",
    "WhileBlock",
    "LoopNBlock",
    "LoopThroughBlock",
    "LoopUntilBlock",
    "FMLEntity",
    "Floor",
    "ProseChunk",
    "ContextBundle",
    "UnderstandDirective",
    # Predicates
    "Predicate",
    "Killed",
    "Flag",
    "Found",
    "DialogueSuccess",
    "Entered",
    "Has",
    "At",
    "Examined",
    "Visited",
    "And",
    "Or",
    "Not",
    # Errors
    "FmlError",
    "FmlParseError",
    "FmlSyntaxError",
    "FmlImportError",
    "FmlStdlibError",
    "FmlValidationError",
    "FmlPredicateError",
    "FmlEmitError",
    "FloorLoadError",
]
