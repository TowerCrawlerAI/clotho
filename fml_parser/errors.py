"""Error hierarchy for the FML / LFR pipeline.

See `docs/design/PARSER.md` § 7 for the error model.
"""

from __future__ import annotations


class FmlError(Exception):
    """Base class for all FML / LFR errors."""


class FmlParseError(FmlError):
    """Raised by the parser when an FML document cannot be lowered."""


class FmlSyntaxError(FmlParseError):
    """Structural invariant violated (multiple H1, sub-entity outside any entity, ...)."""


class FmlImportError(FmlParseError):
    """Import cycle, missing file, or name collision between imported and local entities."""


class FmlStdlibError(FmlParseError):
    """Stdlib resolution failure or stdlib config malformed."""


class FmlValidationError(FmlParseError):
    """Cross-reference failure post-parse (dangling exit, unknown combatant id, ...)."""


class FmlPredicateError(FmlParseError):
    """Predicate mini-language parse error inside `[!complete]` / `[!discovery]` body."""


class FmlEmitError(FmlError):
    """Raised by the reverse emitter when a Floor cannot be canonicalized to FML."""


class FloorLoadError(FmlError):
    """Raised when loading an LFR module fails (missing FLOOR symbol, wrong type, ...)."""
