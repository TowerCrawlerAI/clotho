"""Minimal Luau tokenizer for the LFR table-literal subset.

Produces a flat list of ``Token`` objects consumed by the parser.
Only the token types needed to read ``emit_lua.py`` output are supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


class TK(Enum):
    """Token kinds."""
    # Literals
    STRING = auto()    # "..." or [[...]]
    NUMBER = auto()    # integer or float
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    # Structure
    LBRACE = auto()    # {
    RBRACE = auto()    # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LPAREN = auto()    # (
    RPAREN = auto()    # )
    COMMA = auto()     # ,
    EQUALS = auto()    # =
    DOT = auto()       # .
    COLON = auto()     # :
    SEMICOLON = auto() # ;
    # Keywords / identifiers
    IDENT = auto()     # bare identifier
    LOCAL = auto()
    RETURN = auto()
    FUNCTION = auto()
    END = auto()
    # End of file
    EOF = auto()


@dataclass(slots=True)
class Token:
    kind: TK
    value: object  # str for STRING/IDENT, int/float for NUMBER, None otherwise
    line: int


class TokenizeError(ValueError):
    pass


# ── Patterns ──────────────────────────────────────────────────────────────────

_KEYWORDS: dict[str, TK] = {
    "true":     TK.TRUE,
    "false":    TK.FALSE,
    "nil":      TK.NIL,
    "local":    TK.LOCAL,
    "return":   TK.RETURN,
    "function": TK.FUNCTION,
    "end":      TK.END,
}

_SIMPLE: dict[str, TK] = {
    "{": TK.LBRACE,
    "}": TK.RBRACE,
    "[": TK.LBRACKET,
    "]": TK.RBRACKET,
    "(": TK.LPAREN,
    ")": TK.RPAREN,
    ",": TK.COMMA,
    "=": TK.EQUALS,
    ".": TK.DOT,
    ":": TK.COLON,
    ";": TK.SEMICOLON,
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INT_RE   = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?")


def tokenize(source: str) -> list[Token]:
    """Tokenize *source* into a flat ``Token`` list (including EOF)."""
    tokens: list[Token] = []
    pos = 0
    length = len(source)
    line = 1

    while pos < length:
        ch = source[pos]

        # Newline tracking
        if ch == "\n":
            line += 1
            pos += 1
            continue

        # Whitespace (not newline)
        if ch in " \t\r":
            pos += 1
            continue

        # Line comment  --  (skip to end of line)
        if source[pos:pos+2] == "--" and source[pos:pos+4] != "--[[":
            end = source.find("\n", pos)
            if end == -1:
                break
            pos = end
            continue

        # Long comments  --[[ ... ]]
        if source[pos:pos+4] == "--[[":
            end = source.find("]]", pos + 4)
            if end == -1:
                raise TokenizeError(f"Unterminated long comment at line {line}")
            line += source[pos:end + 2].count("\n")
            pos = end + 2
            continue

        # Long string  [[...]]
        if source[pos:pos+2] == "[[":
            end = source.find("]]", pos + 2)
            if end == -1:
                raise TokenizeError(f"Unterminated long string at line {line}")
            value = source[pos + 2: end]
            line += value.count("\n")
            tokens.append(Token(TK.STRING, value, line))
            pos = end + 2
            continue

        # Double-quoted string  "..."
        if ch == '"':
            result, new_pos, new_lines = _read_dqstring(source, pos, line)
            tokens.append(Token(TK.STRING, result, line))
            pos = new_pos
            line += new_lines
            continue

        # Single-quoted string  '...'
        if ch == "'":
            result, new_pos, new_lines = _read_sqstring(source, pos, line)
            tokens.append(Token(TK.STRING, result, line))
            pos = new_pos
            line += new_lines
            continue

        # Simple single-char tokens
        if ch in _SIMPLE:
            # Disambiguate '==' — the emitter never emits it but guard anyway;
            # a bare '=' is assignment, '==' would be comparison (not needed).
            tokens.append(Token(_SIMPLE[ch], None, line))
            pos += 1
            continue

        # Number (float before int to avoid mis-parse)
        m = _FLOAT_RE.match(source, pos)
        if m:
            tokens.append(Token(TK.NUMBER, float(m.group()), line))
            pos = m.end()
            continue
        m = _INT_RE.match(source, pos)
        if m:
            tokens.append(Token(TK.NUMBER, int(m.group()), line))
            pos = m.end()
            continue

        # Identifier or keyword
        m = _IDENT_RE.match(source, pos)
        if m:
            word = m.group()
            kind = _KEYWORDS.get(word, TK.IDENT)
            tokens.append(Token(kind, word if kind == TK.IDENT else None, line))
            pos = m.end()
            continue

        raise TokenizeError(f"Unexpected character {ch!r} at line {line}")

    tokens.append(Token(TK.EOF, None, line))
    return tokens


def _read_dqstring(source: str, pos: int, line: int) -> tuple[str, int, int]:
    """Parse a double-quoted Lua string; return (value, next_pos, added_lines)."""
    pos += 1  # skip opening "
    chars: list[str] = []
    added_lines = 0
    while pos < len(source):
        ch = source[pos]
        if ch == '"':
            pos += 1
            return "".join(chars), pos, added_lines
        if ch == "\\":
            pos += 1
            esc = source[pos]
            if esc == "n":
                chars.append("\n")
            elif esc == "t":
                chars.append("\t")
            elif esc == "r":
                chars.append("\r")
            elif esc == "\\":
                chars.append("\\")
            elif esc == '"':
                chars.append('"')
            elif esc == "'":
                chars.append("'")
            elif esc == "\n":
                added_lines += 1
                chars.append("\n")
            else:
                chars.append(esc)
            pos += 1
            continue
        if ch == "\n":
            added_lines += 1
        chars.append(ch)
        pos += 1
    raise TokenizeError("Unterminated string literal")


def _read_sqstring(source: str, pos: int, line: int) -> tuple[str, int, int]:
    """Parse a single-quoted Lua string; return (value, next_pos, added_lines)."""
    pos += 1
    chars: list[str] = []
    added_lines = 0
    while pos < len(source):
        ch = source[pos]
        if ch == "'":
            pos += 1
            return "".join(chars), pos, added_lines
        if ch == "\\":
            pos += 1
            esc = source[pos]
            if esc == "n":
                chars.append("\n")
            elif esc == "t":
                chars.append("\t")
            elif esc == "r":
                chars.append("\r")
            elif esc == "\\":
                chars.append("\\")
            elif esc == "'":
                chars.append("'")
            elif esc == '"':
                chars.append('"')
            elif esc == "\n":
                added_lines += 1
                chars.append("\n")
            else:
                chars.append(esc)
            pos += 1
            continue
        if ch == "\n":
            added_lines += 1
        chars.append(ch)
        pos += 1
    raise TokenizeError("Unterminated string literal")
