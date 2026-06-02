"""Predicate mini-language used inside FML `[!complete]` / `[!discovery]` admonitions.

Grammar (see `docs/design/LFR.md` § Predicate mini-language):

    predicate   := disjunction
    disjunction := conjunction ( "or" conjunction )*
    conjunction := negation ( "and" negation )*
    negation    := "not" atom | atom
    atom        := op_call
                 | "(" predicate ")"
    op_call     := identifier "(" arg_list ")"
    arg_list    := identifier ("," identifier)*
    identifier  := [a-z_][a-z0-9_]*

Keywords are case-insensitive: `AND`/`and`, `OR`/`or`, `NOT`/`not`.
Atomic ops: `killed`, `flag`, `found`, `dialogue_success`, `entered`,
            `has`, `at`, `examined`, `visited`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import FmlPredicateError
from .models import (
    And,
    At,
    DialogueSuccess,
    Entered,
    Examined,
    Flag,
    Found,
    Has,
    Killed,
    Not,
    Or,
    Predicate,
    Visited,
)

# ─── Tokenizer ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _Token:
    kind: str  # "ident" | "lparen" | "rparen" | "and" | "or" | "not" | "comma" | "eof"
    value: str
    pos: int


_IDENT_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_OP_NAMES = {
    "killed",
    "flag",
    "found",
    "dialogue_success",
    "entered",
    "has",
    "at",
    "examined",
    "visited",
}


def _tokenize(s: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            tokens.append(_Token("lparen", "(", i))
            i += 1
            continue
        if c == ")":
            tokens.append(_Token("rparen", ")", i))
            i += 1
            continue
        if c == ",":
            tokens.append(_Token("comma", ",", i))
            i += 1
            continue
        m = _IDENT_RE.match(s, i)
        if m:
            raw = m.group()
            upper = raw.upper()
            if upper == "AND" and (m.end() == n or not s[m.end()].isalnum()):
                tokens.append(_Token("and", raw, i))
                i = m.end()
                continue
            if upper == "OR" and (m.end() == n or not s[m.end()].isalnum()):
                tokens.append(_Token("or", raw, i))
                i = m.end()
                continue
            if upper == "NOT" and (m.end() == n or not s[m.end()].isalnum()):
                tokens.append(_Token("not", raw, i))
                i = m.end()
                continue
            tokens.append(_Token("ident", raw.lower(), i))
            i = m.end()
            continue
        raise FmlPredicateError(
            f"unexpected character {c!r} at position {i} in predicate {s!r}"
        )
    tokens.append(_Token("eof", "", n))
    return tokens


# ─── Parser ───────────────────────────────────────────────────────────────────


class _Parser:
    def __init__(self, tokens: list[_Token], source: str):
        self.tokens = tokens
        self.source = source
        self.pos = 0

    @property
    def cur(self) -> _Token:
        return self.tokens[self.pos]

    def consume(self, kind: str) -> _Token:
        if self.cur.kind != kind:
            raise FmlPredicateError(
                f"expected {kind} but got {self.cur.kind}={self.cur.value!r} "
                f"at position {self.cur.pos} in predicate {self.source!r}"
            )
        tok = self.cur
        self.pos += 1
        return tok

    def parse(self) -> Predicate:
        p = self._disjunction()
        if self.cur.kind != "eof":
            raise FmlPredicateError(
                f"unexpected trailing token {self.cur.value!r} "
                f"at position {self.cur.pos} in predicate {self.source!r}"
            )
        return p

    def _disjunction(self) -> Predicate:
        left = self._conjunction()
        while self.cur.kind == "or":
            self.consume("or")
            right = self._conjunction()
            left = Or(left=left, right=right)
        return left

    def _conjunction(self) -> Predicate:
        left = self._atom()
        while self.cur.kind == "and":
            self.consume("and")
            right = self._atom()
            left = And(left=left, right=right)
        return left

    def _atom(self) -> Predicate:
        if self.cur.kind == "not":
            self.consume("not")
            if self.cur.kind == "lparen":
                self.consume("lparen")
                inner = self._disjunction()
                self.consume("rparen")
            else:
                inner = self._op_call()
            return Not(inner=inner)
        if self.cur.kind == "lparen":
            self.consume("lparen")
            inner = self._disjunction()
            self.consume("rparen")
            return inner
        if self.cur.kind == "ident":
            return self._op_call()
        raise FmlPredicateError(
            f"expected NOT, '(' or identifier but got {self.cur.kind}="
            f"{self.cur.value!r} at position {self.cur.pos} in predicate {self.source!r}"
        )

    def _op_call(self) -> Predicate:
        op_tok = self.consume("ident")
        op = op_tok.value
        if op not in _OP_NAMES:
            raise FmlPredicateError(
                f"unknown predicate op {op!r} (allowed: {sorted(_OP_NAMES)})"
            )
        self.consume("lparen")
        arg1_tok = self.consume("ident")
        arg2: str | None = None
        if self.cur.kind == "comma":
            self.consume("comma")
            arg2_tok = self.consume("ident")
            arg2 = arg2_tok.value
        self.consume("rparen")
        return _make_op(op, arg1_tok.value, arg2)


def _make_op(op: str, arg: str, arg2: str | None = None) -> Predicate:
    match op:
        case "killed":
            return Killed(target=arg)
        case "flag":
            return Flag(name=arg)
        case "found":
            return Found(target=arg)
        case "dialogue_success":
            return DialogueSuccess(target=arg)
        case "entered":
            return Entered(target=arg)
        case "has":
            if arg2 is None:
                raise FmlPredicateError("has() requires two arguments: has(actor, item)")
            return Has(actor=arg, item=arg2)
        case "at":
            if arg2 is None:
                raise FmlPredicateError("at() requires two arguments: at(entity, place)")
            return At(entity=arg, place=arg2)
        case "examined":
            return Examined(noun=arg, by=arg2 if arg2 is not None else "player")
        case "visited":
            return Visited(place=arg)
    raise FmlPredicateError(f"internal: unhandled op {op!r}")  # pragma: no cover


def parse_predicate(s: str) -> Predicate:
    """Parse a predicate mini-language string into a Predicate tree."""
    tokens = _tokenize(s.strip())
    return _Parser(tokens, s).parse()


parse_trigger_guard = parse_predicate


# ─── Writer ───────────────────────────────────────────────────────────────────


def render_predicate(p: Predicate) -> str:
    """Render a Predicate tree as canonical mini-language.

    Canonical form: minimum-parentheses, single spaces around operators,
    atomic ops in source-order from the AST.
    """
    return _render(p, parent_prec=0)


# Precedence: OR < AND < NOT/atom. Higher precedence binds tighter.
_PREC_OR = 1
_PREC_AND = 2
_PREC_ATOM = 3


def _render(p: Predicate, parent_prec: int) -> str:
    match p:
        case Killed(target=t):
            return f"killed({t})"
        case Flag(name=n):
            return f"flag({n})"
        case Found(target=t):
            return f"found({t})"
        case DialogueSuccess(target=t):
            return f"dialogue_success({t})"
        case Entered(target=t):
            return f"entered({t})"
        case Has(actor=a, item=item):
            return f"has({a}, {item})"
        case At(entity=e, place=pl):
            return f"at({e}, {pl})"
        case Examined(noun=n, by=by):
            if by == "player":
                return f"examined({n})"
            return f"examined({n}, {by})"
        case Visited(place=pl):
            return f"visited({pl})"
        case Not(inner=inner):
            return f"NOT({_render(inner, _PREC_OR)})"
        case And(left=left, right=right):
            s = f"{_render(left, _PREC_AND)} AND {_render(right, _PREC_AND)}"
            return f"({s})" if parent_prec > _PREC_AND else s
        case Or(left=left, right=right):
            s = f"{_render(left, _PREC_OR)} OR {_render(right, _PREC_OR)}"
            return f"({s})" if parent_prec > _PREC_OR else s
    raise FmlPredicateError(  # pragma: no cover
        f"internal: cannot render predicate of type {type(p).__name__}"
    )
