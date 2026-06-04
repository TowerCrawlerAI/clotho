"""Smoke tests — verify the public API is importable and callable."""

import fml_parser
from fml_parser.emit_lua import emit_lua


def test_parse_fml_is_callable():
    assert callable(fml_parser.parse_fml)


def test_emit_fml_is_callable():
    assert callable(fml_parser.emit_fml)


def test_emit_lua_is_callable():
    assert callable(emit_lua)
