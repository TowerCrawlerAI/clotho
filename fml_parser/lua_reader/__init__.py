"""Luau LFR static reader — parse a .lua LFR file without executing it.

Public API:
    LuaReader — the single entry point.

Usage::

    from parser.lua_reader import LuaReader
    floor = LuaReader().read("path/to/floor.lua")
    floor = LuaReader().read_text(lua_source_string)

The reader parses the subset of Luau table-literal syntax emitted by
``emit_lua.py`` and reconstructs a ``Floor`` Pydantic model.  Trigger
function bodies are captured as opaque ``Script(language='luau', ...)``
strings — no Luau execution occurs.
"""

from .reader import LuaReader

__all__ = ["LuaReader"]
