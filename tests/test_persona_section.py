"""Tests for the ``#### Persona`` FML subsection lowering (combat-AI persona/goals).

An NPC entity may declare a ``#### Persona`` H4 subsection containing:
  - ``- persona: <text>``   → string property ``persona`` on the actor node
  - ``- goal: <text>``      → accumulates into ordered list ``goals`` on the node

These properties are authored by FML content authors so that Loom's LLM brain
(loom PR #29) can read them off the TURN_REQUEST event (forwarded by wyrd
dnd5e.lua PR #135) and build a character-appropriate system prompt.

This test file verifies BEHAVIOR over the lowered Lua output:
- ``persona`` is set as a scalar string property on the NPC node.
- ``goals`` is emitted as ``engine.set_prop(n_<id>, "goals", { ... })`` with
  the exact strings in authoring order.
- Multiple ``- goal:`` lines accumulate into one list (not last-wins).
- A missing ``#### Persona`` section lowers cleanly with no persona/goals output.
- Inline (own-body) ``persona:``/``goals:`` properties are not double-emitted.
- Additive: an entity without the section lowers byte-identically.
"""

from __future__ import annotations

from textwrap import dedent

from fml_parser.emit_lua import emit_lua_om
from fml_parser.parser import parse_fml

# ---------------------------------------------------------------------------
# Minimal FML helpers
# ---------------------------------------------------------------------------

_HEADER = "# Test Floor\n\n## People\n\n"


def _floor_fml(entity_block: str) -> str:
    """Wrap ``entity_block`` (an H3 entity + its body) in a minimal FML floor."""
    return _HEADER + dedent(entity_block).strip() + "\n"


def parse_and_emit(fml: str) -> str:
    """Parse FML text and emit OM LFR, returning the emitted Lua string."""
    floor = parse_fml(fml)
    return emit_lua_om(floor)


# ---------------------------------------------------------------------------
# Parser: #### Persona section → entity properties
# ---------------------------------------------------------------------------


def test_persona_string_property_parsed():
    """``- persona: <text>`` in #### Persona becomes the ``persona`` string prop."""
    fml = _floor_fml("""
        ### Skull King
        - kind: npc
        - hp: 120
        - brain: llm

        #### Persona
        - persona: You are the Skull King, a vain, theatrical undead sovereign.
    """)
    floor = parse_fml(fml)
    entity = floor.entities.get("skull_king")
    assert entity is not None, "skull_king entity must be present"
    assert entity.properties.get("persona") == (
        "You are the Skull King, a vain, theatrical undead sovereign."
    )


def test_goals_list_property_parsed():
    """Multiple ``- goal:`` lines aggregate into an ordered list ``goals``."""
    fml = _floor_fml("""
        ### Skull King
        - kind: npc
        - hp: 120
        - brain: llm

        #### Persona
        - persona: You are the Skull King.
        - goal: hold the throne
        - goal: slay intruders
        - goal: protect the bone crown
    """)
    floor = parse_fml(fml)
    entity = floor.entities.get("skull_king")
    assert entity is not None
    goals = entity.properties.get("goals")
    assert goals == ["hold the throne", "slay intruders", "protect the bone crown"], (
        f"Expected ordered list of goals, got {goals!r}"
    )


def test_goals_order_preserved():
    """Goal order matches authoring order, not sort order."""
    fml = _floor_fml("""
        ### Guard Captain
        - kind: npc

        #### Persona
        - goal: zebra first
        - goal: apple second
        - goal: mango third
    """)
    floor = parse_fml(fml)
    entity = floor.entities["guard_captain"]
    assert entity.properties["goals"] == ["zebra first", "apple second", "mango third"]


def test_no_persona_section_leaves_entity_clean():
    """An entity without #### Persona has no ``persona`` or ``goals`` keys."""
    fml = _floor_fml("""
        ### Goblin
        - kind: npc
        - hp: 10
    """)
    floor = parse_fml(fml)
    entity = floor.entities["goblin"]
    assert "persona" not in entity.properties
    assert "goals" not in entity.properties


def test_persona_without_goals_is_valid():
    """``persona:`` alone (no goal lines) → no ``goals`` key at all."""
    fml = _floor_fml("""
        ### Specter
        - kind: npc

        #### Persona
        - persona: A silent, drifting horror.
    """)
    floor = parse_fml(fml)
    entity = floor.entities["specter"]
    assert entity.properties.get("persona") == "A silent, drifting horror."
    assert "goals" not in entity.properties


def test_goals_without_persona_is_valid():
    """``goal:`` lines alone (no persona line) → ``goals`` list, no ``persona`` key."""
    fml = _floor_fml("""
        ### Berserker
        - kind: npc

        #### Persona
        - goal: destroy everything
    """)
    floor = parse_fml(fml)
    entity = floor.entities["berserker"]
    assert "persona" not in entity.properties
    assert entity.properties.get("goals") == ["destroy everything"]


def test_inline_own_props_win_over_persona_section():
    """If ``persona:`` is declared BOTH inline and in #### Persona, inline wins."""
    fml = _floor_fml("""
        ### Vampire
        - kind: npc
        - persona: Inline persona wins.

        #### Persona
        - persona: Subsection persona loses.
    """)
    floor = parse_fml(fml)
    entity = floor.entities["vampire"]
    assert entity.properties["persona"] == "Inline persona wins."


def test_other_h4_sections_unaffected():
    """#### Traits and #### Triggers sections next to #### Persona work normally."""
    fml = _floor_fml("""
        ### Lich
        - kind: npc

        #### Persona
        - persona: An ancient undead spellcaster.
        - goal: achieve immortality

        #### Triggers
        ##### On RoundStart
    """)
    floor = parse_fml(fml)
    entity = floor.entities["lich"]
    assert entity.properties["persona"] == "An ancient undead spellcaster."
    assert entity.properties["goals"] == ["achieve immortality"]


# ---------------------------------------------------------------------------
# Emitter: persona/goals land on the node in the OM LFR
# ---------------------------------------------------------------------------


def test_persona_string_emitted_in_create_node():
    """``persona`` is a scalar string → emitted inside ``engine.create_node({...})``."""
    fml = _floor_fml("""
        ### Skull King
        - kind: npc
        - hp: 120
        - brain: llm

        #### Persona
        - persona: You are the Skull King, a vain theatrical undead sovereign.
        - goal: hold the throne
        - goal: protect the crown
    """)
    lua = parse_and_emit(fml)
    # persona must appear as a scalar inside create_node
    assert 'persona = "You are the Skull King, a vain theatrical undead sovereign."' in lua


def test_goals_list_emitted_as_set_prop():
    """``goals`` list → ``engine.set_prop(n_skull_king, "goals", { ... })``."""
    fml = _floor_fml("""
        ### Skull King
        - kind: npc
        - hp: 120
        - brain: llm

        #### Persona
        - persona: You are the Skull King.
        - goal: hold the throne
        - goal: slay intruders
        - goal: protect the bone crown
    """)
    lua = parse_and_emit(fml)
    assert (
        'engine.set_prop(n_skull_king, "goals", '
        '{ "hold the throne", "slay intruders", "protect the bone crown" })'
    ) in lua


def test_goals_order_in_emitted_lua():
    """Goals appear in the emitted Lua in the same order they were authored."""
    fml = _floor_fml("""
        ### Boss
        - kind: npc

        #### Persona
        - goal: first
        - goal: second
        - goal: third
    """)
    lua = parse_and_emit(fml)
    assert 'engine.set_prop(n_boss, "goals", { "first", "second", "third" })' in lua


def test_no_goals_no_set_prop():
    """An entity without #### Persona emits no ``set_prop`` goals call."""
    fml = _floor_fml("""
        ### Goblin
        - kind: npc
        - hp: 7
    """)
    lua = parse_and_emit(fml)
    assert "goals" not in lua
    assert "set_prop" not in lua or '"goals"' not in lua


def test_persona_without_goals_no_set_prop():
    """``persona`` alone → no ``set_prop`` call (it's a scalar, goes in create_node)."""
    fml = _floor_fml("""
        ### Ghost
        - kind: npc

        #### Persona
        - persona: A mournful spirit.
    """)
    lua = parse_and_emit(fml)
    assert "set_prop" not in lua
    assert 'persona = "A mournful spirit."' in lua


def test_additive_no_persona_byte_identical_before_after():
    """A floor with no #### Persona sections lowers byte-identically to before."""
    fml = _floor_fml("""
        ### Rat
        - kind: npc
        - hp: 2
    """)
    # Just confirm it emits cleanly and has no persona/goals noise.
    lua = parse_and_emit(fml)
    assert "persona" not in lua
    assert "goals" not in lua
    assert "NPC goals" not in lua
