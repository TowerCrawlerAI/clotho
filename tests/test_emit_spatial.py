"""Tests for the §22 Phase 5 spatial authoring lowering (CoreRequirements
#104/#108): FML `position: [x, y, z]` lowers to the engine's integer cell
payload on the location link, and a container's `blocked:` list lowers to
engine.set_blocked calls.

Behavior is asserted over the emitted Lua (the binding-surface LFR the engine
loads), and malformed authoring raises FmlSyntaxError. The feature is additive:
floors without position/blocked lower exactly as before.
"""

from __future__ import annotations

import pytest

from fml_parser.emit_lua import emit_lua_graph, emit_lua_om
from fml_parser.errors import FmlSyntaxError
from fml_parser.models import FMLEntity, Floor
from fml_parser.parser import parse_fml


def make_floor(entities: list[FMLEntity], **floor_props) -> Floor:
    f = Floor(name="Test Floor", properties=floor_props)
    for ent in entities:
        f += ent
    return f


def make_entity(eid, name, kind, properties=None):
    return FMLEntity(
        id=eid, name=name, kind=kind,
        properties=properties or {}, triggers=[], kind_chain=[kind],
    )


# ─── position → relate-with-cell (#104) ──────────────────────────────────────


def test_position_rides_the_location_relate():
    room = make_entity("arena", "Arena", "room")
    goblin = make_entity("goblin", "Goblin", "actor",
                         properties={"location": "arena", "position": [3, 4, 0]})
    out = emit_lua_graph(make_floor([room, goblin], start_location="arena"))
    assert 'engine.relate("in", n_goblin, n_arena, 3, 4, 0)' in out
    # No bare (cell-less) relate for the same entity.
    assert 'engine.relate("in", n_goblin, n_arena)' not in out


def test_negative_and_zero_coords_lower_verbatim():
    room = make_entity("arena", "Arena", "room")
    ent = make_entity("e", "E", "actor",
                      properties={"location": "arena", "position": [-2, 0, 5]})
    out = emit_lua_graph(make_floor([room, ent], start_location="arena"))
    assert 'engine.relate("in", n_e, n_arena, -2, 0, 5)' in out


def test_no_position_emits_a_plain_relate():
    # Additive: an entity without a position lowers exactly as before.
    room = make_entity("arena", "Arena", "room")
    ent = make_entity("e", "E", "item", properties={"location": "arena"})
    out = emit_lua_graph(make_floor([room, ent], start_location="arena"))
    assert 'engine.relate("in", n_e, n_arena)' in out
    assert "n_e, n_arena," not in out  # no trailing cell args


def test_position_works_in_om_mode_too():
    room = make_entity("arena", "Arena", "room")
    ent = make_entity("e", "E", "actor",
                      properties={"location": "arena", "position": [1, 2, 3]})
    out = emit_lua_om(make_floor([room, ent], start_location="arena"))
    assert 'engine.relate("in", n_e, n_arena, 1, 2, 3)' in out


@pytest.mark.parametrize("bad", [
    [1, 2],            # too few
    [1, 2, 3, 4],      # too many
    [1, 2, "x"],       # non-integer
    [1.5, 2, 3],       # float
    "3, 4, 0",         # not a list
    [1, True, 3],      # bool is not an int coord
])
def test_malformed_position_raises(bad):
    room = make_entity("arena", "Arena", "room")
    ent = make_entity("e", "E", "actor",
                      properties={"location": "arena", "position": bad})
    with pytest.raises(FmlSyntaxError):
        emit_lua_graph(make_floor([room, ent], start_location="arena"))


def test_malformed_position_is_validated_even_without_a_container():
    # A malformed position must be reported even when the entity has no
    # recognized container (it would otherwise be a silent no-op).
    ent = make_entity("e", "E", "actor", properties={"position": [1, 2]})
    with pytest.raises(FmlSyntaxError):
        emit_lua_graph(make_floor([ent], start_location="e"))


def test_position_out_of_range_raises():
    room = make_entity("arena", "Arena", "room")
    ent = make_entity("e", "E", "actor",
                      properties={"location": "arena", "position": [9_999_999, 0, 0]})
    with pytest.raises(FmlSyntaxError):
        emit_lua_graph(make_floor([room, ent], start_location="arena"))


# ─── blocked → set_blocked (#108) ────────────────────────────────────────────


def test_blocked_cells_lower_to_set_blocked():
    room = make_entity("arena", "Arena", "room",
                       properties={"blocked": [[1, 0, 0], [1, 1, 0]]})
    out = emit_lua_graph(make_floor([room], start_location="arena"))
    # Default kind = solid wall → no flags argument.
    assert "engine.set_blocked(n_arena, 1, 0, 0)" in out
    assert "engine.set_blocked(n_arena, 1, 1, 0)" in out


def test_blocked_cell_kinds_map_to_flags():
    room = make_entity("arena", "Arena", "room",
                       properties={"blocked": [
                           [3, 2, 0, "sight"],   # cover → flags 2
                           [5, 3, 0, "move"],    # obstacle → flags 1
                           [1, 0, 0, "wall"],    # explicit wall → flags 3 (no arg)
                       ]})
    out = emit_lua_graph(make_floor([room], start_location="arena"))
    assert "engine.set_blocked(n_arena, 3, 2, 0, 2)" in out
    assert "engine.set_blocked(n_arena, 5, 3, 0, 1)" in out
    assert "engine.set_blocked(n_arena, 1, 0, 0)" in out


def test_no_blocked_emits_no_set_blocked():
    room = make_entity("arena", "Arena", "room")
    out = emit_lua_graph(make_floor([room], start_location="arena"))
    assert "set_blocked" not in out


@pytest.mark.parametrize("bad", [
    [[1, 2]],              # cell too short
    [[1, 2, 3, "x", 4]],   # cell too long
    [[1, 2, "z"]],         # non-integer coord
    [[1, 2, 3, "bogus"]],  # unknown kind
    "1,0,0",               # not a list of cells
    [[1.0, 2, 3]],         # float coord
])
def test_malformed_blocked_raises(bad):
    room = make_entity("arena", "Arena", "room", properties={"blocked": bad})
    with pytest.raises(FmlSyntaxError):
        emit_lua_graph(make_floor([room], start_location="arena"))


def test_spatial_properties_parse_from_real_fml_text():
    # Guards the authored FML forms end-to-end through the real parser: the
    # single-line bracket-list value parses (a nested bullet list would NOT),
    # and `blocked` with a `kind` survives to set_blocked flags.
    txt = (
        "# Test Floor\n\n"
        "- start_location: arena\n\n"
        "### Arena\n\n"
        "- blocked: [[1, 0, 0], [3, 2, 0, sight]]\n\n"
        "> A bare arena.\n\n"
        "### Goblin\n\n"
        "- kind: npc\n"
        "- location: arena\n"
        "- position: [3, 4, 0]\n\n"
        "> A goblin.\n"
    )
    floor = parse_fml(txt)
    assert floor.entities["goblin"].properties["position"] == [3, 4, 0]
    assert floor.entities["arena"].properties["blocked"] == [[1, 0, 0], [3, 2, 0, "sight"]]
    out = emit_lua_graph(floor)
    assert 'engine.relate("in", n_goblin, n_arena, 3, 4, 0)' in out
    assert "engine.set_blocked(n_arena, 1, 0, 0)" in out
    assert "engine.set_blocked(n_arena, 3, 2, 0, 2)" in out


def test_position_and_blocked_are_not_emitted_as_scalar_props():
    # The spatial keys must not leak into the node's property table as set_prop.
    room = make_entity("arena", "Arena", "room",
                       properties={"blocked": [[1, 0, 0]]})
    goblin = make_entity("g", "G", "actor",
                         properties={"location": "arena", "position": [1, 1, 0]})
    out = emit_lua_graph(make_floor([room, goblin], start_location="arena"))
    assert "position" not in out
    assert 'set_prop(n_arena, "blocked"' not in out
