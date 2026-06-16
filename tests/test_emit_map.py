"""Tests for the VTT map pipeline (M1–M6 requirements).

Tests cover:
  M1 — `lower --map` emits map.json alongside the floor output.
  M2 — The auto-layout is deterministic: same FML in → byte-identical map.json.
  M3 — Rooms partition by `level`; up/down exits lower as stairs.
  M4 — `map:` / `token:` reserved keys parse, feed map emitter, are stripped
       from the LFR, and round-trip through unlower.
  M5 — Direction-inconsistent, non-geometric, and one-way edges lower as
       warp/implicit; synthetic fixtures assert each edge case.
  M6 — Golden Bone Garden map: 3 planes, cardinal placement, stairs on the
       ossuary↔catacombs link, warp-with-door on tunnel→crypt.

Vendored fixture rationale (M6):
  The Bone Garden fixture is vendored in tests/fixtures/bone_garden/ rather
  than loading the live sample-dungeon sibling repo.  This keeps clotho's
  test suite self-contained and runnable without the sibling repo being
  present.  The reverse-dependency CI gate (clotho lowers the real
  sample-dungeon before tagging) is what catches any drift in the live floor.
  The vendored copy strips non-room entities and stdlib imports so the parse
  is lightweight, while preserving the room graph and exit structure exactly
  as needed to exercise the map pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fml_parser.emit_lua import emit_lua_om
from fml_parser.emit_map import emit_map, emit_map_json, strip_map_keys
from fml_parser.models import FMLEntity, Floor
from fml_parser.parser import parse_fml

# ── Helpers ──────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BONE_GARDEN_DIR = FIXTURES_DIR / "bone_garden"
GOLDEN_MAP_PATH = BONE_GARDEN_DIR / "golden_map.json"


def _make_room(rid: str, name: str, exits: dict, level: int = 0) -> FMLEntity:
    return FMLEntity(
        id=rid,
        name=name,
        kind="room",
        properties={"exits": exits, "level": level} if level != 0 else {"exits": exits},
        kind_chain=["room"],
    )


def _make_floor(*rooms: FMLEntity, start: str | None = None) -> Floor:
    props = {}
    if start:
        props["start_location"] = start
    f = Floor(name="Test Floor", properties=props)
    for r in rooms:
        f += r
    return f


def _parse_fixture(index_md: Path) -> Floor:
    text = index_md.read_text(encoding="utf-8")
    return parse_fml(text, source_path=index_md)


def _emit_map_for(floor: Floor) -> dict:
    """Emit map (no stripping needed for these lightweight test floors)."""
    lua = emit_lua_om(floor, source_path="test.md")
    return emit_map(floor, lua.encode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# M4 — map:/token: reserved keys
# ─────────────────────────────────────────────────────────────────────────────


def test_m4_map_keys_stripped_from_lua():
    """map: and token: properties must be absent from the emitted floor.lua."""
    room = FMLEntity(
        id="hall",
        name="Hall",
        kind="room",
        properties={
            "exits": {"south": "entry"},
            "map": {"x": 10, "y": 2, "width": 6, "height": 4,
                    "image": "https://cdn.example.com/hall.png"},
        },
        kind_chain=["room"],
    )
    actor = FMLEntity(
        id="skull_king",
        name="Skull King",
        kind="npc",
        properties={"token": "https://cdn.example.com/skull_king.png"},
        kind_chain=["npc"],
    )
    entry = FMLEntity(
        id="entry",
        name="Entry",
        kind="room",
        properties={"exits": {"north": "hall"}},
        kind_chain=["room"],
    )
    floor_map_defaults = {"cell_size": 32, "room_width": 4, "room_height": 4,
                         "palette": "bone_garden"}
    f = Floor(name="Test", properties={"map": floor_map_defaults,
                                        "start_location": "entry"})
    f += room
    f += actor
    f += entry

    # Before stripping: keys are present.
    assert "map" in f.properties
    assert "map" in room.properties
    assert "token" in actor.properties

    # Emit map BEFORE stripping (needs the keys).
    lua_bytes = emit_lua_om(f, source_path="test.md").encode("utf-8")
    map_data = emit_map(f, lua_bytes)

    # Strip map/token keys.
    strip_map_keys(f)

    # After stripping: keys are absent from floor and all entities.
    assert "map" not in f.properties
    assert "map" not in room.properties
    assert "token" not in actor.properties

    # Emit Lua AFTER stripping: must not contain map/token data.
    lua_source = emit_lua_om(f, source_path="test.md")
    assert "map" not in lua_source or "-- map" not in lua_source
    # The map property value must not appear in the Lua output.
    assert "https://cdn.example.com/hall.png" not in lua_source
    assert "https://cdn.example.com/skull_king.png" not in lua_source
    assert "bone_garden" not in lua_source

    # But the map emitter still captured the right data before stripping.
    assert map_data["grid"]["cell_px"] == 32
    assert map_data["rooms"]["hall"]["art"] == {
        "src": "https://cdn.example.com/hall.png", "fit": "cover"
    }
    assert map_data["rooms"]["hall"]["rect"]["w"] == 6
    assert map_data["rooms"]["hall"]["rect"]["h"] == 4
    assert map_data["rooms"]["hall"]["placement"] == "pinned"
    assert map_data["tokens"]["art"]["skull_king"] == {
        "src": "https://cdn.example.com/skull_king.png"
    }
    # Per-room palette: floor-level `palette: bone_garden` propagates to rooms
    # that don't have an explicit per-room palette override.
    assert map_data["rooms"]["hall"]["style"]["palette"] == "bone_garden"
    assert map_data["rooms"]["entry"]["style"]["palette"] == "bone_garden"


def test_m4_lua_emitter_unchanged_without_map():
    """When --map is NOT passed, the Lua output must be byte-identical.

    This test verifies additive/opt-in: strip_map_keys on a floor with no
    map/token keys is a no-op, and the Lua bytes don't change.
    """
    room = FMLEntity(
        id="cell",
        name="Cell",
        kind="room",
        properties={"exits": {"north": "hall"}},
        kind_chain=["room"],
    )
    hall = FMLEntity(
        id="hall",
        name="Hall",
        kind="room",
        properties={"exits": {"south": "cell"}},
        kind_chain=["room"],
    )
    f = Floor(name="Plain", properties={"start_location": "cell"})
    f += room
    f += hall

    lua_before = emit_lua_om(f, source_path="test.md")
    strip_map_keys(f)  # no-op on a floor with no map keys
    lua_after = emit_lua_om(f, source_path="test.md")

    assert lua_before == lua_after, "strip_map_keys must not alter Lua output on a map-key-free floor"


# ─────────────────────────────────────────────────────────────────────────────
# Render hints — `render:` reserved key (VTT presentation, e.g. hide_arrows)
# ─────────────────────────────────────────────────────────────────────────────


def test_render_hints_flow_to_map_and_stripped_from_lua():
    """A room's `render:` dict is emitted into its map.json room record and
    stripped from the LFR; a room without `render:` carries no render field."""
    tunnel = FMLEntity(
        id="tunnel",
        name="Twisting Tunnel",
        kind="room",
        properties={
            "exits": {"south": "hall"},
            "render": {"hide_arrows": True},
        },
        kind_chain=["room"],
    )
    hall = FMLEntity(
        id="hall",
        name="Hall",
        kind="room",
        properties={"exits": {"north": "tunnel"}},
        kind_chain=["room"],
    )
    f = Floor(name="Test", properties={"start_location": "hall"})
    f += tunnel
    f += hall

    # Emit map BEFORE stripping (needs the key).
    lua_bytes = emit_lua_om(f, source_path="test.md").encode("utf-8")
    map_data = emit_map(f, lua_bytes)

    # The hint flows through verbatim onto the room record.
    assert map_data["rooms"]["tunnel"]["render"] == {"hide_arrows": True}
    # A room with no `render:` carries no render field (additive / opt-in).
    assert "render" not in map_data["rooms"]["hall"]

    # Strip presentation keys, then re-emit the LFR.
    strip_map_keys(f)
    assert "render" not in tunnel.properties

    lua_source = emit_lua_om(f, source_path="test.md")
    # The hint must not leak into the engine LFR.
    assert "hide_arrows" not in lua_source


def test_render_hints_absent_keeps_lua_byte_identical():
    """strip_map_keys is a no-op on a floor with no render/map/token keys."""
    a = _make_room("a", "A", {"north": "b"})
    b = _make_room("b", "B", {"south": "a"})
    f = _make_floor(a, b, start="a")

    lua_before = emit_lua_om(f, source_path="test.md")
    strip_map_keys(f)
    lua_after = emit_lua_om(f, source_path="test.md")
    assert lua_before == lua_after


# ─────────────────────────────────────────────────────────────────────────────
# M2 — Determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_m2_determinism_simple():
    """Same Floor → byte-identical map.json on two successive calls."""
    a = _make_room("a", "A", {"north": "b"})
    b = _make_room("b", "B", {"south": "a"})
    f = _make_floor(a, b, start="a")

    lua1 = emit_lua_om(f, source_path="test.md")
    data1 = emit_map_json(f, lua1.encode())

    # Calling again must produce the exact same bytes.
    lua2 = emit_lua_om(f, source_path="test.md")
    data2 = emit_map_json(f, lua2.encode())

    assert data1 == data2, "emit_map_json must be deterministic"


def test_m2_determinism_bone_garden():
    """Bone Garden vendored fixture: lower twice, assert byte-identical output."""
    floor = _parse_fixture(BONE_GARDEN_DIR / "index.md")
    # Stable RELATIVE source_path: the floor.lua header (and thus floor_sha)
    # must not depend on the absolute checkout location, or the golden diverges
    # between a dev machine and CI. See CLAUDE.md "bit-identical lowering".
    lua = emit_lua_om(floor, source_path="bone_garden/index.md")
    lua_bytes = lua.encode("utf-8")

    out1 = emit_map_json(floor, lua_bytes)
    out2 = emit_map_json(floor, lua_bytes)

    assert out1 == out2, "Golden Bone Garden map must be byte-identical across runs"


# ─────────────────────────────────────────────────────────────────────────────
# M3 — Level planes + stairs
# ─────────────────────────────────────────────────────────────────────────────


def test_m3_level_partition():
    """Rooms partition by `level` property into separate planes."""
    surface = _make_room("hall", "Hall", {"down": "dungeon"}, level=0)
    underground = _make_room("dungeon", "Dungeon", {"up": "hall"}, level=-1)
    f = _make_floor(surface, underground, start="hall")
    data = _emit_map_for(f)

    assert data["rooms"]["hall"]["level"] == 0
    assert data["rooms"]["dungeon"]["level"] == -1

    # Levels array must contain both planes, sorted descending (surface first).
    level_nums = [l["level"] for l in data["levels"]]
    assert 0 in level_nums
    assert -1 in level_nums
    assert level_nums.index(0) < level_nums.index(-1)


def test_m3_up_down_exits_become_stairs():
    """up/down exits must produce stairs connections."""
    upper = _make_room("upper", "Upper", {"down": "lower"}, level=0)
    lower = _make_room("lower", "Lower", {"up": "upper"}, level=-1)
    f = _make_floor(upper, lower, start="upper")
    data = _emit_map_for(f)

    stairs = [c for c in data["connections"] if c["kind"] == "stairs"]
    assert len(stairs) >= 1, "Expected at least one stairs connection"

    # Each room emits its own outbound exit so both directions appear.
    # Assert both upper→lower and lower→upper are present.
    upper_down = next(
        (c for c in stairs if c["from"] == "upper" and c["to"] == "lower"), None
    )
    lower_up = next(
        (c for c in stairs if c["from"] == "lower" and c["to"] == "upper"), None
    )
    assert upper_down is not None, "Expected upper→lower stairs connection"
    assert lower_up is not None, "Expected lower→upper stairs connection"
    assert upper_down["one_way"] is False, "Bidirectional up/down link should be one_way: false"
    assert lower_up["one_way"] is False, "Bidirectional up/down link should be one_way: false"
    assert upper_down["path"] == [], "stairs connections have no path"
    assert lower_up["path"] == [], "stairs connections have no path"


def test_m3_cross_level_cardinal_becomes_warp():
    """A cardinal exit crossing a level boundary lowers as warp, not corridor.

    Both directions are emitted (each room's own outbound exit); check that
    all connections between the pair are warp.
    """
    surface = _make_room("surface", "Surface", {"east": "deeper"}, level=0)
    deeper = _make_room("deeper", "Deeper", {"west": "surface"}, level=-1)
    f = _make_floor(surface, deeper, start="surface")
    data = _emit_map_for(f)

    # Find the cross-level connection between surface and deeper.
    cross_level = [
        c for c in data["connections"]
        if {c["from"], c["to"]} == {"surface", "deeper"}
    ]
    assert len(cross_level) >= 1, "Expected a connection between surface and deeper"
    assert all(c["kind"] == "warp" for c in cross_level), (
        f"All cross-level connections must be warp; got {[c['kind'] for c in cross_level]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# M5 — warp/implicit/one_way synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────


def test_m5_non_euclidean_cycle():
    """Non-Euclidean: A→B(north), B→C(north), C→A(north) — a directional
    inconsistency; two edges must be warp (can't embed three northward steps
    in a loop without the third contradicting the layout).

    The first BFS placement wins; subsequent directions that disagree with the
    existing layout become warp.
    """
    a = _make_room("room_a", "Room A", {"north": "room_b"})
    b = _make_room("room_b", "Room B", {"north": "room_c", "south": "room_a"})
    c = _make_room("room_c", "Room C", {"north": "room_a", "south": "room_b"})
    f = _make_floor(a, b, c, start="room_a")
    data = _emit_map_for(f)

    conns_by = {(c["from"], c["dir"]): c for c in data["connections"]}
    # The A→B north and B→C north should be placed as corridor/implicit.
    # The C→A north is the contradiction — must be warp.
    a_north = conns_by.get(("room_a", "north"))
    b_north = conns_by.get(("room_b", "north"))
    c_north = conns_by.get(("room_c", "north"))

    assert a_north is not None, "A north→B connection must be present"
    assert b_north is not None, "B north→C connection must be present"
    assert c_north is not None, "C north→A connection must be present"

    # The first two are geometrically consistent; the closing edge can't be.
    assert a_north["kind"] in ("corridor", "implicit", "door"), \
        f"A→B north should be geometric, got {a_north['kind']}"
    assert b_north["kind"] in ("corridor", "implicit", "door"), \
        f"B→C north should be geometric, got {b_north['kind']}"
    assert c_north["kind"] == "warp", \
        f"C→A north is direction-inconsistent, expected warp, got {c_north['kind']}"


def test_m5_one_way_passage():
    """One-way passage: A→B(north) exists but B has no south exit back.
    The connection must carry one_way: true.
    """
    a = _make_room("trap", "Trap Room", {"north": "exit_room"})
    b = _make_room("exit_room", "Exit Room", {})  # no south exit back
    f = _make_floor(a, b, start="trap")
    data = _emit_map_for(f)

    conn = next(
        (c for c in data["connections"]
         if c["from"] == "trap" and c["dir"] == "north"),
        None,
    )
    assert conn is not None, "One-way north connection must be present"
    assert conn["one_way"] is True, "One-way passage must have one_way: true"

    # Verify there is NO south→trap connection from exit_room.
    reverse = next(
        (c for c in data["connections"]
         if c["from"] == "exit_room" and c["to"] == "trap"),
        None,
    )
    assert reverse is None, "Exit room has no south exit; reverse must not appear"


def test_m5_unit_cell_collision():
    """Collision: two rooms BFS to the same unit cell; the later one shifts.

    Setup: Hub in the middle. North→A, North→B both point north but that's
    impossible from the same source in a single BFS.  Instead: A is north of
    Hub, and C is also north of Hub via a different path (south of C → Hub).
    The collision is detected and resolved by inserting a row/column split.
    """
    hub = _make_room("hub", "Hub", {"north": "room_a", "east": "room_c"})
    a = _make_room("room_a", "Room A", {"south": "hub"})
    # C also claims hub is to its west AND A claims hub is to its south.
    c = _make_room("room_c", "Room C", {"west": "hub"})
    # Add D north of C to force a potential collision.
    d = _make_room("room_d", "Room D", {"south": "room_c", "east": "room_a"})
    # room_d east → room_a: room_a is already at (hub_x, hub_y-1);
    # room_d should be at (room_c_x, room_c_y-1) = (hub_x+1, hub_y-1).
    # room_a is at (hub_x, hub_y-1). Room_d east would place room_a at
    # (hub_x+2, hub_y-1) — but room_a is already placed → warp.
    f = _make_floor(hub, a, c, d, start="hub")
    data = _emit_map_for(f)

    # All four rooms must appear in the map.
    assert "hub" in data["rooms"]
    assert "room_a" in data["rooms"]
    assert "room_c" in data["rooms"]
    assert "room_d" in data["rooms"]

    # All positions must be non-negative.
    for rid, room in data["rooms"].items():
        rect = room["rect"]
        assert rect["x"] >= 0, f"Room {rid} has negative x: {rect['x']}"
        assert rect["y"] >= 0, f"Room {rid} has negative y: {rect['y']}"

    # No two rooms in the same level should share the same (x, y).
    by_level: dict[int, dict[tuple, str]] = {}
    for rid, room in data["rooms"].items():
        lvl = room["level"]
        pos = (room["rect"]["x"], room["rect"]["y"])
        if lvl not in by_level:
            by_level[lvl] = {}
        assert pos not in by_level[lvl], \
            f"Rooms {rid} and {by_level[lvl][pos]} share position {pos} on level {lvl}"
        by_level[lvl][pos] = rid


def test_m5_multi_level_floor():
    """Multi-level: surface + dungeon level; stairs between them."""
    surface = _make_room("throne", "Throne Room", {"down": "vault"}, level=0)
    vault = _make_room("vault", "Vault", {"up": "throne"}, level=-1)
    f = _make_floor(surface, vault, start="throne")
    data = _emit_map_for(f)

    level_nums = {l["level"] for l in data["levels"]}
    assert 0 in level_nums
    assert -1 in level_nums

    stairs = [c for c in data["connections"] if c["kind"] == "stairs"]
    assert len(stairs) >= 1, "Must have stairs between levels"

    # Both directions are emitted (each room's outbound exit).
    # Assert at least one direction is present (throne↔vault).
    link = next(
        (c for c in stairs if
         {c["from"], c["to"]} == {"throne", "vault"}),
        None,
    )
    assert link is not None, "throne↔vault stairs must exist"
    assert link["one_way"] is False


def test_m5_pinned_room():
    """Pinned room: a room with map: x/y must have placement: pinned in map.json
    and be placed at the specified cell coordinates.
    """
    pinned = FMLEntity(
        id="shrine",
        name="Shrine",
        kind="room",
        properties={
            "exits": {"south": "plaza"},
            "map": {"x": 20, "y": 10},
        },
        kind_chain=["room"],
    )
    plaza = FMLEntity(
        id="plaza",
        name="Plaza",
        kind="room",
        properties={"exits": {"north": "shrine"}},
        kind_chain=["room"],
    )
    f = Floor(name="Test", properties={"start_location": "plaza"})
    f += pinned
    f += plaza

    # Read map data before stripping.
    lua = emit_lua_om(f, source_path="test.md")
    data = emit_map(f, lua.encode("utf-8"))

    assert data["rooms"]["shrine"]["placement"] == "pinned", \
        "Room with map: x/y override must have placement: pinned"
    assert data["rooms"]["shrine"]["rect"]["x"] == 20
    assert data["rooms"]["shrine"]["rect"]["y"] == 10


def test_m3_same_level_up_down_is_warp_not_stairs():
    """M3 regression: up/down between two SAME-LEVEL rooms must lower as warp,
    not stairs.

    The Bone Garden's Twisting Tunnels use up/down as maze exits back to the
    same-level catacombs room (both are level -1).  Before the fix, every
    up/down exit was unconditionally classified as stairs, producing 15 phantom
    stair connections in a flat maze.

    Per MAP_FORMAT §3 step 1 and §7, stairs are plane-changes only.
    """
    # Same-level up/down: should produce WARP, not stairs.
    maze_a = _make_room("maze_a", "Maze A", {"up": "hub", "down": "hub", "east": "maze_b"}, level=-1)
    maze_b = _make_room("maze_b", "Maze B", {"west": "maze_a", "up": "hub", "down": "hub"}, level=-1)
    hub = _make_room("hub", "Hub", {}, level=-1)
    f = _make_floor(maze_a, maze_b, hub, start="hub")
    data = _emit_map_for(f)

    # No stairs at all: all rooms are on the same level.
    stairs = [c for c in data["connections"] if c["kind"] == "stairs"]
    assert stairs == [], (
        f"Same-level up/down exits must NOT produce stairs; got {stairs}"
    )

    # The up/down exits from maze_a must be warp.
    up_conn = next(
        (c for c in data["connections"]
         if c["from"] == "maze_a" and c["dir"] == "up"),
        None,
    )
    assert up_conn is not None, "maze_a up→hub connection must be present"
    assert up_conn["kind"] == "warp", (
        f"Same-level 'up' exit must be warp, got {up_conn['kind']}"
    )

    down_conn = next(
        (c for c in data["connections"]
         if c["from"] == "maze_a" and c["dir"] == "down"),
        None,
    )
    assert down_conn is not None, "maze_a down→hub connection must be present"
    assert down_conn["kind"] == "warp", (
        f"Same-level 'down' exit must be warp, got {down_conn['kind']}"
    )


def test_m3_different_level_up_down_is_stairs():
    """M3 confirmation: up/down between DIFFERENT-LEVEL rooms must be stairs.

    This is the expected case: ossuary (level 0) down→catacombs (level -1)
    is a staircase, not a warp.  Both directions are emitted independently
    (each room's outbound exit) so there are exactly 2 stairs connections.
    """
    ossuary = _make_room("ossuary", "Ossuary", {"down": "catacombs"}, level=0)
    catacombs = _make_room("catacombs", "Catacombs", {"up": "ossuary"}, level=-1)
    f = _make_floor(ossuary, catacombs, start="ossuary")
    data = _emit_map_for(f)

    stairs = [c for c in data["connections"] if c["kind"] == "stairs"]
    assert len(stairs) == 2, (
        f"Expected exactly 2 stairs connections (one per direction); got {stairs}"
    )
    # Both directions must be present.
    ossuary_down = next(
        (c for c in stairs if c["from"] == "ossuary" and c["to"] == "catacombs"), None
    )
    catacombs_up = next(
        (c for c in stairs if c["from"] == "catacombs" and c["to"] == "ossuary"), None
    )
    assert ossuary_down is not None, "ossuary→catacombs stairs must exist"
    assert catacombs_up is not None, "catacombs→ossuary stairs must exist"
    assert ossuary_down["one_way"] is False, "Bidirectional up/down stairs must be one_way: false"
    assert catacombs_up["one_way"] is False, "Bidirectional up/down stairs must be one_way: false"
    assert ossuary_down["path"] == [], "Stairs connections must have empty path"
    assert catacombs_up["path"] == [], "Stairs connections must have empty path"


def test_m5_nongeometric_direction_is_warp():
    """Non-geometric directions (in, out, author-invented) lower as warp."""
    cave = _make_room("cave", "Cave", {"in": "inner", "out": "field"})
    inner = _make_room("inner", "Inner Cave", {"out": "cave"})
    field = _make_room("field", "Field", {"in": "cave"})
    f = _make_floor(cave, inner, field, start="field")
    data = _emit_map_for(f)

    in_conn = next(
        (c for c in data["connections"] if c["from"] == "cave" and c["dir"] == "in"),
        None,
    )
    assert in_conn is not None
    assert in_conn["kind"] == "warp", "non-geometric 'in' direction must be warp"

    out_conn = next(
        (c for c in data["connections"] if c["from"] == "cave" and c["dir"] == "out"),
        None,
    )
    if out_conn:
        assert out_conn["kind"] == "warp", "non-geometric 'out' direction must be warp"


# ─────────────────────────────────────────────────────────────────────────────
# M5 — Reciprocal (bidirectional) connections
# ─────────────────────────────────────────────────────────────────────────────


def test_m5_reciprocal_connections_both_emitted():
    """A reciprocal pair A –east→ B / B –west→ A must produce TWO connections.

    The VTT needs each room's own outbound exit in map.json so it can drive
    movement in either direction (drag-to-move, table_move adjacency check,
    connection arrows).  The old dedup suppressed one direction; this test pins
    the fix.
    """
    room_a = _make_room("room_a", "Room A", {"east": "room_b"})
    room_b = _make_room("room_b", "Room B", {"west": "room_a"})
    f = _make_floor(room_a, room_b, start="room_a")
    data = _emit_map_for(f)

    conns_by = {(c["from"], c["dir"], c["to"]): c for c in data["connections"]}

    a_east = conns_by.get(("room_a", "east", "room_b"))
    b_west = conns_by.get(("room_b", "west", "room_a"))

    assert a_east is not None, "room_a –east→ room_b must be present"
    assert b_west is not None, "room_b –west→ room_a must be present"

    # Both are bidirectional (one_way: false) because the reverse exit exists.
    assert a_east["one_way"] is False, "room_a –east→ room_b must be one_way: false"
    assert b_west["one_way"] is False, "room_b –west→ room_a must be one_way: false"


# ─────────────────────────────────────────────────────────────────────────────
# M1 — --map CLI flag
# ─────────────────────────────────────────────────────────────────────────────


def test_m1_cli_map_flag_writes_map_json(tmp_path):
    """--map flag writes map.json beside the output floor.lua."""
    from fml_parser.__main__ import main

    index_md = BONE_GARDEN_DIR / "index.md"
    out_lua = tmp_path / "floor.lua"
    map_json = tmp_path / "map.json"

    rc = main([
        "lower", str(index_md),
        "--om", "--map",
        "-o", str(out_lua),
    ])

    assert rc == 0, "CLI must exit 0"
    assert out_lua.exists(), "floor.lua must be written"
    assert map_json.exists(), "map.json must be written beside floor.lua"

    # map.json must be valid JSON with the expected top-level fields.
    data = json.loads(map_json.read_text())
    assert data["map_version"] == 1
    assert data["floor"] == "the_bone_garden"
    assert "rooms" in data
    assert "connections" in data
    assert "levels" in data


def test_m1_cli_without_map_flag_no_sidecar(tmp_path):
    """Without --map, no map.json is written and the Lua output is unchanged."""
    from fml_parser.__main__ import main

    index_md = BONE_GARDEN_DIR / "index.md"

    # Run WITHOUT --map in its own subdirectory so a prior --map run in the same
    # tmp_path doesn't leave a stale map.json behind for us to trip over.
    no_map_dir = tmp_path / "no_map"
    no_map_dir.mkdir()
    out_lua = no_map_dir / "floor.lua"
    map_json = no_map_dir / "map.json"

    rc = main(["lower", str(index_md), "--om", "-o", str(out_lua)])
    assert rc == 0
    assert out_lua.exists()
    assert not map_json.exists(), "map.json must NOT be written without --map"

    # Run WITH --map in a separate directory to get the reference Lua bytes.
    with_map_dir = tmp_path / "with_map"
    with_map_dir.mkdir()
    out_with_map = with_map_dir / "floor.lua"
    main(["lower", str(index_md), "--om", "--map", "-o", str(out_with_map)])

    # The Lua output must be byte-identical whether or not --map is passed.
    assert out_lua.read_bytes() == out_with_map.read_bytes(), \
        "--map must not alter the floor.lua output"


def test_m1_map_requires_floor_mode():
    """--map with --stdlib-module must exit with code 2."""
    from fml_parser.__main__ import main
    index_md = BONE_GARDEN_DIR / "index.md"
    rc = main(["--stdlib-module", str(index_md), "--map"])
    assert rc == 2


def test_m1_map_requires_om_flag():
    """--map without --om must exit with code 2."""
    from fml_parser.__main__ import main
    with tempfile.TemporaryDirectory() as td:
        index_md = BONE_GARDEN_DIR / "index.md"
        rc = main(["lower", str(index_md), "--map", "-o", str(Path(td) / "f.lua")])
        assert rc == 2


# ─────────────────────────────────────────────────────────────────────────────
# M6 — Golden Bone Garden
# ─────────────────────────────────────────────────────────────────────────────


def _load_golden() -> dict:
    return json.loads(GOLDEN_MAP_PATH.read_text(encoding="utf-8"))


def _generate_bone_garden_map() -> dict:
    floor = _parse_fixture(BONE_GARDEN_DIR / "index.md")
    # Stable RELATIVE source_path: the floor.lua header (and thus floor_sha)
    # must not depend on the absolute checkout location, or the golden diverges
    # between a dev machine and CI. See CLAUDE.md "bit-identical lowering".
    lua = emit_lua_om(floor, source_path="bone_garden/index.md")
    return emit_map(floor, lua.encode("utf-8"))


def test_m6_golden_byte_equality():
    """Lower the vendored Bone Garden fixture twice: assert byte-identical
    map.json, and byte-equal to the checked-in golden fixture.
    """
    floor = _parse_fixture(BONE_GARDEN_DIR / "index.md")
    # Stable RELATIVE source_path: the floor.lua header (and thus floor_sha)
    # must not depend on the absolute checkout location, or the golden diverges
    # between a dev machine and CI. See CLAUDE.md "bit-identical lowering".
    lua = emit_lua_om(floor, source_path="bone_garden/index.md")
    lua_bytes = lua.encode("utf-8")

    # Two runs must produce the same output.
    run1 = emit_map_json(floor, lua_bytes)
    run2 = emit_map_json(floor, lua_bytes)
    assert run1 == run2, "Two runs of emit_map_json must be byte-identical"

    # Must match the checked-in golden fixture.
    golden_str = GOLDEN_MAP_PATH.read_text(encoding="utf-8")
    assert run1 == golden_str, (
        "Generated map.json does not match the golden fixture.\n"
        "If the change is intentional, regenerate the golden file:\n"
        "  python -m fml_parser lower tests/fixtures/bone_garden/index.md "
        "--om --map -o /tmp/floor.lua && cp /tmp/map.json "
        "tests/fixtures/bone_garden/golden_map.json"
    )


def test_m6_three_planes():
    """Bone Garden map must have exactly 3 level planes: 0, -1, -2."""
    data = _generate_bone_garden_map()
    level_nums = sorted([l["level"] for l in data["levels"]], reverse=True)
    assert level_nums == [0, -1, -2], f"Expected levels [0,-1,-2], got {level_nums}"


def test_m6_surface_cardinal_placement():
    """Bone Garden surface: hall north of entry, ossuary west, garden south."""
    data = _generate_bone_garden_map()
    rooms = data["rooms"]

    # All four surface rooms must be on level 0.
    for rid in ("entry", "hall_of_skulls", "ossuary", "garden"):
        assert rooms[rid]["level"] == 0, f"{rid} must be on level 0"

    entry_rect = rooms["entry"]["rect"]
    hall_rect = rooms["hall_of_skulls"]["rect"]
    ossuary_rect = rooms["ossuary"]["rect"]
    garden_rect = rooms["garden"]["rect"]

    # Hall is NORTH of entry → smaller y.
    assert hall_rect["y"] < entry_rect["y"], \
        f"Hall (y={hall_rect['y']}) must be north (smaller y) of Entry (y={entry_rect['y']})"

    # Ossuary is WEST of entry → smaller x.
    assert ossuary_rect["x"] < entry_rect["x"], \
        f"Ossuary (x={ossuary_rect['x']}) must be west (smaller x) of Entry (x={entry_rect['x']})"

    # Garden is SOUTH of entry → larger y.
    assert garden_rect["y"] > entry_rect["y"], \
        f"Garden (y={garden_rect['y']}) must be south (larger y) of Entry (y={entry_rect['y']})"


def test_m6_no_negative_coordinates():
    """All room coordinates must be non-negative."""
    data = _generate_bone_garden_map()
    for rid, room in data["rooms"].items():
        rect = room["rect"]
        assert rect["x"] >= 0, f"Room {rid} has negative x: {rect['x']}"
        assert rect["y"] >= 0, f"Room {rid} has negative y: {rect['y']}"


def test_m6_ossuary_catacombs_stairs():
    """The ossuary↔catacombs up/down link must lower as stairs (not corridor/warp).

    Both directions are emitted (each room's own outbound exit), so there are
    exactly 2 stairs connections for this pair.
    """
    data = _generate_bone_garden_map()

    stairs = [
        c for c in data["connections"]
        if c["kind"] == "stairs"
        and {c["from"], c["to"]} == {"ossuary", "catacombs"}
    ]
    assert len(stairs) == 2, \
        f"Expected exactly 2 stairs connections (one per direction) between ossuary and catacombs, got {stairs}"
    for s in stairs:
        assert s["one_way"] is False, \
            f"ossuary↔catacombs stairs must be bidirectional (one_way: false), got {s}"
        assert s["path"] == [], "stairs connections have no path"
    # Assert both directions specifically present.
    assert any(s["from"] == "ossuary" and s["to"] == "catacombs" for s in stairs), \
        "ossuary→catacombs direction must be present"
    assert any(s["from"] == "catacombs" and s["to"] == "ossuary" for s in stairs), \
        "catacombs→ossuary direction must be present"


def test_m6_tunnel_crypt_warp_with_door():
    """The twisting_tunnel_a7 → crypt edge must lower as warp with door: stone_door."""
    data = _generate_bone_garden_map()

    # The exit is: twisting_tunnel_a7 east → {room: crypt, door: stone_door}
    # twisting_tunnel_a7 is level -1; crypt is level -2 → cross-level → warp.
    conn = next(
        (c for c in data["connections"]
         if c["from"] == "twisting_tunnel_a7"
         and c["to"] == "crypt"
         and c["dir"] == "east"),
        None,
    )
    assert conn is not None, \
        "Expected a twisting_tunnel_a7 east→crypt connection"
    assert conn["kind"] == "warp", \
        f"Cross-level tunnel→crypt edge must be warp, got {conn['kind']}"
    assert conn["door"] == "stone_door", \
        f"tunnel→crypt warp must carry door: stone_door, got {conn['door']}"
    assert conn["one_way"] is True, \
        "tunnel→crypt is one-way (crypt exits south, not east back to tunnel)"


def test_m6_schema_fields_present():
    """The Bone Garden map.json must have all required top-level fields."""
    data = _generate_bone_garden_map()

    assert data["map_version"] == 1
    assert data["floor"] == "the_bone_garden"
    assert data["floor_sha"]  # non-empty sha256
    assert data["generator"].startswith("clotho ")
    assert data["grid"]["type"] == "square"
    assert isinstance(data["grid"]["cell_px"], int)
    assert isinstance(data["levels"], list) and len(data["levels"]) == 3
    assert isinstance(data["rooms"], dict)
    assert isinstance(data["connections"], list)
    assert "overhead" in data["layers"]
    assert "hidden" in data["layers"]
    assert data["layers"]["hidden"] == []
    assert isinstance(data["tokens"]["art"], dict)


def test_m6_stairs_exactly_one_ossuary_catacombs():
    """M6 §7: the real Bone Garden fixture's ONLY stairs link is ossuary↔catacombs.

    Both directions are emitted (each room's own outbound exit), so there are
    exactly 2 stairs connections (ossuary→catacombs and catacombs→ossuary).

    The Twisting Tunnels use up/down as same-level maze exits (both rooms on
    level -1).  They must lower as warp, not stairs.  This test pins the
    regression introduced when _classify_connection unconditionally returned
    stairs for any up/down exit, producing 15 phantom stairs.
    """
    data = _generate_bone_garden_map()
    all_stairs = [c for c in data["connections"] if c["kind"] == "stairs"]
    assert len(all_stairs) == 2, (
        f"MAP_FORMAT §7 states ossuary↔catacombs is the floor's ONLY up/down link; "
        f"expected exactly 2 stairs connections (one per direction), got {len(all_stairs)}: "
        f"{[(c['from'], c['dir'], c['to']) for c in all_stairs]}"
    )
    for s in all_stairs:
        assert {s["from"], s["to"]} == {"ossuary", "catacombs"}, (
            f"All stairs connections must be ossuary↔catacombs, got {s}"
        )


def test_m6_tunnel_updown_catacombs_are_warp():
    """M6: All twisting-tunnel up/down→catacombs connections must be warp
    (same-level exits, not stairs).

    The fixture now faithfully includes the up/down exits on all 7 tunnel
    rooms (twisting_tunnel_a1 … a7).  With the bug fixed, every one lowers
    as warp because the tunnel and catacombs are both on level -1.
    """
    data = _generate_bone_garden_map()

    tunnel_updown = [
        c for c in data["connections"]
        if c["from"].startswith("twisting_tunnel_")
        and c["dir"] in ("up", "down")
    ]
    # 7 rooms × 2 directions = 14 up/down connections from tunnel rooms.
    assert len(tunnel_updown) == 14, (
        f"Expected 14 tunnel up/down connections (7 rooms × up + down), "
        f"got {len(tunnel_updown)}: {[(c['from'], c['dir']) for c in tunnel_updown]}"
    )
    non_warp = [c for c in tunnel_updown if c["kind"] != "warp"]
    assert non_warp == [], (
        f"Same-level tunnel up/down exits must ALL be warp; "
        f"non-warp: {[(c['from'], c['dir'], c['kind']) for c in non_warp]}"
    )


def test_m6_tunnel_crypt_warp_with_door_and_three_levels():
    """M6: twisting_tunnel_a7 east→crypt is warp+door; floor has exactly 3 levels."""
    data = _generate_bone_garden_map()

    # Three levels: 0, -1, -2.
    level_nums = sorted([l["level"] for l in data["levels"]], reverse=True)
    assert level_nums == [0, -1, -2], (
        f"Expected exactly 3 levels [0, -1, -2], got {level_nums}"
    )

    # tunnel_a7 east → crypt: cross-level warp with stone_door.
    conn = next(
        (c for c in data["connections"]
         if c["from"] == "twisting_tunnel_a7"
         and c["to"] == "crypt"
         and c["dir"] == "east"),
        None,
    )
    assert conn is not None, "twisting_tunnel_a7 east→crypt connection must exist"
    assert conn["kind"] == "warp", (
        f"Cross-level tunnel_a7→crypt must be warp, got {conn['kind']}"
    )
    assert conn["door"] == "stone_door", (
        f"tunnel_a7→crypt must carry door: stone_door, got {conn['door']}"
    )
    assert conn["one_way"] is True, (
        "tunnel_a7→crypt is one-way (crypt exits south, not east back to tunnel)"
    )


def test_m4_map_key_stripped_from_om_lua_output():
    """
    Proof that map: key is absent from floor.lua but present in map.json.

    This test:
    1. Creates a room with a `map:` property containing an art URL.
    2. Emits map.json (before stripping) — asserts the art URL is in map.json.
    3. Strips the floor.
    4. Emits floor.lua (after stripping) — asserts the art URL is NOT in the Lua.
    """
    import re as _re

    room = FMLEntity(
        id="hall",
        name="Hall",
        kind="room",
        properties={
            "exits": {},
            "map": {"image": "https://cdn.example.com/art.png"},
        },
        kind_chain=["room"],
    )
    f = Floor(name="Art Test", properties={"start_location": "hall"})
    f += room

    # Step 1: emit map (reads map key).
    lua_bytes = emit_lua_om(f, source_path="test.md").encode()
    map_data = emit_map(f, lua_bytes)

    # Step 2: strip.
    strip_map_keys(f)

    # Step 3: emit Lua.
    lua_after = emit_lua_om(f, source_path="test.md")

    # Assertions: map.json has the art URL.
    assert map_data["rooms"]["hall"]["art"] == {
        "src": "https://cdn.example.com/art.png", "fit": "cover"
    }, "Art URL must be present in map.json"

    # Assertions: floor.lua does NOT have the art URL.
    assert "https://cdn.example.com/art.png" not in lua_after, (
        "Art URL from map: property must NOT appear in floor.lua (Wyrd must never see it)"
    )


def test_cli_render_hint_round_trips_to_map_json(tmp_path):
    """Regression: `lower --om --map` must carry a room's `render:` hint into
    map.json. The CLI snapshots presentation keys before strip_map_keys (which
    strips render too) and re-hydrates them for the map emitter — render must be
    snapshotted/re-hydrated like map/token, else it vanishes from map.json. The
    direct emit_map unit tests don't exercise this CLI path."""
    from fml_parser.__main__ import main

    (tmp_path / "rooms").mkdir()
    (tmp_path / "index.md").write_text(
        "# Mini Floor\n\n"
        "[Tunnel](rooms/tunnel.md)\n[Hall](rooms/hall.md)\n\n"
        "- start_location: hall\n\n> mini.\n",
        encoding="utf-8",
    )
    (tmp_path / "rooms" / "tunnel.md").write_text(
        "# Tunnel\n\n- kind: room\n- render:\n  - hide_arrows: true\n"
        "- exits:\n  - south: hall\n\n> a tunnel.\n",
        encoding="utf-8",
    )
    (tmp_path / "rooms" / "hall.md").write_text(
        "# Hall\n\n- kind: room\n- exits:\n  - north: tunnel\n\n> a hall.\n",
        encoding="utf-8",
    )

    out_lua = tmp_path / "floor.lua"
    rc = main(["lower", str(tmp_path / "index.md"), "--om", "--map", "-o", str(out_lua)])
    assert rc == 0, "CLI must exit 0"

    map_data = json.loads((tmp_path / "map.json").read_text())
    assert map_data["rooms"]["tunnel"]["render"] == {"hide_arrows": True}, (
        "render hint must survive the CLI snapshot/strip/re-hydrate into map.json"
    )
    assert "render" not in map_data["rooms"]["hall"]
    # And it must never leak into the engine LFR.
    assert "hide_arrows" not in out_lua.read_text()
