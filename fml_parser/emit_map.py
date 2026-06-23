"""Map emitter — `map.json` sidecar for the VTT pipeline.

Implements MAP_FORMAT.md §2 (schema), §3 (auto-layout), §4 (FML reserved keys).

Public function:
    emit_map(floor, floor_lua_bytes, *, cell_px, room_w, room_h) -> dict

The dict is JSON-serialisable with json.dumps(result, sort_keys=False).
It is intentionally RNG-free and deterministic: given the same Floor and
floor_lua_bytes, this function returns a byte-identical JSON document
regardless of environment, Python version, or dict-insertion order.

Determinism measures (must not be relaxed):
  - All room-id iteration is sorted (sorted() everywhere we iterate over sets
    or dicts keyed by room ids).
  - All exit-name iteration is sorted.
  - BFS queue is a list (order = sorted neighbours), never a set.
  - Component packing order is determined by sorted first-room-id of each
    component.
  - No `random`, `time`, `uuid`, or other non-deterministic calls anywhere.
  - json.dumps is called with sort_keys=False and ensure_ascii=True so that
    the key order in the output is the order we built it (which is itself
    deterministic).

Reserved keys stripped from LFR (MAP_FORMAT.md §4):
  - Floor-level: properties["map"]  (dict with cell_size, room_width,
                                      room_height, palette)
  - Per-entity:  entity.properties["map"]  (dict with x, y, width, height,
                                            image, overhead_image)
                 entity.properties["token"] (absolute URL string)

The caller (emit_lua_om / __main__.py) is responsible for stripping those
keys BEFORE calling the Lua emitter so that Wyrd never sees them.  The
helper `strip_map_keys(floor)` mutates the Floor in place; call it once
after parsing and before both emitters.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

from .models import FMLEntity, Floor
from .slugify import slugify

# ── Package version (matches pyproject.toml) ─────────────────────────────────

_VERSION = "0.1.0"  # keep in sync with pyproject.toml

# ── Cardinal direction vectors (unit embedding, MAP_FORMAT §3 step 2) ────────

# direction name → (dx, dy)  where x=column, y=row, north = y-1
_CARDINAL_VECTORS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east":  (1, 0),
    "west":  (-1, 0),
    # Diagonals: combined from N+E etc.
    "northeast": (1, -1),
    "northwest": (-1, -1),
    "southeast": (1, 1),
    "southwest": (-1, 1),
}

# ── Directions that cross vertical levels, not horizontal planes ──────────────

_VERTICAL_DIRS = frozenset(["up", "down"])

# ── Non-geometric directions that always lower as warp ───────────────────────

_NONGEOMETRIC_DIRS = frozenset(["in", "out"])

# ── Opposite-direction lookup (used by bidirectionality check) ───────────────

_OPPOSITE: dict[str, str] = {
    "north": "south", "south": "north",
    "east": "west", "west": "east",
    "northeast": "southwest", "southwest": "northeast",
    "northwest": "southeast", "southeast": "northwest",
    "up": "down", "down": "up",
    "in": "out", "out": "in",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public helper: strip map/token keys from a Floor (mutates in place)
# ─────────────────────────────────────────────────────────────────────────────

def strip_map_keys(floor: Floor) -> None:
    """Remove `map:`, `token:`, and `render:` from the floor's property dicts.

    Called BEFORE the Lua emitter so that Wyrd never sees presentation data
    (MAP_FORMAT.md §4, Decisions §18).  Mutates the Floor object in place.
    Safe to call even when no map/token/render keys are present.

    `render:` carries VTT render hints (e.g. `hide_arrows`) consumed only by the
    map emitter; like map/token it is presentation-only and must never reach the
    engine LFR.
    """
    # Floor-level map defaults.
    floor.properties.pop("map", None)

    # Per-entity keys (walk all entities including sub-entities).
    for entity in floor.all_entities():
        # Entrance south-centre fallback (#119): a mapped room with no authored
        # `entrance` gets one at the south-edge centre (x = w//2, y = h-1) so a
        # spawned player enters from the "front" of the battlemap instead of the
        # origin (where an unpositioned boss sits). Injected as a top-level
        # `entrance` property HERE — the last point map dims exist before the Lua
        # emitter runs (which then lowers it to engine.set_entrance). Authored
        # entrances are left untouched.
        _inject_default_entrance(entity)
        entity.properties.pop("map", None)
        entity.properties.pop("token", None)
        entity.properties.pop("render", None)


def _inject_default_entrance(entity: FMLEntity) -> None:
    """Give a mapped room a south-centre `entrance` when none is authored."""
    if entity.properties.get("entrance") is not None:
        return
    if entity.kind != "room" and "room" not in entity.kind_chain:
        return
    m = entity.properties.get("map")
    if not isinstance(m, dict):
        return
    w, h = m.get("width"), m.get("height")
    if (isinstance(w, int) and not isinstance(w, bool) and w > 0
            and isinstance(h, int) and not isinstance(h, bool) and h > 0):
        entity.properties["entrance"] = [w // 2, h - 1, 0]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers: read FML map/token keys
# ─────────────────────────────────────────────────────────────────────────────

def _floor_map_defaults(floor: Floor) -> dict[str, Any]:
    """Read the floor-level `map:` property dict, if present."""
    v = floor.properties.get("map")
    return v if isinstance(v, dict) else {}


def _entity_map_overrides(entity: FMLEntity) -> dict[str, Any]:
    """Read per-entity `map:` overrides, if present."""
    v = entity.properties.get("map")
    return v if isinstance(v, dict) else {}


def _entity_token_url(entity: FMLEntity) -> str | None:
    """Read per-entity `token:` URL, if present."""
    v = entity.properties.get("token")
    return str(v) if v is not None else None


def _entity_render_hints(entity: FMLEntity) -> dict[str, Any]:
    """Read per-entity `render:` VTT hints dict, if present.

    Render hints are presentation-only directives (e.g. `hide_arrows: true` to
    suppress connection arrows on a room) passed through to map.json and stripped
    from the LFR.  Recognized keys are interpreted by the VTT client; the emitter
    passes the dict through verbatim so new hints need no emitter changes.
    """
    v = entity.properties.get("render")
    return v if isinstance(v, dict) else {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers: room/exit extraction from the Floor model
# ─────────────────────────────────────────────────────────────────────────────

def _is_room(entity: FMLEntity) -> bool:
    """True if the entity is a room (kind == "room" or kind_chain contains "room")."""
    if entity.kind == "room":
        return True
    if entity.kind_chain and "room" in entity.kind_chain:
        return True
    # Fallback: has `exits` property.
    return "exits" in entity.properties


def _room_level(entity: FMLEntity) -> int:
    """Return the room's level (default 0)."""
    v = entity.properties.get("level", 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _room_exits(entity: FMLEntity) -> dict[str, Any]:
    """Return the exits dict from the entity's properties.

    Exit values can be:
      - a bare string: "target_id"
      - a Markdown-link-resolved slug (parser converts links to slugs)
      - a dict with keys `room` and `door` (object form)
    """
    v = entity.properties.get("exits")
    return v if isinstance(v, dict) else {}


def _exit_target_and_door(exit_value: Any) -> tuple[str, str | None]:
    """Return (target_room_id, door_entity_id_or_None) from an exit value."""
    if isinstance(exit_value, dict):
        room = exit_value.get("room")
        door = exit_value.get("door")
        return (str(room) if room else ""), (str(door) if door else None)
    elif isinstance(exit_value, str):
        return exit_value, None
    else:
        return str(exit_value), None


# ─────────────────────────────────────────────────────────────────────────────
# Auto-layout (MAP_FORMAT.md §3)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_rooms(floor: Floor) -> dict[str, FMLEntity]:
    """Return all room entities keyed by id, in SORTED id order."""
    rooms: dict[str, FMLEntity] = {}
    for entity in floor.entities.values():
        if _is_room(entity):
            rooms[entity.id] = entity
    # Return in sorted order (deterministic).
    return {k: rooms[k] for k in sorted(rooms)}


def _collect_pinned_cells(
    rooms: dict[str, FMLEntity],
) -> dict[str, tuple[int, int]]:
    """Return {room_id: (cell_x, cell_y)} for rooms with FML-pinned x/y.

    The `map: x` / `map: y` values in FML are CELL coordinates (the same
    coordinate system as the output `rect.x` / `rect.y` fields).  These rooms
    bypass the unit→cell conversion and are placed at the specified cell position
    directly.  Their approximate unit position (for BFS collision avoidance) is
    derived in `_layout_level` as `(cell_x // cell_scale, cell_y // cell_scale)`.
    """
    pinned: dict[str, tuple[int, int]] = {}
    for rid, entity in rooms.items():
        overrides = _entity_map_overrides(entity)
        if "x" in overrides and "y" in overrides:
            try:
                cx = int(overrides["x"])
                cy = int(overrides["y"])
                pinned[rid] = (cx, cy)
            except (TypeError, ValueError):
                pass
    return pinned


class _Layout:
    """BFS-based auto-layout state for a single level plane.

    Maintains:
      - pos: {room_id: (ux, uy)} in unit coordinates.
      - shifted_cols: sorted list of inserted column shifts (for collision
        resolution).
      - shifted_rows: sorted list of inserted row shifts.
    """

    def __init__(self) -> None:
        self.pos: dict[str, tuple[int, int]] = {}
        # Track which unit columns/rows have been "shifted" to resolve
        # collisions.  Each entry is the unit-coordinate at which a split
        # was inserted; positions at or past that coordinate are offset by 1.
        # We apply shifts cumulatively and in sorted order.
        self._col_splits: list[int] = []  # sorted list of column insertions
        self._row_splits: list[int] = []  # sorted list of row insertions

    def _apply_col_splits(self, ux: int) -> int:
        """Translate a raw unit x through any column splits inserted so far."""
        offset = sum(1 for s in self._col_splits if s <= ux)
        return ux + offset

    def _apply_row_splits(self, uy: int) -> int:
        """Translate a raw unit y through any row splits inserted so far."""
        offset = sum(1 for s in self._row_splits if s <= uy)
        return uy + offset

    def set_pos(self, room_id: str, ux: int, uy: int) -> None:
        self.pos[room_id] = (ux, uy)

    def get_pos(self, room_id: str) -> tuple[int, int] | None:
        return self.pos.get(room_id)

    def insert_col_split(self, at_ux: int) -> None:
        """Insert a column split at `at_ux`; shift all rooms with ux >= at_ux."""
        import bisect
        bisect.insort(self._col_splits, at_ux)
        # Shift affected rooms.
        new_pos = {}
        for rid, (ux, uy) in self.pos.items():
            if ux >= at_ux:
                new_pos[rid] = (ux + 1, uy)
            else:
                new_pos[rid] = (ux, uy)
        self.pos = new_pos

    def insert_row_split(self, at_uy: int) -> None:
        """Insert a row split at `at_uy`; shift all rooms with uy >= at_uy."""
        import bisect
        bisect.insort(self._row_splits, at_uy)
        new_pos = {}
        for rid, (ux, uy) in self.pos.items():
            if uy >= at_uy:
                new_pos[rid] = (ux, uy + 1)
            else:
                new_pos[rid] = (ux, uy)
        self.pos = new_pos

    def occupied(self) -> set[tuple[int, int]]:
        return set(self.pos.values())


def _bfs_layout(
    start_id: str,
    rooms: dict[str, FMLEntity],
    pinned: dict[str, tuple[int, int]],
    start_ux: int,
    start_uy: int,
    layout: _Layout,
) -> None:
    """BFS from `start_id`, assigning unit positions.

    Uses first-assignment-wins (MAP_FORMAT §3 step 3): once a room has a
    position, direction-inconsistent edges become warp and don't move rooms.

    Collision handling (§3 step 4): if the target cell is occupied by a
    DIFFERENT room, insert a row/column split at the target position (along
    the arrival direction) so that a new cell opens up for the new room.

    Iteration order: sorted exit names at each BFS node (§3 step 7).
    """
    if start_id in layout.pos and layout.pos[start_id] != (start_ux, start_uy):
        # Pinned conflicts with BFS target: skip.
        return

    if start_id not in layout.pos:
        # Check pinned map for this room first.
        if start_id in pinned:
            layout.set_pos(start_id, *pinned[start_id])
        else:
            layout.set_pos(start_id, start_ux, start_uy)

    queue: deque[str] = deque([start_id])
    while queue:
        current_id = queue.popleft()
        entity = rooms.get(current_id)
        if entity is None:
            continue
        cx, cy = layout.pos[current_id]
        exits = _room_exits(entity)
        # Sorted exit names for deterministic BFS order (§3 step 7).
        for dir_name in sorted(exits.keys()):
            exit_val = exits[dir_name]
            target_id, _door = _exit_target_and_door(exit_val)
            if not target_id or target_id not in rooms:
                continue
            # Skip vertical and non-geometric (they don't affect layout).
            if dir_name in _VERTICAL_DIRS or dir_name in _NONGEOMETRIC_DIRS:
                continue
            vec = _CARDINAL_VECTORS.get(dir_name)
            if vec is None:
                # Author-invented direction → warp; no layout.
                continue
            dx, dy = vec
            nx, ny = cx + dx, cy + dy

            if target_id in layout.pos:
                # First-assignment-wins: room already placed.
                # Check if the edge is geometrically consistent.
                ex, ey = layout.pos[target_id]
                if (ex, ey) != (nx, ny):
                    # Direction-inconsistent → warp (no movement).
                    pass
                # Either consistent or warp; either way don't move it.
                continue

            # Target not placed yet.  Check for collision.
            occupied = layout.occupied()
            if (nx, ny) in occupied:
                # Another room already occupies that cell.
                # Insert a split along the arrival direction to make room.
                if dx != 0:
                    # Horizontal arrival: insert a column split.
                    layout.insert_col_split(nx if dx > 0 else nx + 1)
                    # Recalculate nx after split.
                    cx2, cy2 = layout.pos[current_id]
                    nx, ny = cx2 + dx, cy2 + dy
                elif dy != 0:
                    # Vertical arrival: insert a row split.
                    layout.insert_row_split(ny if dy > 0 else ny + 1)
                    cx2, cy2 = layout.pos[current_id]
                    nx, ny = cx2 + dx, cy2 + dy

            # If still occupied after split (e.g. pinned rooms), probe further.
            occupied = layout.occupied()
            probe_count = 0
            while (nx, ny) in occupied and probe_count < 20:
                nx += dx
                ny += dy
                probe_count += 1

            if start_id in pinned:
                # If start was pinned, don't override neighbours' pinned pos.
                if target_id in pinned:
                    # Both pinned: don't override.
                    layout.set_pos(target_id, *pinned[target_id])
                else:
                    layout.set_pos(target_id, nx, ny)
            else:
                if target_id in pinned:
                    layout.set_pos(target_id, *pinned[target_id])
                else:
                    layout.set_pos(target_id, nx, ny)
            queue.append(target_id)


def _layout_level(
    level_rooms: dict[str, FMLEntity],
    floor_start_id: str | None,
    pinned: dict[str, tuple[int, int]],
    component_offset_x: int = 0,
) -> dict[str, tuple[int, int]]:
    """Lay out all rooms in a single level plane.

    Returns {room_id: (ux, uy)} for this level (relative unit coords).

    BFS per connected component; components are packed left-to-right with a
    two-unit gutter between them (§3 step 4).

    Step 7: sorted iteration everywhere.
    """
    layout = _Layout()

    # Pre-place pinned rooms (§3 step 6).
    for rid in sorted(pinned.keys()):
        if rid in level_rooms:
            layout.set_pos(rid, *pinned[rid])

    # Find connected components via undirected adjacency on intra-level edges.
    all_ids = sorted(level_rooms.keys())
    adj: dict[str, set[str]] = {rid: set() for rid in all_ids}
    for rid in all_ids:
        entity = level_rooms[rid]
        exits = _room_exits(entity)
        for dir_name in sorted(exits.keys()):
            if dir_name in _VERTICAL_DIRS or dir_name in _NONGEOMETRIC_DIRS:
                continue
            if _CARDINAL_VECTORS.get(dir_name) is None:
                continue
            exit_val = exits[dir_name]
            target_id, _ = _exit_target_and_door(exit_val)
            if target_id in level_rooms:
                adj[rid].add(target_id)
                adj[target_id].add(rid)

    # BFS to discover components (sorted iteration).
    visited: set[str] = set()
    components: list[list[str]] = []
    for rid in all_ids:
        if rid in visited:
            continue
        comp: list[str] = []
        bfs_q: deque[str] = deque([rid])
        while bfs_q:
            cur = bfs_q.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nbr in sorted(adj[cur]):
                if nbr not in visited:
                    bfs_q.append(nbr)
        components.append(comp)
    # Sort components by their lexicographically-first room id (§3 step 7).
    components.sort(key=lambda c: sorted(c)[0])

    # Lay out each component.
    # Track the rightmost unit-x used so far for packing.
    max_x_used = component_offset_x - 1

    for comp in components:
        comp_sorted = sorted(comp)

        # Determine BFS start: the floor's start_location if in this component,
        # else the lexicographically first room in the component (§3 step 2).
        if floor_start_id and floor_start_id in comp_sorted:
            start_id = floor_start_id
        else:
            start_id = comp_sorted[0]

        # Where to place the start of this component?
        # Pinned start room: use its pinned position.
        if start_id in pinned and start_id in level_rooms:
            start_ux, start_uy = pinned[start_id]
        else:
            start_ux = max_x_used + 3  # 2-unit gutter between components
            start_uy = 0

        _bfs_layout(start_id, level_rooms, pinned, start_ux, start_uy, layout)

        # Place any rooms in the component that BFS didn't reach
        # (disconnected islands due to only non-embeddable edges).
        for rid in comp_sorted:
            if rid not in layout.pos:
                if rid in pinned:
                    layout.set_pos(rid, *pinned[rid])
                else:
                    max_x_used = max(
                        (x for x, y in layout.pos.values()), default=max_x_used
                    )
                    layout.set_pos(rid, max_x_used + 3, 0)

        # Update max_x_used for next component's packing.
        if layout.pos:
            max_x_used = max(x for x, y in layout.pos.values())

    return dict(layout.pos)


# ─────────────────────────────────────────────────────────────────────────────
# Connection classification (MAP_FORMAT §3 steps 1–5)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_connection(
    from_id: str,
    dir_name: str,
    to_id: str,
    door_id: str | None,
    from_level: int,
    to_level: int,
    from_pos: tuple[int, int] | None,
    to_pos: tuple[int, int] | None,
    is_one_way: bool,
) -> dict[str, Any]:
    """Return a connection record (without path; path filled in later)."""
    # Vertical links (up/down): stairs only when the rooms are on different
    # levels (MAP_FORMAT §3 step 1, §7).  Same-level up/down exits are
    # non-geometric maze inputs (e.g. the Bone Garden's Twisting Tunnels
    # all use up/down → catacombs on the same level -1 plane) and must
    # lower as warp, not stairs.
    if dir_name in _VERTICAL_DIRS:
        if from_level != to_level:
            return {
                "from": from_id,
                "dir": dir_name,
                "to": to_id,
                "kind": "stairs",
                "door": door_id,
                "one_way": is_one_way,
                "path": [],
            }
        # Same level: treat like any other non-geometric direction → warp.
        return {
            "from": from_id,
            "dir": dir_name,
            "to": to_id,
            "kind": "warp",
            "door": door_id,
            "one_way": is_one_way,
            "path": [],
        }

    # Cross-level cardinal edge → warp (§3 step 1).
    if from_level != to_level:
        return {
            "from": from_id,
            "dir": dir_name,
            "to": to_id,
            "kind": "warp",
            "door": door_id,
            "one_way": is_one_way,
            "path": [],
        }

    # Non-geometric directions → warp (§3 step 2).
    if dir_name in _NONGEOMETRIC_DIRS or _CARDINAL_VECTORS.get(dir_name) is None:
        return {
            "from": from_id,
            "dir": dir_name,
            "to": to_id,
            "kind": "warp",
            "door": door_id,
            "one_way": is_one_way,
            "path": [],
        }

    # No positions (rooms not placed) → warp.
    if from_pos is None or to_pos is None:
        return {
            "from": from_id,
            "dir": dir_name,
            "to": to_id,
            "kind": "warp",
            "door": door_id,
            "one_way": is_one_way,
            "path": [],
        }

    # Check geometric consistency (§3 step 3).
    dx, dy = _CARDINAL_VECTORS[dir_name]
    expected_to = (from_pos[0] + dx, from_pos[1] + dy)
    if to_pos != expected_to:
        # Direction-inconsistent → warp.
        return {
            "from": from_id,
            "dir": dir_name,
            "to": to_id,
            "kind": "warp",
            "door": door_id,
            "one_way": is_one_way,
            "path": [],
        }

    # Geometrically consistent. Determine corridor vs door vs implicit.
    fx, fy = from_pos
    tx, ty = to_pos
    # Edge-adjacent (gutter consumed): implicit (§3 step 5).
    # Two rooms are edge-adjacent if |dx| + |dy| == 1 and the rooms touch.
    # In unit coordinates, adjacent means distance == 1 along the movement axis.
    dist = abs(tx - fx) + abs(ty - fy)
    if dist == 1:
        # Could be edge-adjacent after room scaling; if the room rects touch,
        # it's implicit. Since we track unit positions and room sizes separately,
        # we use a simple heuristic: distance-1 unit neighbours are implicit
        # only when room sizes fill the gap.  For the default 3x3+2gutter case,
        # gap is 2 units so distance-1 rooms (from pinning) are adjacent = implicit.
        kind = "implicit"
    else:
        # Corridor or door.
        kind = "door" if door_id else "corridor"

    return {
        "from": from_id,
        "dir": dir_name,
        "to": to_id,
        "kind": kind,
        "door": door_id,
        "one_way": is_one_way,
        "path": [],  # path filled in below
    }


def _route_path(
    from_rect: dict[str, int],
    to_rect: dict[str, int],
    dir_name: str,
) -> list[list[int]]:
    """Route an orthogonal corridor path between two rooms (§3 step 5).

    Works off the rooms' final (centered) cell RECTS so the corridor meets each
    room at the midpoint of its facing wall — regardless of the two rooms' sizes.
    Returns a list of [col, row] cell waypoints.
    """
    dx, dy = _CARDINAL_VECTORS.get(dir_name, (0, 0))

    def _facing_midpoint(rect: dict[str, int], ddx: int, ddy: int) -> tuple[int, int]:
        """Midpoint of the wall of `rect` facing direction (ddx, ddy)."""
        cx_mid = rect["x"] + rect["w"] // 2
        cy_mid = rect["y"] + rect["h"] // 2
        col = rect["x"] + rect["w"] if ddx > 0 else rect["x"] if ddx < 0 else cx_mid
        row = rect["y"] + rect["h"] if ddy > 0 else rect["y"] if ddy < 0 else cy_mid
        return col, row

    from_col, from_row = _facing_midpoint(from_rect, dx, dy)
    to_col, to_row = _facing_midpoint(to_rect, -dx, -dy)

    # Simple L-shaped orthogonal route.
    if from_col == to_col or from_row == to_row:
        # Already collinear — straight line.
        return [[from_col, from_row], [to_col, to_row]]
    # L-shape: go horizontal first, then vertical.
    return [[from_col, from_row], [to_col, from_row], [to_col, to_row]]


# ─────────────────────────────────────────────────────────────────────────────
# Cell-coordinate transformation — content-sized tracks + centered cells (§3)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_tracks(
    unit_pos: dict[str, tuple[int, int]],
    sizes: dict[str, tuple[int, int]],
    default_w: int,
    default_h: int,
    gutter: int,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Content-sized track grid for one level plane (MAP_FORMAT §3).

    Each unit COLUMN is sized to the widest room placed in it and each unit ROW
    to the tallest; empty interior tracks take the default room size so the grid
    stays contiguous (cardinal steps land on real, separated bands). Returns
    ``(col_x, col_w, row_y, row_h)``: for each used unit coordinate, the starting
    cell offset and the band size of its column / row. A room is then centered in
    its (col, row) band by :func:`_track_rect`, so adjacent rooms of any sizes
    align on their shared wall midpoints. Deterministic: only sorted/range
    iteration, no RNG.
    """
    if not unit_pos:
        return {}, {}, {}, {}
    col_w: dict[int, int] = {}
    row_h: dict[int, int] = {}
    for rid, (ux, uy) in unit_pos.items():
        w, h = sizes[rid]
        col_w[ux] = max(col_w.get(ux, 0), w)
        row_h[uy] = max(row_h.get(uy, 0), h)
    # Fill empty interior tracks with the default room size.
    min_ux, max_ux = min(col_w), max(col_w)
    min_uy, max_uy = min(row_h), max(row_h)
    for ux in range(min_ux, max_ux + 1):
        col_w.setdefault(ux, default_w)
    for uy in range(min_uy, max_uy + 1):
        row_h.setdefault(uy, default_h)
    # Cumulative cell offsets with a gutter between every track (origin at the
    # first used track, so the plane is normalized to (0, 0)).
    col_x: dict[int, int] = {}
    acc = 0
    for ux in range(min_ux, max_ux + 1):
        col_x[ux] = acc
        acc += col_w[ux] + gutter
    row_y: dict[int, int] = {}
    acc = 0
    for uy in range(min_uy, max_uy + 1):
        row_y[uy] = acc
        acc += row_h[uy] + gutter
    return col_x, col_w, row_y, row_h


def _track_rect(
    ux: int, uy: int, w: int, h: int,
    col_x: dict[int, int], col_w: dict[int, int],
    row_y: dict[int, int], row_h: dict[int, int],
) -> dict[str, int]:
    """Cell rect for a room of size (w, h) centered within its (col, row) band."""
    x = col_x[ux] + (col_w[ux] - w) // 2
    y = row_y[uy] + (row_h[uy] - h) // 2
    return {"x": x, "y": y, "w": w, "h": h}


# ─────────────────────────────────────────────────────────────────────────────
# Main emitter
# ─────────────────────────────────────────────────────────────────────────────

def emit_map(
    floor: Floor,
    floor_lua_bytes: bytes,
    *,
    default_cell_px: int = 64,
    default_room_w: int = 3,
    default_room_h: int = 3,
) -> dict[str, Any]:
    """Compute and return the map.json dict for `floor`.

    `floor_lua_bytes` is the bytes of the emitted floor.lua (used for sha256).
    Reserved keys (`map:`, `token:`) must still be present in floor at call
    time (they should be stripped by the caller AFTER this call, before writing
    the Lua LFR — or use the public `strip_map_keys` helper).

    Returns a dict suitable for `json.dumps(result, indent=2)`.
    Determinism: this function is pure given the same inputs.
    """
    # ── Read floor-level map defaults ────────────────────────────────────────
    floor_map_defs = _floor_map_defaults(floor)
    cell_px = int(floor_map_defs.get("cell_size", default_cell_px))
    room_w = int(floor_map_defs.get("room_width", default_room_w))
    room_h = int(floor_map_defs.get("room_height", default_room_h))
    floor_palette = str(floor_map_defs.get("palette", "default"))

    # cell_scale: unit-to-cell multiplier = room_size + gutter
    # Default 3x3 room + 2-cell gutter = 5 cells per unit.
    gutter = 2
    cell_scale = max(room_w, room_h) + gutter

    # ── Collect rooms ─────────────────────────────────────────────────────────
    rooms = _collect_rooms(floor)

    # ── Read per-entity map overrides ─────────────────────────────────────────
    entity_map_overrides: dict[str, dict[str, Any]] = {}
    for rid, entity in rooms.items():
        ovr = _entity_map_overrides(entity)
        if ovr:
            entity_map_overrides[rid] = ovr

    # Per-room cell size (w, h): override or floor default. Drives both the
    # content-sized track grid and each room's rect.
    room_sizes: dict[str, tuple[int, int]] = {}
    for rid in rooms:
        ovr = entity_map_overrides.get(rid, {})
        room_sizes[rid] = (
            int(ovr.get("width", room_w)),
            int(ovr.get("height", room_h)),
        )

    # ── Read per-entity render hints (VTT presentation, e.g. hide_arrows) ─────
    entity_render_hints: dict[str, dict[str, Any]] = {}
    for rid, entity in rooms.items():
        hints = _entity_render_hints(entity)
        if hints:
            entity_render_hints[rid] = hints

    # ── Collect pinned rooms (cell coordinates) ──────────────────────────────
    # pinned_cells: {room_id: (cell_x, cell_y)} — author-specified cell coords.
    # For BFS, we convert these to approximate unit coords.
    pinned_cells = _collect_pinned_cells(rooms)

    # pinned_units: approximate unit positions for pinned rooms (for BFS seeding).
    pinned_units: dict[str, tuple[int, int]] = {
        rid: (cx // cell_scale, cy // cell_scale)
        for rid, (cx, cy) in pinned_cells.items()
    }

    # ── Determine start room ──────────────────────────────────────────────────
    start_id: str | None = None
    sp = floor.properties.get("start_location")
    if isinstance(sp, str) and sp in rooms:
        start_id = sp

    # ── Partition by level (§3 step 1) ───────────────────────────────────────
    level_ids: dict[int, list[str]] = {}
    for rid, entity in rooms.items():
        lvl = _room_level(entity)
        level_ids.setdefault(lvl, []).append(rid)
    # Deterministic: sorted within each level.
    for lvl in level_ids:
        level_ids[lvl].sort()

    # ── Layout each level ─────────────────────────────────────────────────────
    # all_unit_pos: {room_id: (ux, uy)} for BFS-placed (non-pinned) rooms.
    # Pinned rooms are NOT in this dict (they use pinned_cells directly).
    all_unit_pos: dict[str, tuple[int, int]] = {}
    # unit_rects: {room_id: cell rect} for BFS-placed (non-pinned) rooms, centered
    # within their content-sized track band (per level). Pinned rooms are not here.
    unit_rects: dict[str, dict[str, int]] = {}
    for lvl in sorted(level_ids.keys()):
        level_room_ids = level_ids[lvl]
        level_rooms_dict = {rid: rooms[rid] for rid in level_room_ids}
        # Pass the approximate unit positions for pinned rooms to guide BFS.
        level_pinned_units = {
            rid: pinned_units[rid] for rid in level_room_ids if rid in pinned_units
        }
        # Start room only relevant if it's on this level.
        level_start = start_id if (start_id and start_id in level_rooms_dict) else None
        unit_pos = _layout_level(level_rooms_dict, level_start, level_pinned_units)

        # Normalize non-pinned rooms so the min x and min y are both 0.
        # Pinned rooms are excluded from normalization (their cell coords are absolute).
        non_pinned_pos = {
            rid: pos for rid, pos in unit_pos.items()
            if rid not in pinned_units
        }
        if non_pinned_pos:
            min_x = min(x for x, y in non_pinned_pos.values())
            min_y = min(y for x, y in non_pinned_pos.values())
            if min_x < 0 or min_y < 0:
                shift_x = -min_x if min_x < 0 else 0
                shift_y = -min_y if min_y < 0 else 0
                non_pinned_pos = {
                    rid: (ux + shift_x, uy + shift_y)
                    for rid, (ux, uy) in non_pinned_pos.items()
                }

        all_unit_pos.update(non_pinned_pos)

        # Content-sized track grid for this level plane + centered rects (§3).
        col_x, col_w, row_y, row_h = _compute_tracks(
            non_pinned_pos, room_sizes, room_w, room_h, gutter
        )
        for rid, (ux, uy) in non_pinned_pos.items():
            w, h = room_sizes[rid]
            unit_rects[rid] = _track_rect(ux, uy, w, h, col_x, col_w, row_y, row_h)

    # ── Build room records ────────────────────────────────────────────────────
    room_records: dict[str, Any] = {}
    for rid in sorted(rooms.keys()):
        entity = rooms[rid]
        lvl = _room_level(entity)
        ovr = entity_map_overrides.get(rid, {})
        is_pinned = rid in pinned_cells
        placement = "pinned" if is_pinned else "auto"

        # Build cell rect.
        if is_pinned:
            # Pinned rooms use their author-specified cell coordinates directly.
            cx, cy = pinned_cells[rid]
            w = int(ovr.get("width", room_w))
            h = int(ovr.get("height", room_h))
            rect = {"x": cx, "y": cy, "w": w, "h": h}
        else:
            rect = unit_rects.get(rid)
            if rect is None:
                # Room not placed (isolated with no embeddable edges).
                w, h = room_sizes[rid]
                rect = {"x": 0, "y": 0, "w": w, "h": h}

        # Art from per-room map overrides. Optional `offset: [px, px]` and
        # `scale: N` let an author nudge/zoom the background image within the
        # room rect (the client applies them atop its fit; the clip stays the
        # room rect). Absent keys keep the prior {src, fit} shape byte-identical.
        art: dict[str, Any] | None = None
        image_url = ovr.get("image")
        if image_url:
            art = {"src": str(image_url), "fit": "cover"}
            offset = ovr.get("offset")
            if (isinstance(offset, list) and len(offset) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in offset)):
                art["offset"] = [offset[0], offset[1]]
            scale = ovr.get("scale")
            if isinstance(scale, (int, float)) and not isinstance(scale, bool) and scale > 0:
                art["scale"] = scale

        # Palette.
        palette = str(ovr.get("palette", floor_palette))

        room_records[rid] = {
            "name": entity.name,
            "level": lvl,
            "shape": "rect",
            "rect": rect,
            "placement": placement,
            "art": art,
            "style": {"palette": palette},
        }

        # Render hints (presentation-only; passed through verbatim). Only emitted
        # when the room authored a non-empty `render:` dict, so floors without
        # hints keep byte-identical map.json output.
        render_hints = entity_render_hints.get(rid)
        if render_hints:
            room_records[rid]["render"] = render_hints

    # Helper: get cell rect for a room (works for both pinned and unit-placed).
    def _room_cell_rect(rid: str) -> dict[str, int] | None:
        if rid in pinned_cells:
            cx, cy = pinned_cells[rid]
            w, h = room_sizes[rid]
            return {"x": cx, "y": cy, "w": w, "h": h}
        return unit_rects.get(rid)

    # ── Build connections ─────────────────────────────────────────────────────
    connections: list[dict[str, Any]] = []
    # Track which (from, dir) pairs we've emitted to avoid duplicates within
    # the same room+direction.  We do NOT suppress a reverse exit — each room's
    # outbound exits are emitted independently so the VTT can drive movement in
    # both directions.
    emitted: set[tuple[str, str]] = set()

    # Iterate sorted: from_id, then dir_name (§3 step 7).
    for from_id in sorted(rooms.keys()):
        entity = rooms[from_id]
        exits = _room_exits(entity)
        from_level = _room_level(entity)
        from_rect = _room_cell_rect(from_id)
        # Convert cell rect to unit-like position for classify (we use cell rects
        # for the classify check now).
        from_unit = all_unit_pos.get(from_id)

        for dir_name in sorted(exits.keys()):
            exit_val = exits[dir_name]
            to_id, door_id = _exit_target_and_door(exit_val)

            if not to_id:
                continue
            # Skip exits pointing to non-room entities.
            if to_id not in rooms:
                continue

            # Deduplicate: skip if we already emitted this exact (from, dir) pair.
            key = (from_id, dir_name)
            if key in emitted:
                continue
            emitted.add(key)

            to_entity = rooms[to_id]
            to_level = _room_level(to_entity)
            to_unit = all_unit_pos.get(to_id)

            # Check if the reverse exit exists (bidirectional).
            # We do NOT suppress the reverse exit here — it will be emitted when
            # we iterate over to_id's exits.  one_way is false only when the
            # reverse exit actually points back to from_id.
            reverse_exits = _room_exits(to_entity)
            opposite_dir = _OPPOSITE.get(dir_name)
            reverse_target_id = None
            if opposite_dir and opposite_dir in reverse_exits:
                rev_target, _ = _exit_target_and_door(reverse_exits[opposite_dir])
                if rev_target == from_id:
                    reverse_target_id = from_id

            is_one_way = (reverse_target_id is None)

            conn = _classify_connection(
                from_id, dir_name, to_id, door_id,
                from_level, to_level,
                from_unit, to_unit,
                is_one_way,
            )

            # Route corridor paths (§3 step 5) off the rooms' centered rects.
            to_rect = _room_cell_rect(to_id)
            if (
                conn["kind"] in ("corridor", "door")
                and from_rect is not None
                and to_rect is not None
            ):
                conn["path"] = _route_path(from_rect, to_rect, dir_name)

            connections.append(conn)

    # Sort connections for determinism: by (from, dir).
    connections.sort(key=lambda c: (c["from"], c["dir"]))

    # ── Build levels array ────────────────────────────────────────────────────
    # Sort levels descending (0 first, then -1, -2, ...) to match the spec §2
    # example which shows the surface plane first.
    levels: list[dict[str, Any]] = []
    for lvl in sorted(level_ids.keys(), reverse=True):
        # Compute bounding box for this level from room_records (already built).
        max_col = 0
        max_row = 0
        for rid in level_ids[lvl]:
            r = room_records.get(rid)
            if r:
                rect = r["rect"]
                max_col = max(max_col, rect["x"] + rect["w"])
                max_row = max(max_row, rect["y"] + rect["h"])
        w = max(max_col + gutter, 1)
        h = max(max_row + gutter, 1)
        levels.append({"level": lvl, "width": w, "height": h})

    # ── Build layers ──────────────────────────────────────────────────────────
    overhead_layers: list[dict[str, Any]] = []
    for rid in sorted(rooms.keys()):
        entity = rooms[rid]
        ovr = entity_map_overrides.get(rid, {})
        overhead_url = ovr.get("overhead_image")
        if overhead_url:
            rect = _room_cell_rect(rid) or {"x": 0, "y": 0, "w": room_w, "h": room_h}
            overhead_layers.append({
                "src": str(overhead_url),
                "level": _room_level(entity),
                "rect": rect,
                "rooms": [rid],
            })

    # ── Token art ────────────────────────────────────────────────────────────
    token_art: dict[str, dict[str, str]] = {}
    for entity in floor.all_entities():
        url = _entity_token_url(entity)
        if url:
            token_art[entity.id] = {"src": url}
    # Sort by entity id for determinism.
    token_art = {k: token_art[k] for k in sorted(token_art.keys())}

    # ── floor_sha ─────────────────────────────────────────────────────────────
    floor_sha = hashlib.sha256(floor_lua_bytes).hexdigest()

    # ── Floor slug ────────────────────────────────────────────────────────────
    floor_slug = slugify(floor.name)

    # ── Assemble map.json dict ────────────────────────────────────────────────
    result: dict[str, Any] = {
        "map_version": 1,
        "floor": floor_slug,
        "floor_sha": floor_sha,
        "generator": f"clotho {_VERSION}",
        "grid": {"type": "square", "cell_px": cell_px},
        "levels": levels,
        "rooms": room_records,
        "connections": connections,
        "layers": {
            "overhead": overhead_layers,
            "hidden": [],
        },
        "tokens": {
            "art": token_art,
        },
    }

    return result


def emit_map_json(
    floor: Floor,
    floor_lua_bytes: bytes,
    *,
    indent: int = 2,
    default_cell_px: int = 64,
    default_room_w: int = 3,
    default_room_h: int = 3,
) -> str:
    """Convenience wrapper: returns the map.json as a JSON string.

    Determinism: json.dumps with sort_keys=False preserves insertion order,
    which is itself deterministic (we build dicts in a fixed order above).
    ensure_ascii=True avoids platform-dependent unicode encoding differences.
    """
    data = emit_map(
        floor,
        floor_lua_bytes,
        default_cell_px=default_cell_px,
        default_room_w=default_room_w,
        default_room_h=default_room_h,
    )
    return json.dumps(data, indent=indent, ensure_ascii=True)
