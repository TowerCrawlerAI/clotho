# fml-parser — CLAUDE.md

Python tool that **lowers FML markdown → Luau LFR** — the format the engine (`wyrd`) loads. Part of the
TowerCrawlerAI workspace; see the root `CLAUDE.md` for the multi-repo map + build/release pipeline, and
`../wyrd/docs/design/ENGINE_MODEL.md` for the engine the output targets.

## What it does

Deterministic, LLM-free lowering: same FML input → **bit-identical** LFR. No randomness, no network, no
LLMs in the build path.

## CLI

- `python -m fml_parser lower <index.md> -o out.lua` → a **floor LFR** (via `emit_lua`).
- `python -m fml_parser --stdlib-module <index.md> -o out.lua` → a **stdlib verb-module** (via `emit_lua_stdlib_module`).
- `-o -` (or omitted) → stdout. Non-zero exit + structured stderr on `FmlSyntaxError` / `FmlImportError`.

(Console entry point `fml-parser`; added in F1, #9.)

## Package layout (`fml_parser/`)

- `parser.py` — `parse_fml(text, source_path)`: walks the import graph, resolves kinds, returns a `Floor`.
- `emit_lua.py` — `emit_lua(floor)` (floor LFR) + `emit_lua_stdlib_module(floor)` (verb module).
- `emitter.py` — Luau emission helpers.
- `models.py` — Pydantic models (Floor, Entity, Trigger, …) + validation.
- `dice_value.py` — the FML property-value forms (incl. dice expressions).
- `trigger_body_parser.py`, `predicate_lang.py` — trigger bodies + `when` / completion predicates.
- `tree_shake.py` — prune unreachable catalog entries.
- `lua_reader/` — round-trip reader (LFR → models), for `unlower`.
- `slugify.py`, `errors.py`, `__main__.py` (CLI).

## Spatial authoring (§22 Phase 5)

The floor emitter lowers two optional spatial properties (engine reqs #104/#108):

- `position: [x, y, z]` on an entity that has a `location:`/`at_location:` →
  `engine.relate("in", n_<id>, n_<container>, x, y, z)` (the integer cell payload
  on the location edge). Integer cells only, range ±2^20.
- `blocked: [[x,y,z], [x,y,z,kind], …]` on a container →
  `engine.set_blocked(n_<id>, x, y, z[, flags])`. `kind` ∈ `wall` (default; move+sight),
  `move`, `sight`. Both validate at lower time (FmlSyntaxError on malformed).

Entrances (engine req #119) — where an arriving actor lands (else the container origin):

- `entrance: [x, y, z]` on a room → `engine.set_entrance(n_<room>, x, y, z)` (its default
  spawn cell). A mapped room with no authored `entrance` gets a conservative default
  `(0, 1, 0)` (one cell south of centre) injected by `strip_map_keys`. Cells are
  **centre-origin** (cell (0,0,0) = room centre, +y = south, per the client's
  `_render_pos`). The default is deliberately small because a room's render rect is
  content-sized and does NOT track the `map:` image dimensions, so an offset derived from
  map width/height can fall off the rect. Authors set `entrance:` explicitly for a precise
  spawn cell (e.g. the far south edge of a large pinned battlemap).
- per-exit `enter_at: [x,y,z]` in the object-form exit `<dir>: {room, enter_at}` →
  `engine.set_exit_entry(n_<src>, "<dir>", x, y, z)` (overrides the destination's default
  when leaving via `<dir>`).
- `map:` gains optional `fit:` (cover | contain | tile | stretch; default cover),
  `offset: [px, px]`, and `scale: N` → the room's `art` record in `map.json`
  (`{src, fit, offset, scale}`); presentation-only, never reaches the LFR. Background
  rendering is independent of the cell grid — `tile` repeats, `stretch` fills the rect
  ignoring aspect.

Additive: floors without these keys lower byte-identically. See `emit_lua.py`
`_parse_cell` / `_parse_blocked_cells` / `_exit_enter_at`, `emit_map.py`
`_inject_default_entrance`; spec in `../wiki/design/FML.md` §8.3.

## Output contracts (do not drift)

- LFR + stdlib-verb-module shapes: `../wyrd/docs/design/ENGINE_MODEL.md` §9 (verbs as stored
  procedures), §10 (built-in relations), §11 (binding surface); `CoreRequirements.md` #14 (verb-table shape).
- **Determinism is load-bearing** (the skein/replay contract): no LLM, no `random`, stable ordering.

## Testing

`pytest -q` (CI: `.github/workflows/ci.yml`, F2 #10). Add a test with every behavior change.

## Coming (open tickets)

- **F6** — rework the emitter to the relation / emergent-verb model (verbs = relation-targeted stored
  procedures) once engine EG2/EG5 land.
- **F4** — versioned releases; **F3** — the pre-release gate (lower stdlib + sample-dungeon with the
  candidate parser; block the tag unless both lower cleanly).

## Editing

Normal git repo, base `main`. Work on a branch, open a PR. Never commit `.claude/` or build artifacts.
