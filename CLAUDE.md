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

Additive: floors without these keys lower byte-identically. See `emit_lua.py`
`_parse_cell` / `_parse_blocked_cells`; spec in `../wiki/design/FML.md` §8.3.

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
