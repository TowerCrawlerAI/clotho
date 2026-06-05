# F6 — Binding-surface (graph) emitter

**Status:** spec (architect). Implements the `--graph` lowering target for the LPG engine
(`crawler --engine graph`). Scope decision: **lean emitter + stdlib rewrites** — emit emergent verbs +
native-Lua stages; do NOT transpile the FML trigger sub-language. Content that needs real logic carries
an explicit `lua`/`luau` script block; everything else is emergent.

## Target: the engine binding surface

The engine loads a **DB-init LFR**: a Luau chunk that calls `engine.*` bindings to populate the graph and
register verbs, then designates the start actor. The chunk is run for its side effects — **no return value
is required** (the loader, post engine-core PR #37, treats absent/non-integer return as success). Authoritative
contract (engine-core `src/luau_bindings.cpp`):

| Binding | Signature | Notes |
|---|---|---|
| `engine.define_relation(name, opts?)` | `opts = {src_card, dst_card, symmetric, role, has_payload, acyclic, excl_group}` → rel_id | cards: `"one"`/`"various"` (default various/various). Built-ins are pre-registered — see below. |
| `engine.define_kind(name)` | → node id | **Stub today**: mints a node with a `name` prop. Records kind names; no inheritance. |
| `engine.create_node(props?)` | `props` = table of scalar (string/int/float/bool) initial props → node id (int > 0) | Non-scalar values silently ignored — set those via `set_prop` after, or skip. |
| `engine.set_prop(id, key, value)` | value = bool/number/string | For props not set at create time. |
| `engine.relate(rel, a, b)` | `rel` = name or id; `a`,`b` = integer node ids → true | edge `a --rel--> b`. Raises on cycle/cardinality violation. |
| `engine.define_verb(spec)` | `spec = {name, noun, noun2, preposition, scope, target_rel, subject_is_src, aliases, stages}` → verb_id | see Verbs below |
| `engine.set_start_actor(id)` | — | records the player actor for graph-mode turn_loop |

### Built-ins — DO NOT re-emit

Bootstrapped at engine create (`tg_rel_bootstrap` / `tg_verb_bootstrap`):

- **Relations:** `in`, `on`, `carried`, `worn` (the `location` exclusion group), `part-of`. A custom
  emitter relation like **`map`** (room adjacency) is **not** built-in and **must** be `define_relation`d.
- **Verbs:** `take`, `drop`, `put-in`, `go`. Re-defining any of these via `define_verb` **fails**
  (duplicate name). The stdlib rewrite drops/omits these four; the engine provides them.

`go` resolves movement over **`map`** edges: `go <room>` succeeds if the destination room is adjacent
(`relate("map", current_room, dest)`). A floor's `exits` therefore lower to `map` edges (see World).

## Two emission concerns

The emitter has a stdlib (schema+verbs) half and a floor (world-instances) half. A single
`emit_lua_graph(floor)` handles both; which parts fire depends on what the `Floor` contains.

### Schema + verbs (stdlib half)

1. **Kinds** — one `engine.define_kind(name)` per `kind_definition` entity, in `section_mappings` order.
2. **Custom relations** — `engine.define_relation(name, opts)` for any relation the content needs beyond
   the built-ins. At minimum, emit `map` (`{symmetric=true, role="map"}`) when any entity declares `exits`.
3. **Verbs** — one `engine.define_verb{...}` per `kind=="verb"` entity, EXCEPT the four built-ins
   (`take/drop/put-in/go`) which are skipped. Grammar fields map from verb properties:
   - `noun`/`noun2` ← verb props `noun`/`noun2` (`"required"`/`"optional"`/`"none"`).
   - `preposition` ← prop `preposition`.
   - `scope` ← prop `scope` (sense: `see`/`touch`/`hear`/`smell`/`know`; default touch).
   - `target_rel` ← prop `target_rel` (must be a known relation: built-in or define_relation'd).
   - `subject_is_src` ← prop `subject_is_src` (bool).
   - `aliases` ← prop `aliases`/`phrases` (string or list) + matching `understand_directives`.
   - `stages` ← **only** from triggers carrying a `lua`/`luau` `Script`. For each such trigger, emit
     `<stage_key> = function(ctx) <script.source verbatim> end`, where `stage_key` = `_verb_stage_key(trigger.name)`.
     Triggers with an FML body and **no** script are **not** transpiled (lean scope): emit nothing for them
     and `log`/comment a warning so the stdlib author knows to rewrite that verb (emergent or native-lua).
   - A verb with `target_rel` and no stages → emergent (the engine derives the edge write). This is the
     common case and the goal of the rewrite.

### World instances (floor half)

For each world entity (everything that is not `kind_definition`/`verb`/`event`/other schema kinds — i.e.
rooms, items, actors):

1. `local n_<id> = engine.create_node({ name = <name>, <scalar props> })` — emit scalar props inline;
   skip `location`, `exits`, and non-scalar props (handled below / via set_prop).
2. **Containment** — the entity's `location` slug → `engine.relate("in", n_<id>, n_<location>)`.
   (Lean default: `in`. A later pass may honor an explicit `on`/`worn`/`carried` hint property.)
3. **Exits** — `exits = {dir = room_slug, ...}` → `engine.relate("map", n_<room>, n_<dest>)` per entry.
   Requires the `map` relation (emit its `define_relation` once, before any `relate("map", ...)`).
4. **Start actor** — the floor's player/start entity (the entity of the player kind, or a floor
   `start`/`player` property) → `engine.set_start_actor(n_<id>)`.

Emission order matters: define_relation/define_kind first, then create_node for every entity (so all
`n_<id>` locals exist), then relate (location + exits), then define_verb, then set_start_actor.

## Determinism

Same FML in → bit-identical LFR out (the skein/replay contract). Iterate entities/props/relations in a
stable order (declared/section order, never dict-hash order). No randomness, no timestamps.

## CLI

Add a mode to `python -m fml_parser`: `--graph` (or `lower --graph <index.md>`) selecting
`emit_lua_graph`. Keep the legacy `emit_lua` / `--stdlib-module` paths intact (the old engine path still
uses them) until the graph engine is the default.

## Out of scope (this pass)

- FML trigger-body transpilation (if/while/loop/output/property-set/action-line → binding surface).
  Deferred; content needing logic uses explicit `lua` script stages instead.
- `define_kind` inheritance (engine stub today).
- Directional exit *labels* as first-class (lean pass maps exits to undirected/`map` adjacency; `go <room>`).

## Validation gate

The emitter is proven by lowering a small synthetic floor (and then `stdlib/core` in B3.2, the Bone Garden
in B3.3) and loading the result on the **real** `crawler --engine graph` binary over the JSONL contract —
not just by Python unit tests. A unit test asserting the emitted text shape is necessary but not sufficient.
