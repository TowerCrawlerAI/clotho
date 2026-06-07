# fml-parser

The TowerCrawlerAI FML → LFR emitter. Deterministic Python parser that reads Floor Markdown Language (`.md`) files, walks the import graph, applies kind-chain inheritance + tree-shake, and emits Luau LFR (`.lua`) that the `engine-core` binary consumes.

## Build / install

```bash
pip install -e .
```

Dependencies (declared in `pyproject.toml`):
- `markdown-it-py` — the literate Markdown parser
- `pydantic` — entity model validation

## Use

```python
from fml_parser.parser import parse_file
from fml_parser.emit_lua import emit_lua

floor = parse_file("path/to/index.md")
lua_source = emit_lua(floor)
open("output.lua", "w").write(lua_source)
```

Or via CLI:
```bash
# Legacy flat LFR (entity-table format)
python -m fml_parser lower path/to/index.md -o output.lua

# Graph binding-surface LFR (engine.create_node / engine.relate)
python -m fml_parser lower path/to/index.md --graph -o output.lua

# Wyrd (prototype object-model) LFR — emits wyrd.* table calls
python -m fml_parser lower path/to/index.md --wyrd -o output.lua
# --om is a backwards-compatible alias for --wyrd (identical output):
python -m fml_parser lower path/to/index.md --om -o output.lua

# Stdlib verb-module
python -m fml_parser --stdlib-module path/to/index.md -o stdlib.lua
```

## Architecture

- `parser.py` — top-level entry. Walks the import graph (`[stdlib]`-style Markdown links), parses each file exactly once, returns a `Floor` model.
- `emit_lua.py` — lowers the parsed `Floor` to Luau. Applies kind-chain inheritance at lower time. Tree-shakes unreferenced catalog entries.
- `models.py` — Pydantic models for `Floor`, `Entity`, `SubEntity`, `Trigger`, properties, etc.
- `dice_value.py` — six-form FML property value dispatch (PR-equivalent of the dice-as-first-class work).
- `emitter.py` — emission helpers (string formatting, Luau-safe identifiers, etc.).
- `errors.py` — typed parser errors with file/line context.
- `lua_reader/` — round-trip reader (LFR → models, for `tower unlower`).
- `tree_shake.py` — BFS reachability pass; prunes unreachable catalog entries.

## Companion repos

Under `TowerCrawlerAI/`:
- [`engine-core`](../engine-core) — C engine that consumes the LFR this parser emits
- [`stdlib`](../stdlib) — FML stdlib that gets imported into floors
- [`sample-dungeon`](../sample-dungeon) — Bone Garden test content
- [`wiki`](../wiki) — `design/FML.md`, `design/LFR.md`, `design/PARSER.md` for the normative spec

## FML spec

Authoritative: `wiki/design/FML.md`. Key invariants:

- H1 = document title (exactly one per file)
- H3 = entity declaration
- H5 = sub-entity
- H6 = trigger handler (stage + event)
- H2 / H4 = convention only
- Property bullets outside blockquotes; prose inside blockquotes
- Imports are include-once with inline expansion
- Trigger stage names are `Test`, `Before`, `On`, `After`, `Report`, `InsteadOf`

## Determinism

Same FML input + same import graph → bit-identical LFR. No LLMs in the build path. No randomness. This is load-bearing for skein record/replay further downstream.

## Releasing

Before cutting a release tag (`git tag vX.Y.Z && git push --tags`), the
**Integration Gate** CI job (`integration-gate.yml`) **must be green**.

That workflow (`F3`, closes #4) checks out `TowerCrawlerAI/stdlib` and
`TowerCrawlerAI/sample-dungeon` and lowers both with the *candidate* parser:

```bash
# stdlib verb-module
python -m fml_parser --stdlib-module stdlib/core/index.md -o stdlib.lua

# floor LFR
python -m fml_parser lower sample-dungeon/index.md -o floor.lua
```

Both commands must exit 0 and produce a non-empty `.lua` file.  If either
fails, the gate blocks and the tag **must not** be pushed until the failure
is fixed and the gate is green again.

## Status

Carried over from the TowerAI monorepo split on 2026-06-02.
