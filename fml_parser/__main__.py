"""CLI entry point for fml-parser.

Usage:
    python -m fml_parser lower <index.md> [-o <out.lua>]
    python -m fml_parser lower <index.md> --om --map -o <out.lua>
    python -m fml_parser lower <index.md> --om --map --art-manifest <manifest.json> -o <out.lua>
    python -m fml_parser --stdlib-module <index.md> [-o <out.lua>]
    python -m fml_parser artgen <index.md> [--provider curated|firefly] [--base-url /art] [-o <manifest.json>]

`-o -` or omitting -o writes to stdout.

Exit codes:
    0  success
    1  FML syntax / import error (structured message on stderr)
    2  bad invocation (argparse usage error)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from .errors import FmlImportError, FmlSyntaxError
from .emit_lua import emit_lua, emit_lua_graph, emit_lua_om, emit_lua_stdlib_module
from .emit_map import emit_map_json, strip_map_keys
from .parser import parse_fml


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fml-parser",
        description="Lower FML Markdown files to Luau LFR.",
    )

    # Mutually-exclusive mode flags: either a sub-command ("lower") or --stdlib-module.
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--stdlib-module",
        metavar="INDEX_MD",
        help="Emit a stdlib verb-module (emit_lua_stdlib_module) instead of a floor LFR.",
    )

    ap.add_argument(
        "-o",
        "--output",
        metavar="OUT_LUA",
        default="-",
        help="Output path (default: stdout).  Pass '-' for stdout.",
    )

    ap.add_argument(
        "--graph",
        action="store_true",
        default=False,
        help="Emit binding-surface (graph) LFR via emit_lua_graph instead of the legacy emit_lua.",
    )

    ap.add_argument(
        "--wyrd",
        "--om",
        dest="om",
        action="store_true",
        default=False,
        help="Emit wyrd (prototype object-model) LFR via emit_lua_om "
        "(for the wyrd engine dispatch path). "
        "Composes with 'lower' and --stdlib-module. Implies the graph binding surface. "
        "The flag --om is a working alias for backwards compatibility.",
    )

    ap.add_argument(
        "--map",
        action="store_true",
        default=False,
        help="Emit a map.json sidecar alongside the floor LFR (MAP_FORMAT.md §6). "
        "Requires 'lower --om' mode. "
        "map.json is written beside the -o output file, or to cwd when -o is '-'.",
    )

    ap.add_argument(
        "--art-manifest",
        metavar="MANIFEST_JSON",
        default=None,
        help="Path to an art manifest (produced by 'artgen') to merge into map.json. "
        "Requires --map. Each room listed in the manifest has its art field set to "
        '{"src": <image>, "fit": <fit>}. Rooms absent from the manifest are unchanged.',
    )

    # Positional sub-command + source path.
    ap.add_argument(
        "command",
        nargs="?",
        choices=["lower"],
        help="Lowering command.  Use 'artgen' for art-manifest generation (dispatched early).",
    )
    ap.add_argument(
        "source",
        nargs="?",
        metavar="INDEX_MD",
        help="Path to the FML index.md to lower.",
    )

    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI main -- returns an integer exit code."""
    if argv is None:
        argv = sys.argv[1:]

    # ── artgen subcommand ─────────────────────────────────────────────────────
    # Intercept before the main argparse runs so that artgen-specific flags
    # (--provider, --base-url) are parsed by artgen's own ArgumentParser.
    if argv and argv[0] == "artgen":
        from .artgen import artgen_main
        return artgen_main(argv[1:])

    ap = _build_parser()
    args = ap.parse_args(argv)

    # Resolve the mode and source path.
    if args.stdlib_module is not None:
        # --stdlib-module mode
        source_str = args.stdlib_module
        mode = "stdlib"
    elif args.command == "lower":
        if args.source is None:
            ap.error("the 'lower' command requires a source INDEX_MD argument")
        source_str = args.source
        mode = "floor"
    else:
        ap.error("specify either 'lower <index.md>', 'artgen <index.md>', "
                 "or '--stdlib-module <index.md>'"
                 )
        return 2  # unreachable; ap.error() calls sys.exit(2)

    emit_map_flag: bool = args.map
    art_manifest_path: str | None = args.art_manifest

    # Validate --map usage.
    if emit_map_flag:
        if mode != "floor":
            print(
                "fml-parser: error: --map requires 'lower' mode (not --stdlib-module)",
                file=sys.stderr,
            )
            return 2
        if not args.om:
            print(
                "fml-parser: error: --map requires --om (the wyrd object-model emitter)",
                file=sys.stderr,
            )
            return 2

    # Validate --art-manifest usage.
    if art_manifest_path is not None and not emit_map_flag:
        print(
            "fml-parser: error: --art-manifest requires --map",
            file=sys.stderr,
        )
        return 2

    source_path = Path(source_str).resolve()

    # Read the FML source.
    try:
        text = source_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(
            f"fml-parser: error: source file not found: {source_path}",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"fml-parser: error: cannot read {source_path}: {exc}", file=sys.stderr)
        return 1

    # Parse.
    try:
        floor = parse_fml(text, source_path=source_path)
    except FmlSyntaxError as exc:
        print(f"fml-parser: syntax error: {exc}", file=sys.stderr)
        return 1
    except FmlImportError as exc:
        print(f"fml-parser: import error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"fml-parser: unexpected error during parse: {exc}", file=sys.stderr)
        return 1

    # When --map is active, snapshot the `map:` / `token:` presentation keys
    # BEFORE stripping them from the floor.  We need:
    #   (a) the LFR (floor.lua) bytes for the sha256 — which requires a
    #       stripped floor so Wyrd never sees presentation data.
    #   (b) the map/token keys themselves — stripped from the floor before (a).
    # Solution: deep-copy the relevant slices, strip the floor, emit Lua to get
    # the bytes, then call emit_map_json with the floor re-hydrated from the
    # snapshot (then re-strip).  The floor ends up presentation-free in all cases.
    if emit_map_flag:
        _floor_map_snap = copy.deepcopy(floor.properties.get("map", {}))
        _entity_map_snap: dict = {}
        _entity_token_snap: dict = {}
        for entity in floor.all_entities():
            em = entity.properties.get("map")
            if isinstance(em, dict):
                _entity_map_snap[entity.id] = copy.deepcopy(em)
            tk = entity.properties.get("token")
            if tk is not None:
                _entity_token_snap[entity.id] = tk

    # Strip map/token keys so the Lua emitter never sees them (MAP_FORMAT §4).
    strip_map_keys(floor)

    # Emit Lua.
    try:
        if mode == "stdlib" and args.om:
            # NOTE (P6a): the stdlib+om path emits a structural module but does
            # NOT populate the om kind:<name> registry the floor's _proto helper
            # reads — the emitted LFR self-flags this with a WARNING comment.
            # Full om stdlib lowering (kind nodes + behaviour fragments) is the
            # deferred behaviour-port phase; see emit_lua_om docstring.
            lua_source = emit_lua_om(
                floor, source_path=str(source_path), stdlib_module=True
            )
        elif args.om:
            lua_source = emit_lua_om(floor, source_path=str(source_path))
        elif mode == "stdlib" and args.graph:
            lua_source = emit_lua_graph(
                floor, source_path=str(source_path), stdlib_module=True
            )
        elif mode == "stdlib":
            lua_source = emit_lua_stdlib_module(floor, source_path=str(source_path))
        elif args.graph:
            lua_source = emit_lua_graph(floor, source_path=str(source_path))
        else:
            lua_source = emit_lua(floor, source_path=str(source_path))
    except Exception as exc:
        print(f"fml-parser: emit error: {exc}", file=sys.stderr)
        return 1

    # Write Lua output.
    out = args.output
    if out == "-":
        sys.stdout.write(lua_source)
    else:
        out_path = Path(out)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(lua_source, encoding="utf-8")
        except OSError as exc:
            print(
                f"fml-parser: error: cannot write {out_path}: {exc}",
                file=sys.stderr,
            )
            return 1

    # Emit map.json sidecar if --map was requested.
    if emit_map_flag:
        # Re-hydrate the floor with the snapshotted presentation keys so
        # emit_map_json can read them, then re-strip after.
        if _floor_map_snap:
            floor.properties["map"] = _floor_map_snap
        for entity in floor.all_entities():
            if entity.id in _entity_map_snap:
                entity.properties["map"] = _entity_map_snap[entity.id]
            if entity.id in _entity_token_snap:
                entity.properties["token"] = _entity_token_snap[entity.id]

        # Load art manifest if provided.
        art_manifest: dict | None = None
        if art_manifest_path is not None:
            try:
                art_manifest = json.loads(
                    Path(art_manifest_path).read_text(encoding="utf-8")
                )
            except FileNotFoundError:
                print(
                    f"fml-parser: error: art manifest not found: {art_manifest_path}",
                    file=sys.stderr,
                )
                return 1
            except (OSError, json.JSONDecodeError) as exc:
                print(
                    f"fml-parser: error: cannot read art manifest "
                    f"{art_manifest_path}: {exc}",
                    file=sys.stderr,
                )
                return 1

        try:
            lua_bytes = lua_source.encode("utf-8")
            map_data = json.loads(emit_map_json(floor, lua_bytes))

            # Merge art manifest into map.json (Part 2 of spec).
            if art_manifest is not None:
                _merge_art_manifest(map_data, art_manifest)

            map_json_str = json.dumps(map_data, indent=2, ensure_ascii=True)
        except Exception as exc:
            print(f"fml-parser: map emit error: {exc}", file=sys.stderr)
            return 1
        finally:
            # Re-strip so the floor is clean in all exit paths.
            strip_map_keys(floor)

        # Determine map.json output path.
        if out == "-":
            # stdout mode: write map.json to cwd (documented behaviour).
            map_path = Path("map.json")
        else:
            map_path = Path(out).parent / "map.json"

        try:
            map_path.parent.mkdir(parents=True, exist_ok=True)
            map_path.write_text(map_json_str, encoding="utf-8")
        except OSError as exc:
            print(
                f"fml-parser: error: cannot write {map_path}: {exc}",
                file=sys.stderr,
            )
            return 1

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Art manifest merge helper
# ─────────────────────────────────────────────────────────────────────────────


def _merge_art_manifest(map_data: dict, manifest: dict) -> None:
    """Merge ``manifest`` into ``map_data`` in place.

    For each ``<room_id>`` in ``manifest["rooms"]``, set
    ``map_data["rooms"][<room_id>]["art"] = {"src": <image>, "fit": <fit>}``.

    For each entry in ``manifest["tokens"]``, merge into
    ``map_data["tokens"]["art"][<id>] = {"src": <image>}``.

    Rooms / tokens absent from the manifest are left unchanged.
    Rooms in the manifest that don't appear in map_data are silently skipped.
    """
    manifest_rooms: dict = manifest.get("rooms", {})
    for room_id, art_record in manifest_rooms.items():
        room = map_data.get("rooms", {}).get(room_id)
        if room is None:
            continue  # Room not in this floor's map — skip.
        room["art"] = {
            "src": art_record["image"],
            "fit": art_record.get("fit", "tile"),
        }

    manifest_tokens: dict = manifest.get("tokens", {})
    token_art: dict = map_data.setdefault("tokens", {}).setdefault("art", {})
    for token_id, token_record in manifest_tokens.items():
        token_art[token_id] = {"src": token_record["image"]}


if __name__ == "__main__":
    sys.exit(main())
