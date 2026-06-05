"""CLI entry point for fml-parser.

Usage:
    python -m fml_parser lower <index.md> [-o <out.lua>]
    python -m fml_parser --stdlib-module <index.md> [-o <out.lua>]

`-o -` or omitting -o writes to stdout.

Exit codes:
    0  success
    1  FML syntax / import error (structured message on stderr)
    2  bad invocation (argparse usage error)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import FmlImportError, FmlSyntaxError
from .emit_lua import emit_lua, emit_lua_graph, emit_lua_stdlib_module
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

    # Positional sub-command + source path.
    ap.add_argument(
        "command",
        nargs="?",
        choices=["lower"],
        help="Lowering command.  Currently only 'lower' is defined.",
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
        ap.error("specify either 'lower <index.md>' or '--stdlib-module <index.md>'")
        return 2  # unreachable; ap.error() calls sys.exit(2)

    # --graph selects the binding-surface (LPG) emitter. It composes with BOTH
    # modes: `lower --graph` emits a floor LFR (instances), `--stdlib-module
    # --graph` emits a stdlib LFR (schema + verbs).
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

    # Emit.
    try:
        if mode == "stdlib" and args.graph:
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

    # Write output.
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
