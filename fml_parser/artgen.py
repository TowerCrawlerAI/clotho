"""Art-generation pipeline for the VTT floor backgrounds.

This module provides the `artgen` CLI subcommand: a fully OFFLINE step that
produces an art-manifest JSON file mapping room IDs to image URLs.  The
manifest is later consumed by `lower --art-manifest` to inject `room.art`
into `map.json` deterministically.

CLI:
    python -m fml_parser artgen <index.md> \\
        [--provider curated|firefly] \\
        [--base-url /art] \\
        [-o <manifest.json>]

Output manifest shape::

    {
      "version": 1,
      "rooms": { "<room_id>": {"image": "<url>", "fit": "tile"} },
      "tokens": {}
    }

Rooms are written in sorted id order for determinism.

Provider protocol:

    ``ArtProvider.resolve(room: FMLEntity) -> dict | None``

Returns ``{"image": "<url>", "fit": "<fit>"}`` or ``None`` when the provider
cannot resolve art for the given room.

Adding a new provider: subclass (or duck-type) ``ArtProvider``, implement
``resolve(room)``, and register it in ``_PROVIDERS``.

Determinism guarantee:
    No LLM, no ``random``, no network I/O for the default ``curated``
    provider.  Given the same floor, the same base-url, and the same
    provider, the manifest is byte-identical across runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Protocol

from .models import FMLEntity


# ─────────────────────────────────────────────────────────────────────────────
# Provider protocol
# ─────────────────────────────────────────────────────────────────────────────


class ArtProvider(Protocol):
    """Interface every art provider must implement.

    ``resolve`` maps a room entity to an art record or ``None``.

    Return value shape::

        {"image": "<absolute-or-relative URL>", "fit": "tile"|"cover"|"contain"}

    Return ``None`` to indicate the provider has no art for this room
    (it will be omitted from the manifest).
    """

    def resolve(self, room: FMLEntity) -> dict | None:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Curated provider — CC0 textures, fully offline
# ─────────────────────────────────────────────────────────────────────────────

# Texture names (without extension) for CC0 tile files at <base-url>/<name>.jpg.
_CURATED_TEXTURES = {
    "wet_stone",
    "silt",
    "ashlar",
    "dark_stone",
    "flagstone",
}


def _coerce_bool(v: object) -> bool:
    """Coerce an FML property value to bool.

    FML properties may be Python ``True``/``False``, the strings ``"true"``
    / ``"false"``, or absent.  This helper handles all cases.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1")
    return bool(v)


class CuratedProvider:
    """Map room properties → one of five CC0 texture names.

    Precedence order (first match wins):
    1. ``water`` truthy            → ``wet_stone``
    2. ``outdoor`` truthy          → ``silt``
    3. ``level`` ≤ −2              → ``ashlar``
    4. ``underground`` truthy OR ``level`` < 0  → ``dark_stone``
    5. default                     → ``flagstone``

    All textures use ``fit: tile`` because they are seamless-texture JPEGs.
    """

    def __init__(self, base_url: str = "/art") -> None:
        # Strip trailing slash for clean URL construction.
        self._base = base_url.rstrip("/")

    def resolve(self, room: FMLEntity) -> dict | None:
        props = room.properties

        # 1. water
        if _coerce_bool(props.get("water")):
            name = "wet_stone"
        # 2. outdoor
        elif _coerce_bool(props.get("outdoor")):
            name = "silt"
        # 3. level ≤ −2
        elif self._room_level(props) <= -2:
            name = "ashlar"
        # 4. underground OR level < 0
        elif _coerce_bool(props.get("underground")) or self._room_level(props) < 0:
            name = "dark_stone"
        # 5. default
        else:
            name = "flagstone"

        return {
            "image": f"{self._base}/{name}.jpg",
            "fit": "tile",
        }

    @staticmethod
    def _room_level(props: dict) -> int:
        """Extract the integer ``level`` property, defaulting to 0."""
        v = props.get("level", 0)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Firefly provider — stub with real prompt-builder
# ─────────────────────────────────────────────────────────────────────────────


def build_room_prompt(room: FMLEntity) -> str:
    """Build a top-down battlemap generation prompt for ``room``.

    Composes the room's name, prose description, and relevant material/mood
    properties into a prompt string suitable for a text-to-image API.  The
    prompt targets seamless battlemap texture art: overhead view, no grid,
    no UI chrome.

    Args:
        room: The parsed FML room entity.

    Returns:
        A single-line prompt string ready to pass to an image-generation API.
    """
    name = room.name or room.id
    props = room.properties

    # Gather prose (may be a str or a ProseValue duck-type).
    prose_raw = room.prose
    if hasattr(prose_raw, "lines"):
        # ProseValue: join lines.
        desc = " ".join(str(line) for line in prose_raw.lines).strip()
    else:
        desc = str(prose_raw).strip() if prose_raw else ""

    # Build material / mood cues from properties.
    cues: list[str] = []
    if props.get("water"):
        cues.append("wet stone, standing water, dripping ceiling")
    if props.get("outdoor"):
        cues.append("open sky, natural ground, silt or earth floor")
    level = _room_level_from_props(props)
    if level <= -2:
        cues.append("deep underground, dry ancient stone, ashlar masonry")
    elif props.get("underground") or level < 0:
        cues.append("underground dungeon, dark stone walls, torchlight")
    if props.get("luminosity") is not None:
        try:
            lum = int(props["luminosity"])
        except (TypeError, ValueError):
            lum = 5
        if lum == 0:
            cues.append("pitch black, total darkness")
        elif lum < 5:
            cues.append("dim light, shadowy")
        else:
            cues.append("well lit")

    cue_str = ", ".join(cues) if cues else "stone dungeon interior"

    # Truncate description to a reasonable length to keep the prompt clean.
    if len(desc) > 200:
        desc = desc[:197] + "..."

    prompt = (
        f"top-down dungeon battlemap floor, {name}, {desc}, "
        f"{cue_str}, seamless texture, no grid, overhead view, fantasy RPG"
    )
    return prompt


def _room_level_from_props(props: dict) -> int:
    """Extract integer ``level`` from a properties dict, defaulting to 0."""
    v = props.get("level", 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


class FireflyProvider:
    """Adobe Firefly image-generation provider — stub with real prompt-builder.

    The provider builds a complete, ready-to-use prompt via
    :func:`build_room_prompt`, but raises :exc:`NotImplementedError` at
    resolve-time because no API credentials or CDN upload are wired here.

    To activate this provider in a production deployment:
    1. Obtain an Adobe Firefly API key (https://developer.adobe.com/firefly-api/).
    2. Call the ``/v3/images/generate`` endpoint with the prompt returned by
       ``build_room_prompt`` (or this provider's ``_build_prompt`` method).
    3. Upload the resulting image bytes to a CDN or static-file server.
    4. Replace the ``raise NotImplementedError(...)`` below with a call to
       your upload helper and return the resulting URL.

    The intent is that dropping an API key into the environment activates
    the full pipeline with no other changes.
    """

    def resolve(self, room: FMLEntity) -> dict | None:
        """Build the prompt and raise NotImplementedError (API not wired).

        The raised error message includes the ready-to-use prompt so callers
        can inspect it without running the provider.

        Raises:
            NotImplementedError: Always, with the generated prompt embedded.
        """
        prompt = build_room_prompt(room)
        # ── Where the API call and CDN upload would go ────────────────────
        # response = firefly_client.generate_image(prompt=prompt, ...)
        # url = cdn_upload(response.image_bytes, key=f"rooms/{room.id}.jpg")
        # return {"image": url, "fit": "cover"}
        # ─────────────────────────────────────────────────────────────────
        raise NotImplementedError(
            f"firefly provider needs an image API key + uploader; "
            f"prompt was: {prompt}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Provider registry
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type] = {
    "curated": CuratedProvider,
    "firefly": FireflyProvider,
}


def make_provider(name: str, *, base_url: str = "/art") -> ArtProvider:
    """Instantiate a provider by name.

    Args:
        name: Provider identifier (``"curated"`` or ``"firefly"``).
        base_url: Base URL prefix for the curated provider's image URLs.
            Ignored by other providers.

    Returns:
        A provider instance implementing the :class:`ArtProvider` protocol.

    Raises:
        ValueError: If ``name`` is not a registered provider.
    """
    cls = _PROVIDERS.get(name)
    if cls is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown provider {name!r}; known providers: {known}")
    # Only the curated provider accepts base_url; others use **kwargs guard.
    if name == "curated":
        return cls(base_url=base_url)
    return cls()


# ─────────────────────────────────────────────────────────────────────────────
# Manifest emitter
# ─────────────────────────────────────────────────────────────────────────────


def generate_manifest(
    floor,  # Floor — avoid circular import; typed as Any at runtime
    provider: ArtProvider,
) -> dict:
    """Resolve art for every room in ``floor`` and return a manifest dict.

    Only ``room``-kind entities (those where ``entity.is_a("room")`` or
    ``entity.kind == "room"``) are processed.  Rooms for which the provider
    returns ``None`` are omitted from the manifest.

    Room IDs are emitted in sorted order for determinism.

    Args:
        floor: A :class:`~fml_parser.models.Floor` instance (parsed FML).
        provider: An :class:`ArtProvider` instance.

    Returns:
        A dict with keys ``"version"``, ``"rooms"``, ``"tokens"``
        suitable for ``json.dumps``.
    """
    rooms_out: dict[str, dict] = {}

    # Collect all room entities, in sorted ID order for determinism.
    room_entities = [
        entity
        for entity in floor.entities.values()
        if _is_room_entity(entity)
    ]
    room_entities.sort(key=lambda e: e.id)

    for entity in room_entities:
        art = provider.resolve(entity)
        if art is not None:
            rooms_out[entity.id] = art

    return {
        "version": 1,
        "rooms": rooms_out,
        "tokens": {},
    }


def _is_room_entity(entity: FMLEntity) -> bool:
    """True if entity is a room (kind == 'room' or kind_chain contains 'room')."""
    if entity.kind == "room":
        return True
    if entity.kind_chain and "room" in entity.kind_chain:
        return True
    # Fallback: has an ``exits`` property (same heuristic as emit_map.py).
    return "exits" in entity.properties


def emit_manifest_json(manifest: dict, *, indent: int = 2) -> str:
    """Serialise a manifest dict to a deterministic JSON string.

    Uses ``sort_keys=False`` (we control insertion order) and
    ``ensure_ascii=True`` to avoid platform-dependent unicode differences.
    """
    return json.dumps(manifest, indent=indent, ensure_ascii=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (called from __main__.py)
# ─────────────────────────────────────────────────────────────────────────────


def artgen_main(argv: list[str]) -> int:
    """CLI handler for the ``artgen`` subcommand.

    Args:
        argv: Argument list AFTER the ``artgen`` token has been stripped.

    Returns:
        Exit code (0 = success, 1 = error, 2 = bad invocation).
    """
    import argparse

    from .errors import FmlImportError, FmlSyntaxError
    from .parser import parse_fml

    ap = argparse.ArgumentParser(
        prog="fml-parser artgen",
        description="Generate an art manifest for VTT room backgrounds.",
    )
    ap.add_argument("source", metavar="INDEX_MD",
                    help="Path to the FML index.md to process.")
    ap.add_argument(
        "--provider",
        default="curated",
        choices=list(_PROVIDERS),
        help="Art provider (default: curated).",
    )
    ap.add_argument(
        "--base-url",
        default="/art",
        metavar="URL",
        help="Base URL prefix for curated texture images (default: /art).",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="-",
        metavar="MANIFEST_JSON",
        help="Output path (default: stdout).  Pass '-' for stdout.",
    )

    args = ap.parse_args(argv)

    source_path = Path(args.source).resolve()

    # Read source.
    try:
        text = source_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"fml-parser artgen: error: source file not found: {source_path}",
              file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"fml-parser artgen: error: cannot read {source_path}: {exc}",
              file=sys.stderr)
        return 1

    # Parse.
    try:
        floor = parse_fml(text, source_path=source_path)
    except FmlSyntaxError as exc:
        print(f"fml-parser artgen: syntax error: {exc}", file=sys.stderr)
        return 1
    except FmlImportError as exc:
        print(f"fml-parser artgen: import error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"fml-parser artgen: unexpected error during parse: {exc}",
              file=sys.stderr)
        return 1

    # Instantiate provider.
    try:
        provider = make_provider(args.provider, base_url=args.base_url)
    except ValueError as exc:
        print(f"fml-parser artgen: error: {exc}", file=sys.stderr)
        return 2

    # Generate manifest.
    try:
        manifest = generate_manifest(floor, provider)
    except NotImplementedError as exc:
        print(f"fml-parser artgen: provider error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"fml-parser artgen: error generating manifest: {exc}",
              file=sys.stderr)
        return 1

    # Serialise.
    manifest_str = emit_manifest_json(manifest)

    # Write output.
    out = args.output
    if out == "-":
        sys.stdout.write(manifest_str)
        sys.stdout.write("\n")
    else:
        out_path = Path(out)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(manifest_str + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"fml-parser artgen: error: cannot write {out_path}: {exc}",
                  file=sys.stderr)
            return 1

    return 0
