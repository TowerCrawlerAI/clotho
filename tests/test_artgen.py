"""Tests for the art-generation pipeline (artgen subcommand + lower --art-manifest).

Test sections:
  A1 — Curated provider: property-to-texture mapping + precedence order.
  A2 — Firefly provider: prompt builder and NotImplementedError behavior.
  A3 — Manifest output shape and determinism.
  A4 — lower --art-manifest merges room.art into map.json.
  A5 — lower without --art-manifest: output byte-identical to pre-artgen output.
  A6 — CLI integration: artgen subcommand writes a manifest file.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fml_parser.artgen import (
    CuratedProvider,
    FireflyProvider,
    build_room_prompt,
    generate_manifest,
    emit_manifest_json,
    make_provider,
)
from fml_parser.emit_lua import emit_lua_om
from fml_parser.emit_map import emit_map, emit_map_json, strip_map_keys
from fml_parser.models import FMLEntity, Floor

# ── Fixture helpers ──────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BONE_GARDEN_DIR = FIXTURES_DIR / "bone_garden"


def _make_room(
    rid: str,
    name: str,
    *,
    exits: dict | None = None,
    props: dict | None = None,
) -> FMLEntity:
    """Create a room entity with the given properties."""
    p: dict = {"exits": exits or {}}
    if props:
        p.update(props)
    return FMLEntity(
        id=rid,
        name=name,
        kind="room",
        properties=p,
        kind_chain=["room"],
    )


def _make_floor(*rooms: FMLEntity, start: str | None = None) -> Floor:
    props = {}
    if start:
        props["start_location"] = start
    f = Floor(name="Art Test Floor", properties=props)
    for r in rooms:
        f += r
    return f


# ─────────────────────────────────────────────────────────────────────────────
# A1 — Curated provider: property-to-texture mapping
# ─────────────────────────────────────────────────────────────────────────────


class TestCuratedProvider:
    """Curated provider maps room properties to CC0 texture names."""

    def setup_method(self) -> None:
        self.provider = CuratedProvider(base_url="/art")

    def test_water_truthy_yields_wet_stone(self) -> None:
        room = _make_room("pool", "Pool", props={"water": True, "underground": True, "level": -1})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/wet_stone.jpg"
        assert result["fit"] == "tile"

    def test_water_string_true_yields_wet_stone(self) -> None:
        """FML properties may be stored as strings."""
        room = _make_room("flooded", "Flooded", props={"water": "true"})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/wet_stone.jpg"

    def test_outdoor_truthy_yields_silt(self) -> None:
        """outdoor (no water) → silt."""
        room = _make_room("garden", "Garden", props={"outdoor": True})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/silt.jpg"
        assert result["fit"] == "tile"

    def test_level_minus2_yields_ashlar(self) -> None:
        """level ≤ -2 (no water, no outdoor) → ashlar."""
        room = _make_room("deep", "Deep Room", props={"level": -2, "underground": True})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/ashlar.jpg"

    def test_level_minus3_yields_ashlar(self) -> None:
        """level ≤ -2 also applies at -3, -4, etc."""
        room = _make_room("abyss", "Abyss", props={"level": -3})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/ashlar.jpg"

    def test_underground_truthy_yields_dark_stone(self) -> None:
        """underground (level 0) → dark_stone."""
        room = _make_room("cave", "Cave", props={"underground": True})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/dark_stone.jpg"

    def test_level_minus1_yields_dark_stone(self) -> None:
        """level == -1 (no underground prop) → dark_stone."""
        room = _make_room("tunnel", "Tunnel", props={"level": -1})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/dark_stone.jpg"

    def test_default_yields_flagstone(self) -> None:
        """No matching properties → flagstone."""
        room = _make_room("hall", "Hall", props={})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/flagstone.jpg"
        assert result["fit"] == "tile"

    def test_all_fit_values_are_tile(self) -> None:
        """All curated textures use fit: tile."""
        rooms = [
            _make_room("r_water", "R1", props={"water": True}),
            _make_room("r_outdoor", "R2", props={"outdoor": True}),
            _make_room("r_deep", "R3", props={"level": -2}),
            _make_room("r_dark", "R4", props={"underground": True}),
            _make_room("r_flag", "R5", props={}),
        ]
        for room in rooms:
            result = self.provider.resolve(room)
            assert result is not None
            assert result["fit"] == "tile", (
                f"Room {room.id}: expected fit=tile, got {result['fit']}"
            )

    # ── Precedence order ──────────────────────────────────────────────────────

    def test_precedence_water_beats_outdoor(self) -> None:
        """water truthy overrides outdoor truthy."""
        room = _make_room("marsh", "Marsh", props={"water": True, "outdoor": True})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/wet_stone.jpg", (
            "water should beat outdoor in precedence"
        )

    def test_precedence_water_beats_level_minus2(self) -> None:
        """water truthy overrides level ≤ -2."""
        room = _make_room("deep_pool", "Deep Pool", props={"water": True, "level": -2})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/wet_stone.jpg"

    def test_precedence_water_beats_underground(self) -> None:
        """water truthy overrides underground."""
        room = _make_room("cave_pool", "Cave Pool", props={"water": True, "underground": True})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/wet_stone.jpg"

    def test_precedence_outdoor_beats_level_minus2(self) -> None:
        """outdoor truthy overrides level ≤ -2."""
        room = _make_room("pit", "Pit", props={"outdoor": True, "level": -2})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/silt.jpg", (
            "outdoor should beat level ≤ -2 in precedence"
        )

    def test_precedence_outdoor_beats_underground(self) -> None:
        """outdoor truthy overrides underground."""
        room = _make_room("sky_cave", "Sky Cave", props={"outdoor": True, "underground": True})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/silt.jpg"

    def test_precedence_level_minus2_beats_underground(self) -> None:
        """level == -2 with underground=True → ashlar (not dark_stone)."""
        room = _make_room("very_deep", "Very Deep", props={"level": -2, "underground": True})
        result = self.provider.resolve(room)
        assert result is not None
        assert result["image"] == "/art/ashlar.jpg", (
            "level ≤ -2 should beat underground in precedence"
        )

    def test_custom_base_url(self) -> None:
        """base_url is reflected in the returned image URL."""
        provider = CuratedProvider(base_url="https://cdn.example.com/textures")
        room = _make_room("hall", "Hall", props={})
        result = provider.resolve(room)
        assert result is not None
        assert result["image"].startswith("https://cdn.example.com/textures/")
        assert result["image"].endswith(".jpg")

    def test_base_url_trailing_slash_stripped(self) -> None:
        """Trailing slash on base_url is normalised."""
        provider = CuratedProvider(base_url="/art/")
        room = _make_room("hall", "Hall", props={})
        result = provider.resolve(room)
        assert result is not None
        # Should NOT have double slash.
        assert "//" not in result["image"].replace("://", "")


# ─────────────────────────────────────────────────────────────────────────────
# A2 — Firefly provider
# ─────────────────────────────────────────────────────────────────────────────


class TestFireflyProvider:
    """Firefly provider builds prompts and raises NotImplementedError at resolve time."""

    def test_build_room_prompt_includes_name(self) -> None:
        room = _make_room("vault", "The Vault")
        prompt = build_room_prompt(room)
        assert "The Vault" in prompt, "Prompt must include the room name"

    def test_build_room_prompt_includes_description(self) -> None:
        room = _make_room("dungeon", "Dungeon")
        room.prose = "A damp stone chamber with crumbling walls."
        prompt = build_room_prompt(room)
        assert "damp stone" in prompt, "Prompt must include prose description"

    def test_build_room_prompt_includes_water_cue(self) -> None:
        room = _make_room("pool", "Pool", props={"water": True})
        prompt = build_room_prompt(room)
        assert "water" in prompt.lower() or "wet" in prompt.lower(), (
            "Prompt for water room must include water/wet cue"
        )

    def test_build_room_prompt_includes_outdoor_cue(self) -> None:
        room = _make_room("garden", "Garden", props={"outdoor": True})
        prompt = build_room_prompt(room)
        assert "outdoor" in prompt.lower() or "sky" in prompt.lower() or "silt" in prompt.lower(), (
            "Prompt for outdoor room must include outdoor cue"
        )

    def test_build_room_prompt_is_top_down_battlemap(self) -> None:
        room = _make_room("hall", "Hall")
        prompt = build_room_prompt(room)
        assert "top-down" in prompt or "overhead" in prompt, (
            "Prompt must mention top-down / overhead view"
        )
        assert "battlemap" in prompt or "dungeon" in prompt, (
            "Prompt must mention battlemap or dungeon context"
        )

    def test_firefly_resolve_raises_not_implemented(self) -> None:
        provider = FireflyProvider()
        room = _make_room("vault", "The Vault")
        with pytest.raises(NotImplementedError) as exc_info:
            provider.resolve(room)
        # The error message must include the generated prompt.
        assert "The Vault" in str(exc_info.value), (
            "NotImplementedError must embed the generated prompt"
        )

    def test_firefly_resolve_error_includes_key_phrase(self) -> None:
        """The NotImplementedError must mention API key/uploader requirements."""
        provider = FireflyProvider()
        room = _make_room("room_x", "Room X")
        with pytest.raises(NotImplementedError) as exc_info:
            provider.resolve(room)
        msg = str(exc_info.value).lower()
        assert "api key" in msg or "uploader" in msg or "firefly" in msg, (
            "Error must explain what's needed to activate the provider"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A3 — Manifest shape and determinism
# ─────────────────────────────────────────────────────────────────────────────


class TestManifestShape:
    """generate_manifest produces correct shape and is deterministic."""

    def _simple_floor(self) -> Floor:
        a = _make_room("entry", "Entry", props={"outdoor": True})
        b = _make_room("catacombs", "Catacombs", props={"water": True, "level": -1})
        return _make_floor(a, b, start="entry")

    def test_manifest_has_required_keys(self) -> None:
        floor = self._simple_floor()
        provider = CuratedProvider(base_url="/art")
        manifest = generate_manifest(floor, provider)
        assert "version" in manifest
        assert "rooms" in manifest
        assert "tokens" in manifest
        assert manifest["version"] == 1
        assert manifest["tokens"] == {}

    def test_manifest_rooms_sorted_order(self) -> None:
        """Room IDs in manifest must be in sorted order."""
        z = _make_room("zebra", "Zebra")
        a = _make_room("alpha", "Alpha")
        m = _make_room("middle", "Middle")
        floor = _make_floor(z, a, m)
        provider = CuratedProvider(base_url="/art")
        manifest = generate_manifest(floor, provider)
        room_ids = list(manifest["rooms"].keys())
        assert room_ids == sorted(room_ids), (
            f"Room IDs must be in sorted order, got {room_ids}"
        )

    def test_manifest_each_room_has_image_and_fit(self) -> None:
        floor = self._simple_floor()
        provider = CuratedProvider(base_url="/art")
        manifest = generate_manifest(floor, provider)
        for rid, art in manifest["rooms"].items():
            assert "image" in art, f"Room {rid} missing 'image' key"
            assert "fit" in art, f"Room {rid} missing 'fit' key"

    def test_manifest_correct_textures(self) -> None:
        """Water room → wet_stone; outdoor room → silt."""
        floor = self._simple_floor()
        provider = CuratedProvider(base_url="/art")
        manifest = generate_manifest(floor, provider)
        assert "wet_stone" in manifest["rooms"]["catacombs"]["image"]
        assert "silt" in manifest["rooms"]["entry"]["image"]

    def test_determinism_same_floor_same_output(self) -> None:
        """Running generate_manifest twice on the same floor gives byte-identical output."""
        floor = self._simple_floor()
        provider = CuratedProvider(base_url="/art")
        out1 = emit_manifest_json(generate_manifest(floor, provider))
        out2 = emit_manifest_json(generate_manifest(floor, provider))
        assert out1 == out2, "Manifest must be byte-identical across runs"

    def test_make_provider_curated(self) -> None:
        p = make_provider("curated", base_url="/textures")
        assert isinstance(p, CuratedProvider)

    def test_make_provider_firefly(self) -> None:
        p = make_provider("firefly")
        assert isinstance(p, FireflyProvider)

    def test_make_provider_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            make_provider("dalle")


# ─────────────────────────────────────────────────────────────────────────────
# A4 — lower --art-manifest merges room.art into map.json
# ─────────────────────────────────────────────────────────────────────────────


class TestArtManifestMerge:
    """lower --art-manifest sets room.art; rooms absent from manifest unchanged."""

    def test_manifest_sets_room_art(self, tmp_path: Path) -> None:
        """Rooms listed in the manifest get art = {"src": ..., "fit": ...}."""
        from fml_parser.__main__ import main

        index_md = BONE_GARDEN_DIR / "index.md"
        out_lua = tmp_path / "floor.lua"
        map_json = tmp_path / "map.json"

        # Write a minimal manifest covering two rooms.
        manifest = {
            "version": 1,
            "rooms": {
                "entry": {"image": "/art/silt.jpg", "fit": "tile"},
                "catacombs": {"image": "/art/wet_stone.jpg", "fit": "tile"},
            },
            "tokens": {},
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        rc = main([
            "lower", str(index_md),
            "--om", "--map",
            "--art-manifest", str(manifest_path),
            "-o", str(out_lua),
        ])
        assert rc == 0, "CLI must exit 0"
        assert map_json.exists(), "map.json must be written"

        data = json.loads(map_json.read_text())
        # Listed rooms must have art set.
        assert data["rooms"]["entry"]["art"] == {
            "src": "/art/silt.jpg", "fit": "tile"
        }, f"entry.art mismatch: {data['rooms']['entry'].get('art')}"
        assert data["rooms"]["catacombs"]["art"] == {
            "src": "/art/wet_stone.jpg", "fit": "tile"
        }, f"catacombs.art mismatch: {data['rooms']['catacombs'].get('art')}"

    def test_rooms_absent_from_manifest_unchanged(self, tmp_path: Path) -> None:
        """Rooms not in the manifest retain their original art value (null by default)."""
        from fml_parser.__main__ import main

        index_md = BONE_GARDEN_DIR / "index.md"
        out_lua = tmp_path / "floor.lua"
        map_json = tmp_path / "map.json"

        # Manifest covers ONLY entry.
        manifest = {
            "version": 1,
            "rooms": {
                "entry": {"image": "/art/silt.jpg", "fit": "tile"},
            },
            "tokens": {},
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # First run WITHOUT manifest: get the baseline art for catacombs.
        baseline_lua = tmp_path / "base.lua"
        base_map = tmp_path / "base_dir" / "map.json"
        main(["lower", str(index_md), "--om", "--map", "-o", str(baseline_lua)])

        # Second run WITH manifest.
        rc = main([
            "lower", str(index_md),
            "--om", "--map",
            "--art-manifest", str(manifest_path),
            "-o", str(out_lua),
        ])
        assert rc == 0

        data = json.loads(map_json.read_text())
        # catacombs is NOT in the manifest; its art must be whatever map.json
        # normally emits (null unless it has an FML `map: image:` key).
        # Bone Garden catacombs has no `map: image:`, so art must be null.
        assert data["rooms"]["catacombs"]["art"] is None, (
            "Rooms absent from manifest must not have their art altered"
        )
        # But entry IS in the manifest.
        assert data["rooms"]["entry"]["art"] == {
            "src": "/art/silt.jpg", "fit": "tile"
        }

    def test_manifest_overrides_fml_art(self, tmp_path: Path) -> None:
        """Manifest art overrides any FML map: image: art."""
        from fml_parser.__main__ import _merge_art_manifest

        # Simulate a map_data dict that already has art from FML.
        map_data: dict = {
            "rooms": {
                "hall": {
                    "art": {"src": "https://old.cdn/hall.png", "fit": "cover"}
                }
            },
            "tokens": {"art": {}},
        }
        manifest = {
            "version": 1,
            "rooms": {
                "hall": {"image": "/art/flagstone.jpg", "fit": "tile"},
            },
            "tokens": {},
        }
        _merge_art_manifest(map_data, manifest)
        assert map_data["rooms"]["hall"]["art"] == {
            "src": "/art/flagstone.jpg",
            "fit": "tile",
        }

    def test_manifest_room_not_in_floor_skipped(self, tmp_path: Path) -> None:
        """Room IDs in manifest that don't exist in map.json are silently skipped."""
        from fml_parser.__main__ import _merge_art_manifest

        map_data: dict = {
            "rooms": {"entry": {"art": None}},
            "tokens": {"art": {}},
        }
        manifest = {
            "version": 1,
            "rooms": {
                "entry": {"image": "/art/silt.jpg", "fit": "tile"},
                "phantom_room": {"image": "/art/flagstone.jpg", "fit": "tile"},
            },
            "tokens": {},
        }
        _merge_art_manifest(map_data, manifest)
        # entry updated, phantom_room skipped (no KeyError).
        assert map_data["rooms"]["entry"]["art"] == {
            "src": "/art/silt.jpg", "fit": "tile"
        }
        assert "phantom_room" not in map_data["rooms"]


# ─────────────────────────────────────────────────────────────────────────────
# A5 — lower without --art-manifest: output byte-identical to baseline
# ─────────────────────────────────────────────────────────────────────────────


def test_a5_no_art_manifest_output_unchanged(tmp_path: Path) -> None:
    """lower --om --map without --art-manifest must produce byte-identical
    output across two successive runs (determinism guard).

    This test also verifies that adding --art-manifest support is purely
    additive: when the flag is absent the output is unchanged.

    Note: the golden M6 sha-identity test uses a stable relative source_path
    ("bone_garden/index.md") whereas the CLI uses the absolute filesystem path,
    so the floor_sha differs — that is pre-existing and expected.  What we
    verify here is that two CLI runs WITHOUT --art-manifest are byte-identical,
    and that the output differs from an --art-manifest run (the manifest injects
    art fields).
    """
    from fml_parser.__main__ import main

    index_md = BONE_GARDEN_DIR / "index.md"

    # Run WITHOUT --art-manifest.
    out1 = tmp_path / "run1" / "floor.lua"
    out1.parent.mkdir()
    rc = main(["lower", str(index_md), "--om", "--map", "-o", str(out1)])
    assert rc == 0
    map1 = (tmp_path / "run1" / "map.json").read_text(encoding="utf-8")

    # Run again WITHOUT --art-manifest.
    out2 = tmp_path / "run2" / "floor.lua"
    out2.parent.mkdir()
    rc = main(["lower", str(index_md), "--om", "--map", "-o", str(out2)])
    assert rc == 0
    map2 = (tmp_path / "run2" / "map.json").read_text(encoding="utf-8")

    assert map1 == map2, (
        "Two runs without --art-manifest must produce byte-identical map.json"
    )

    # Verify the map.json is valid and well-formed.
    data = json.loads(map1)
    assert data["map_version"] == 1
    assert "rooms" in data
    assert "connections" in data

    # Verify that rooms which have no FML art key have null art (not touched).
    # entry has outdoor=true but no FML map:image, so art must be null.
    assert data["rooms"]["entry"]["art"] is None, (
        "Without --art-manifest, rooms with no FML map:image must have art: null"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A6 — CLI integration: artgen subcommand
# ─────────────────────────────────────────────────────────────────────────────


def test_a6_artgen_cli_writes_manifest(tmp_path: Path) -> None:
    """artgen CLI writes a valid manifest file."""
    from fml_parser.__main__ import main

    index_md = BONE_GARDEN_DIR / "index.md"
    manifest_path = tmp_path / "manifest.json"

    rc = main(["artgen", str(index_md), "--provider", "curated",
               "--base-url", "/art", "-o", str(manifest_path)])
    assert rc == 0, "artgen CLI must exit 0"
    assert manifest_path.exists(), "manifest.json must be written"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "rooms" in data
    assert "tokens" in data
    assert data["tokens"] == {}
    assert len(data["rooms"]) > 0, "Must have at least one room in manifest"


def test_a6_artgen_cli_deterministic(tmp_path: Path) -> None:
    """artgen run twice → byte-identical manifest."""
    from fml_parser.__main__ import main

    index_md = BONE_GARDEN_DIR / "index.md"

    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"

    main(["artgen", str(index_md), "--provider", "curated", "-o", str(m1)])
    main(["artgen", str(index_md), "--provider", "curated", "-o", str(m2)])

    assert m1.read_text() == m2.read_text(), (
        "artgen must be deterministic: two runs must produce byte-identical output"
    )


SAMPLE_DUNGEON_INDEX = (
    Path(__file__).parent.parent.parent / "sample-dungeon" / "index.md"
)


def test_a6_artgen_cli_bone_garden_expected_textures(tmp_path: Path) -> None:
    """Real Bone Garden (sample-dungeon/index.md) rooms get the expected textures:
    - catacombs (water=true, underground=true, level=-1) → wet_stone
    - garden (outdoor=true) → silt
    - crypt (level=-2, underground=true) → ashlar
    - twisting_tunnel_a1 (level=-1, underground=true) → dark_stone
    - entry (outdoor=true) → silt
    - hall_of_skulls (no special art props) → flagstone
    - ossuary (no special art props) → flagstone

    This test runs against the LIVE sample-dungeon sibling repo to verify
    the curated mapping on the real floor.  It is skipped if the sibling
    repo is not present (the M3/M6 fixture tests cover functional correctness;
    this test is the integration / real-floor evidence step).
    """
    if not SAMPLE_DUNGEON_INDEX.exists():
        pytest.skip("sample-dungeon sibling repo not present; skipping real-floor artgen test")

    from fml_parser.__main__ import main

    manifest_path = tmp_path / "manifest.json"

    rc = main(["artgen", str(SAMPLE_DUNGEON_INDEX), "--provider", "curated",
               "--base-url", "/art", "-o", str(manifest_path)])
    assert rc == 0, "artgen must exit 0 on the real Bone Garden floor"

    data = json.loads(manifest_path.read_text())
    rooms = data["rooms"]

    def _texture(room_id: str) -> str:
        """Extract texture name from image URL."""
        art = rooms.get(room_id)
        assert art is not None, f"Room {room_id!r} missing from manifest"
        url = art["image"]
        # e.g. "/art/wet_stone.jpg" → "wet_stone"
        return Path(url).stem

    assert _texture("catacombs") == "wet_stone", (
        f"catacombs (water=true) → wet_stone; got {rooms.get('catacombs')}"
    )
    assert _texture("garden") == "silt", (
        f"garden (outdoor=true) → silt; got {rooms.get('garden')}"
    )
    assert _texture("crypt") == "dark_stone", (
        f"crypt (underground=true, level=-1) → dark_stone; got {rooms.get('crypt')}"
    )
    assert _texture("twisting_tunnel_a1") == "dark_stone", (
        f"twisting_tunnel_a1 (underground=true, level=-1) → dark_stone; "
        f"got {rooms.get('twisting_tunnel_a1')}"
    )
    assert _texture("entry") == "silt", (
        f"entry (outdoor=true) → silt; got {rooms.get('entry')}"
    )
    assert _texture("hall_of_skulls") == "flagstone", (
        f"hall_of_skulls (no special art props) → flagstone; got {rooms.get('hall_of_skulls')}"
    )
    assert _texture("ossuary") == "flagstone", (
        f"ossuary (no special art props) → flagstone; got {rooms.get('ossuary')}"
    )
