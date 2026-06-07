"""Tests for the P6 object-model emitter (emit_lua_om).

emit_lua_om produces the prototype/fragment LFR for the crawler `--engine om`
dispatch path. It is a thin wrapper over emit_lua_graph(om=True): same DB-init
shape, but each world entity gets a prototype EDGE from its kind (resolved by
name from the `kind:<name>` registry on `object`), the string `kind` property is
dropped, verbs are grammar-only, and behaviour triggers are skipped (the
behaviour port is a later phase). These tests verify that contract — and that
om=False (graph mode) output is unchanged.
"""

from __future__ import annotations

from fml_parser.emit_lua import emit_lua_graph, emit_lua_om
from fml_parser.models import FMLEntity, Floor, Script, Trigger


def make_floor(entities: list[FMLEntity], **floor_props) -> Floor:
    f = Floor(name="Test Floor", properties=floor_props)
    for ent in entities:
        f += ent
    return f


def make_entity(eid, name, kind, properties=None, triggers=None, kind_chain=None):
    return FMLEntity(
        id=eid, name=name, kind=kind,
        properties=properties or {}, triggers=triggers or [],
        kind_chain=kind_chain or [kind],
    )


def _tiny_floor() -> Floor:
    cell = make_entity("cell", "Cell", "room",
                       properties={"exits": {"north": "hall"}})
    hall = make_entity("hall", "Hall", "room",
                       properties={"exits": {"south": "cell"}})
    key = make_entity("key", "Key", "item",
                      properties={"location": "cell"})
    grate = make_entity("grate", "Grate", "scenery",
                        properties={"location": "cell"})
    return make_floor([cell, hall, key, grate], start_location="cell")


# ─── prototype edges ─────────────────────────────────────────────────────────


def test_emits_proto_helper_and_edges():
    out = emit_lua_om(_tiny_floor(), source_path="floor.md")
    # the _proto helper is defined once, reading the kind:<name> registry
    assert "local function _proto(n, kind)" in out
    assert 'engine.get_prop(om.object(), "kind:" .. kind)' in out
    assert "om.set_prototype(n, k)" in out
    # each world entity gets a prototype edge from its kind
    assert '_proto(n_key, "item")' in out
    assert '_proto(n_grate, "scenery")' in out
    assert '_proto(n_cell, "room")' in out


def test_drops_string_kind_property():
    # an entity that declared `kind` inline must NOT emit it as a node property
    # (the prototype edge replaces it).
    item = make_entity("sigil", "Sigil", "item",
                       properties={"location": "cell", "kind": "item", "type": "knowledge"})
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, item], start_location="cell"))
    assert "kind =" not in out          # no string kind prop anywhere
    assert '_proto(n_sigil, "item")' in out
    assert 'type = "knowledge"' in out  # other props survive


def test_synthesized_player_gets_actor_prototype():
    out = emit_lua_om(_tiny_floor())
    assert 'local n__player = engine.create_node({ name = "you" })' in out
    assert '_proto(n__player, "actor")' in out
    assert "engine.set_start_actor(n__player)" in out


# ─── verbs are grammar-only; behaviours skipped ──────────────────────────────


def test_verbs_grammar_only_no_stages():
    verb = make_entity(
        "drink", "drink", "verb",
        properties={"noun": "required", "scope": "touch"},
        triggers=[Trigger(name="On", script=Script(language="lua",
                                                   source="engine.output('glug')"))],
    )
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, verb], start_location="cell"))
    assert "engine.define_verb({" in out
    assert 'name = "drink"' in out
    assert 'noun = "required"' in out
    # the On stage closure must NOT be emitted (grammar-only)
    assert "stages = {" not in out
    assert "glug" not in out


def test_entity_triggers_lower_to_om_on():
    # a lua/luau instance trigger lowers to om.on(node, "On<Event>", fn) — NOT
    # engine.set_trigger (that's the graph path) — with the body preserved.
    npc = make_entity(
        "oracle", "Oracle", "npc",
        properties={"location": "cell"},
        triggers=[Trigger(name="On Answer", script=Script(language="lua",
                                                          source="engine.output('hmm')"))],
    )
    cell = make_entity("cell", "Cell", "room")
    out = emit_lua_om(make_floor([cell, npc], start_location="cell"))
    assert "engine.set_trigger" not in out          # not the graph path
    assert 'om.on(n_oracle, "OnAnswer", function(ctx)' in out
    assert "engine.output('hmm')" in out            # body preserved
    assert '_proto(n_oracle, "npc")' in out          # still placed structurally


def test_om_event_name_mapping():
    # the <stage> <Event> heading collapses to On<Event> for the om event bus.
    from fml_parser.emit_lua import _om_event_name
    assert _om_event_name("On Open") == "OnOpen"
    assert _om_event_name("After Enter") == "OnEnter"
    assert _om_event_name("On RoundStart") == "OnRoundStart"
    assert _om_event_name("InsteadOf Damaged") == "OnDamaged"
    assert _om_event_name("On Answer") == "OnAnswer"


def test_non_lua_trigger_body_warned_not_emitted():
    # an FML action-vocabulary body (no lua/luau script) is not transpiled (same
    # as graph mode) — it warns rather than emitting a broken om.on.
    room = make_entity(
        "hall", "Hall", "room",
        triggers=[Trigger(name="After Enter",
                          script=Script(language="python", source="set_flag('x')"))],
    )
    out = emit_lua_om(make_floor([room], start_location="hall"))
    assert "om.on(n_hall" not in out
    assert "WARNING" in out


# ─── containment + exits preserved ───────────────────────────────────────────


def test_containment_and_exits_preserved():
    out = emit_lua_om(_tiny_floor())
    assert 'engine.relate("in", n_key, n_cell)' in out
    assert 'engine.set_prop(n_cell, "exit_north", n_hall)' in out
    assert 'engine.set_prop(n_hall, "exit_south", n_cell)' in out


# ─── determinism + graph-mode non-regression ─────────────────────────────────


def test_deterministic():
    f = _tiny_floor()
    assert emit_lua_om(f) == emit_lua_om(f)


def test_graph_mode_unchanged_by_om_param():
    # om=False must be byte-identical to the legacy graph emitter default.
    f = _tiny_floor()
    assert emit_lua_graph(f, source_path="x.md") == emit_lua_graph(
        f, source_path="x.md", om=False
    )
    # and the om output must differ (proves the branch is active)
    assert emit_lua_om(f, source_path="x.md") != emit_lua_graph(f, source_path="x.md")
