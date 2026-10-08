"""Stage 4a - Control Adapter: world state -> generator conditioning.

For every scene it produces the conditioning a video generator receives. Two arms are built from the same plan:

  orchestrated  enriched prompt (scene text + every visible identity with its locked appearance + setting),
                negative prompt (drift we want to suppress), a layout of boxes (positions persist per location,
                relations solved geometrically), fixed per-entity and per-location seeds, and continuity from the
                previous scene's last frame. This is what a layout/identity adapter (GLIGEN-, ControlNet-,
                IP-Adapter-style) would be fed in a diffusion model.
  baseline      what plain re-prompting gives a text-to-video model: only the scene's own sentence. Anything the
                sentence does not say (outfits, the bench from two scenes ago, where people stood) is left for the
                generator to make up, with a fresh random seed per clip.

Both arms go to the same generator, so any difference in the output comes from the conditioning alone.
"""
import random
import zlib

from .planner import HUMAN_KINDS, KNOWN_LOCATIONS, LOCATIONS
from .state_graph import CARRYABLE, ANIMAL_COLORS, CLOTH_COLORS, HAIR, ROBOT_COLORS, describe_entity

ASPECT = 16 / 9
GROUND = 0.9  # normalised y where feet touch the floor

# (height, width) in units of frame height
KIND_SIZE = {
    "adult": (0.50, 0.17), "child": (0.40, 0.15), "robot": (0.44, 0.20), "dog": (0.20, 0.30),
    "cat": (0.15, 0.22), "fox": (0.17, 0.28), "rabbit": (0.15, 0.14), "bear": (0.36, 0.46),
    "horse": (0.46, 0.56), "bench": (0.17, 0.44), "table": (0.21, 0.44), "chair": (0.27, 0.16),
    "sofa": (0.25, 0.56), "bed": (0.21, 0.60), "tree": (0.64, 0.32), "car": (0.25, 0.56), "lamp": (0.44, 0.13),
    "plant": (0.24, 0.14), "clock": (0.12, 0.12), "umbrella": (0.30, 0.30), "ball": (0.07, 0.07),
    "cup": (0.06, 0.05), "teapot": (0.09, 0.12), "book": (0.04, 0.10), "box": (0.10, 0.12), "kite": (0.12, 0.12),
    "bag": (0.11, 0.09), "guitar": (0.25, 0.09), "vase": (0.12, 0.07), "hat": (0.05, 0.10), "apple": (0.05, 0.05),
    "bowl": (0.04, 0.10), "bottle": (0.10, 0.04), "laptop": (0.07, 0.12), "phone": (0.05, 0.03),
    "toy": (0.08, 0.08), "cake": (0.08, 0.12), "suitcase": (0.15, 0.12), "balloon": (0.13, 0.08),
    "flower": (0.12, 0.06), "basket": (0.09, 0.12), "candle": (0.07, 0.03), "lantern": (0.10, 0.06),
    "map": (0.05, 0.10), "key": (0.03, 0.05), "pot": (0.08, 0.10), "bicycle": (0.21, 0.36),
}
ADULTS = {"woman", "man", "lady", "grandma", "grandpa", "mother", "mom", "father", "dad", "king", "chef",
          "farmer", "astronaut", "person", "princess", "student"}
SIZE_SCALE = {"large": 1.35, "small": 0.7, "tall": 1.12, "short": 0.86}
SEAT_HEIGHT = {"bench": 0.5, "sofa": 0.5, "chair": 0.5, "bed": 0.55}  # fraction of object height to sit on
BIG_PROPS = {"tree", "car", "sofa", "bed", "table", "bench", "lamp", "bicycle"}
MOTION_VERBS = {"walk", "run", "go", "jump", "dance", "wave"}


def _size(ent):
    kind = ent["kind"]
    if ent["type"] == "character" and kind in HUMAN_KINDS:
        h, w = KIND_SIZE["adult" if kind in ADULTS else "child"]
    else:
        h, w = KIND_SIZE.get(kind, (0.09, 0.10))
    s = SIZE_SCALE.get(ent["attributes"].get("size"), 1.0)
    return h * s, w * s / ASPECT


def _where(x):
    return "left" if x < 0.36 else "right" if x > 0.64 else "centre"


def _z(ent, rel):
    if rel and rel[0] == "behind":
        return 0
    if ent["kind"] in BIG_PROPS:
        return 1
    if ent["type"] == "character":
        return 3 if rel and rel[0] == "in_front_of" else 2
    return 4


class _Layout:
    """Places boxes for one scene. Everything is normalised: x = box centre, y_base = where it rests."""

    def __init__(self, memory, rng=None):
        self.memory = memory        # id -> x from earlier scenes at this location (None for the baseline)
        self.rng = rng
        self.boxes = {}

    def free_x(self, w):
        taken = [(b["x1"], b["w"]) for b in self.boxes.values() if b["held_by"] is None]
        cands = [0.1 + 0.8 * i / 16 for i in range(17)]
        if self.rng is not None:
            self.rng.shuffle(cands)
            return max(cands[:6], key=lambda c: min([abs(c - x) - (w + ow) / 2 for x, ow in taken] or [1]))
        return max(cands, key=lambda c: (min([abs(c - x) - (w + ow) / 2 for x, ow in taken] or [1]), -abs(c - 0.5)))

    def place(self, eid, ent, relation, holder=None):
        h, w = _size(ent)
        x, y_base, pose = None, GROUND, "stand"
        if holder is not None:
            hb = self.boxes[holder]
            x, y_base, pose = hb["x1"], None, "held"
        elif relation and relation[1] in self.boxes:
            rel, oid = relation
            ob = self.boxes[oid]
            if rel == "on":
                x = ob["x1"]
                frac = SEAT_HEIGHT.get(ob["kind"], 1.0)
                y_base = ob["y_base"] - ob["h"] * frac
                if ent["type"] == "character" and ob["kind"] in SEAT_HEIGHT:
                    pose = "sit"
                elif ent["type"] == "character":
                    pose = "stand"
                else:
                    # several things on one table: spread them out
                    n_on = sum(1 for b in self.boxes.values() if b.get("on") == oid)
                    x = ob["x1"] + (n_on % 3 - 1) * ob["w"] * 0.3
            elif rel == "under":
                x, pose = ob["x1"], "low"
            elif rel in ("next_to", "near", "left_of", "right_of"):
                gap = (ob["w"] + w) / 2 + (0.015 if rel == "next_to" else 0.06)
                side = {"left_of": -1, "right_of": 1}.get(rel)
                if side is None:
                    side = 1 if ob["x1"] < 0.5 else -1
                    if self.memory and eid in self.memory:
                        side = 1 if self.memory[eid] >= ob["x1"] else -1
                x = ob["x1"] + side * gap
            elif rel == "behind":
                x = ob["x1"] + ob["w"] * 0.25
            elif rel == "in_front_of":
                x = ob["x1"] - ob["w"] * 0.15
        if x is None and self.memory is not None and eid in self.memory:
            x = self.memory[eid]
        if x is None:
            x = self.free_x(w)
        x = min(max(x, w / 2 + 0.01), 1 - w / 2 - 0.01)
        if pose == "low":
            h *= 0.7
        self.boxes[eid] = {"id": eid, "name": ent["name"], "kind": ent["kind"], "type": ent["type"],
                           "attributes": dict(ent["attributes"]), "x0": x, "x1": x, "y_base": y_base,
                           "w": w, "h": h, "pose": pose, "motion": None, "held_by": holder,
                           "z": 5 if holder else _z(ent, relation),
                           "on": relation[1] if relation and relation[0] == "on" else None,
                           "rel": bool(relation and relation[1] in self.boxes)}
        return self.boxes[eid]

    def animate(self, scene, entering):
        """Turns the scene's verbs into motion; walkers end somewhere new, and that end point is remembered."""
        for a in scene["actions"]:
            b = self.boxes.get(a["actor"])
            if b is None or b["held_by"] or b["pose"] == "sit":
                continue
            verb = a["verb"]
            if verb in ("walk", "run"):
                dx = 0.16 if verb == "walk" else 0.28
                direction = -1 if b["x1"] > 0.55 else 1
                if b["rel"]:
                    # walking next to / up to something: arrive at the related spot instead of leaving it
                    b["x0"] = min(max(b["x1"] - direction * dx, b["w"] / 2), 1 - b["w"] / 2)
                else:
                    b["x1"] = min(max(b["x0"] + direction * dx, b["w"] / 2), 1 - b["w"] / 2)
                b["motion"] = verb
            elif verb in ("jump", "dance", "wave"):
                b["motion"] = verb
            elif verb == "sleep":
                b["pose"] = "sleep"
        for eid in entering:
            b = self.boxes.get(eid)
            if b and not b["held_by"] and b["pose"] == "stand":
                b["x0"] = -0.08 if b["x1"] < 0.5 else 1.08
                b["motion"] = b["motion"] or "walk"
        # whatever is carried moves with its holder
        for b in self.boxes.values():
            if b["held_by"] and b["held_by"] in self.boxes:
                hb = self.boxes[b["held_by"]]
                b["x0"], b["x1"], b["motion"] = hb["x0"], hb["x1"], hb["motion"]


def _ordered(ids, rel_of, held_of):
    """Supports before the things that rest on them, holders before what they hold."""
    done, out = set(), []

    def visit(eid, depth=0):
        if eid in done or depth > 8:
            return
        dep = held_of.get(eid) or (rel_of.get(eid) or (None, None))[1]
        if dep in ids:
            visit(dep, depth + 1)
        done.add(eid)
        out.append(eid)

    for eid in sorted(ids, key=lambda e: (held_of.get(e) is not None, rel_of.get(e) is not None)):
        visit(eid)
    return out


def orchestrated(scene, snap, world, n_scenes, prev_loc):
    env = snap["env"]
    loc = env["location"]
    memory = world.positions.setdefault(loc, {})
    rel_of = {r["subject"]: (r["relation"], r["object"]) for r in snap["relations"] if r["relation"] != "held_by"}
    held_of = {r["subject"]: r["object"] for r in snap["relations"] if r["relation"] == "held_by"}
    lay = _Layout(memory)
    for eid in _ordered(snap["visible"], rel_of, held_of):
        lay.place(eid, snap["entities"][eid], rel_of.get(eid), held_of.get(eid))
    entering = [eid for eid in snap["visible"] if eid not in memory and snap["entities"][eid]["type"] == "character"
                and prev_loc is not None]
    lay.animate(scene, entering)
    for eid, b in lay.boxes.items():
        if not b["held_by"]:
            memory[eid] = b["x1"]

    chars = [b for b in lay.boxes.values() if b["type"] == "character"]
    objs = [b for b in lay.boxes.values() if b["type"] == "object"]

    def line(b):
        ent = snap["entities"][b["id"]]
        where = f"held by {snap['entities'][b['held_by']]['name']}" if b["held_by"] else _where(b["x1"])
        return f"{describe_entity(ent)} at {where}"

    setting = loc + ("" if env["indoors"] or env["weather"] == "clear" else f", {env['weather']}") + \
        ("" if env["time_of_day"] == "day" else f", {env['time_of_day']}")
    prompt = (f"Scene {scene['index']}/{n_scenes}. Setting: {setting}. {scene['text']} "
              f"Characters: {'; '.join(line(b) for b in chars) or 'none'}. "
              f"Objects: {'; '.join(line(b) for b in objs) or 'none'}. "
              + (f"Continue from the last frame of scene {scene['index'] - 1}; keep every identity, outfit, "
                 f"colour and the layout of the {loc} unchanged." if scene["index"] > 1 else
                 "Establishing shot; this look is the reference for every later scene."))
    negative = ", ".join(["different outfit", "changed hair colour", "different face", "extra people"] +
                         [f"missing {b['name']}" for b in objs if not b["held_by"]][:6])
    return {
        "arm": "orchestrated", "scene": scene["index"], "prompt": prompt, "negative_prompt": negative,
        "background": {"location": loc, "weather": env["weather"], "time_of_day": env["time_of_day"],
                       "seed": zlib.crc32(loc.encode())},
        "entities": sorted(lay.boxes.values(), key=lambda b: b["z"]),
        "continue_from_previous": prev_loc == loc,
    }


def _literal(text, value):
    return value and value.lower() in text.lower()


def baseline(scene, plan, world_entities, rng_seed):
    """Text-only conditioning: what a T2V model gets when each scene is prompted on its own."""
    rng = random.Random(rng_seed)
    text = scene["text"]
    loc = scene["location"] if scene["location"] and any(
        _literal(text, k) for k, v in LOCATIONS.items() if v == scene["location"]) else rng.choice(KNOWN_LOCATIONS)
    weather = scene["weather"] if scene["weather"] and scene["weather"] != "clear" else "clear"
    tod = scene["time_of_day"] or "day"

    ents = {}
    for eid in scene["present"]:
        src = world_entities[eid]
        attrs = {}
        for k, v in src["attributes"].items():
            if k in ("size", "clothing") or _literal(text, v):
                attrs[k] = v
                continue
            palette = {"hair_color": HAIR, "clothing_color": CLOTH_COLORS}.get(k)
            if palette is None and k == "color":
                palette = ANIMAL_COLORS if src["kind"] not in HUMAN_KINDS and src["type"] == "character" else \
                    ROBOT_COLORS if src["kind"] == "robot" else CLOTH_COLORS + ["brown"]
            attrs[k] = rng.choice(palette) if palette else None
        if not _literal(text, src["attributes"].get("clothing", "")) and "clothing" in attrs:
            attrs["clothing"] = rng.choice(["jacket", "shirt", "sweater", "dress"] if src["kind"] in
                                           ("girl", "woman", "lady") else ["jacket", "shirt", "sweater"])
        attrs = {k: v for k, v in attrs.items() if v is not None and k != "state"}
        ents[eid] = {"name": src["name"], "type": src["type"], "kind": src["kind"], "attributes": attrs}

    rel_of = {r["subject"]: (r["relation"], r["object"]) for r in scene["relations"]
              if r["subject"] in ents and r["object"] in ents}
    held_of = {}
    for a in scene["actions"]:
        if a["verb"] in ("pick_up", "hold") and a["target"] in ents and (
                ents[a["target"]]["type"] == "object" or ents[a["target"]]["kind"] in CARRYABLE):
            held_of[a["target"]] = a["actor"]
            rel_of.pop(a["target"], None)
        if a["verb"] == "sit" and a["target"] in ents and a["actor"] not in rel_of:
            rel_of[a["actor"]] = ("on", a["target"])
        if a["verb"] in ("open", "close") and a["target"] in ents:
            ents[a["target"]]["attributes"]["state"] = "open" if a["verb"] == "open" else "closed"
    lay = _Layout(None, rng)
    for eid in _ordered(list(ents), rel_of, held_of):
        lay.place(eid, ents[eid], rel_of.get(eid), held_of.get(eid))
    lay.animate(scene, [])
    return {
        "arm": "baseline", "scene": scene["index"], "prompt": text, "negative_prompt": "",
        "background": {"location": loc, "weather": weather, "time_of_day": tod, "seed": rng.randrange(1 << 30)},
        "entities": sorted(lay.boxes.values(), key=lambda b: b["z"]),
        "continue_from_previous": False,
    }
