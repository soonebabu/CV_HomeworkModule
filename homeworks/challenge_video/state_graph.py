"""Stage 3 - Persistent World / State Graph.

Holds what is true about the world between scenes, so nothing has to be re-prompted:
  characters  - identity + visual appearance (filled in once, then locked)
  objects     - identity, attributes (colour, open/closed) and where they are (a place, on something, held by someone)
  context     - current location, weather, time of day, spatial relations and the actions going on

apply(scene) moves the world forward one scene and returns a snapshot: the full expected state for that scene.
The snapshots double as the "narrative ground truth" the evaluation checks both videos against.
"""
import copy
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .planner import ANIMAL_KINDS, HUMAN_KINDS

INDOORS = {"kitchen", "room", "office", "cafe"}
HANDHELD = {"umbrella", "book", "bag", "box", "laptop", "map", "phone", "cup", "bottle", "kite", "balloon",
            "lantern", "basket", "toy", "apple", "key", "suitcase", "guitar"}
CARRYABLE = {"cat", "kitten", "rabbit", "puppy", "fox"}  # small animals a person can pick up
MIRROR = {"next_to": "next_to", "near": "near", "left_of": "right_of", "right_of": "left_of"}
HAIR = ["black", "brown", "yellow", "red", "gray"]
CLOTH_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "teal"]
ANIMAL_COLORS = ["brown", "white", "black", "orange", "gray"]
ROBOT_COLORS = ["gray", "white", "blue", "teal"]
OBJECT_DEFAULTS = {
    "bench": ["brown", "green"], "table": ["brown"], "chair": ["brown", "red"], "tree": ["green"],
    "sofa": ["purple", "blue", "red"], "bed": ["blue", "white"], "plant": ["green"], "flower": ["pink", "red"],
    "apple": ["red", "green"], "lamp": ["yellow"], "book": ["red", "blue", "green"], "car": ["red", "blue", "white"],
    "teapot": ["white", "blue"], "cup": ["white", "blue"], "clock": ["white"], "basket": ["brown"],
}


def stable_choice(key, options):
    """Same key -> same pick, across runs and processes (unlike hash())."""
    return options[zlib.crc32(key.encode("utf-8")) % len(options)]


def default_attributes(ent):
    """Appearance the story didn't specify. Picked once from the entity id and then locked."""
    kind, eid = ent["kind"], ent["id"]
    if ent["type"] == "character" and kind in HUMAN_KINDS:
        female = kind in ("girl", "woman", "lady", "grandma", "mother", "mom", "princess")
        return {"hair_color": stable_choice(eid + "/hair", HAIR if kind not in ("grandma", "grandpa") else ["gray"]),
                "clothing": stable_choice(eid + "/cl", ["dress", "jacket", "sweater"] if female
                                          else ["jacket", "shirt", "sweater"]),
                "clothing_color": stable_choice(eid + "/cc", CLOTH_COLORS),
                "size": "medium"}
    if ent["type"] == "character" and kind in ANIMAL_KINDS:
        return {"color": stable_choice(eid + "/c", ANIMAL_COLORS), "size": "medium"}
    if ent["type"] == "character":
        return {"color": stable_choice(eid + "/c", ROBOT_COLORS), "size": "medium"}
    return {"color": stable_choice(eid + "/c", OBJECT_DEFAULTS.get(kind, CLOTH_COLORS)), "size": "medium"}


def describe_entity(ent):
    a = ent["attributes"]
    if ent["type"] == "character" and ent["kind"] in HUMAN_KINDS:
        parts = [f"{a.get('hair_color')} hair", f"{a.get('clothing_color')} {a.get('clothing')}"]
    else:
        parts = [a.get("color", "")]
    if a.get("size") not in (None, "medium"):
        parts.append(a["size"])
    if a.get("state"):
        parts.append(a["state"])
    return f"{ent['name']} ({ent['kind']}; {', '.join(p for p in parts if p)})" if ent["name"] != ent["kind"] \
        else f"{ent['kind']} ({', '.join(p for p in parts if p)})"


class WorldState:
    def __init__(self, plan):
        self.order = [e["id"] for e in plan["entities"]]
        self.entities = {}
        for e in plan["entities"]:
            ent = {"id": e["id"], "name": e["name"], "type": e["type"], "kind": e["kind"],
                   "attributes": {}, "filled": [], "location": None, "held_by": None, "relation": None}
            defaults = default_attributes(e)
            for k, v in defaults.items():
                if k in e["attributes"]:
                    ent["attributes"][k] = e["attributes"][k]
                else:
                    ent["attributes"][k] = v
                    ent["filled"].append(k)
            for k, v in e["attributes"].items():
                ent["attributes"].setdefault(k, v)
            self.entities[e["id"]] = ent
        self.env = {"location": None, "weather": "clear", "time_of_day": "day"}
        # where each entity stands, per location: written by the control adapter so positions persist
        self.positions = {}
        self.snapshots = []

    def _hold(self, target, actor, log):
        target["held_by"], target["relation"] = actor["id"], None
        for other in self.entities.values():
            if other["relation"] and other["relation"][1] == target["id"]:
                other["relation"] = None
        log.append(f"{target['name']} held by {actor['name']}")

    def apply(self, scene):
        log = []
        env = self.env
        new_loc = scene["location"] or env["location"] or "park"
        if new_loc != env["location"]:
            log.append(f"location: {env['location'] or '-'} -> {new_loc}")
        env["location"] = new_loc
        for key in ("weather", "time_of_day"):
            if scene.get(key) and scene[key] != env[key]:
                log.append(f"{key.replace('_', ' ')}: {env[key]} -> {scene[key]}")
                env[key] = scene[key]

        for c in scene["attribute_changes"]:
            ent = self.entities[c["entity"]]
            old = ent["attributes"].get(c["attribute"])
            if old != c["value"]:
                ent["attributes"][c["attribute"]] = c["value"]
                log.append(f"{ent['name']}.{c['attribute']}: {old} -> {c['value']}")

        for eid in scene["exits"]:
            ent = self.entities[eid]
            ent["location"], ent["relation"] = None, None
            log.append(f"{ent['name']} leaves")

        for eid in scene["present"]:
            ent = self.entities[eid]
            if ent["type"] == "character":
                if ent["location"] != new_loc:
                    log.append(f"{ent['name']}: {ent['location'] or 'enters'} -> {new_loc}"
                               if ent["location"] else f"{ent['name']} enters ({new_loc})")
                    ent["relation"] = None
                ent["location"] = new_loc
            elif ent["held_by"] is None and ent["location"] != new_loc:
                log.append(f"{ent['name']} {'appears' if ent['location'] is None else 'moves'} in {new_loc}")
                ent["location"] = new_loc

        stated = {r["subject"] for r in scene["relations"]}
        for a in scene["actions"]:
            actor = self.entities[a["actor"]]
            target = self.entities.get(a["target"]) if a["target"] else None
            verb = a["verb"]
            if verb in ("pick_up", "hold") and target and target["held_by"] != actor["id"] and \
                    (target["type"] == "object" or (target["kind"] in CARRYABLE and actor["kind"] in HUMAN_KINDS)):
                self._hold(target, actor, log)
            elif verb == "put" and target and target["held_by"]:
                target["held_by"] = None
                target["location"] = new_loc
                log.append(f"{actor['name']} puts down {target['name']}")
            elif verb in ("open", "close") and target and target["type"] == "object":
                state = "open" if verb == "open" else "closed"
                if target["attributes"].get("state") != state:
                    target["attributes"]["state"] = state
                    log.append(f"{target['name']}: {state}")
                if verb == "open" and target["kind"] in HANDHELD and target["held_by"] != actor["id"]:
                    self._hold(target, actor, log)
            elif verb == "read" and target and target["kind"] in HANDHELD and target["held_by"] != actor["id"]:
                self._hold(target, actor, log)
            elif verb == "sit" and target and target["type"] == "object" and actor["id"] not in stated:
                actor["relation"] = ("on", target["id"])
            elif verb in ("walk", "run", "go", "stand", "dance", "jump") and actor["id"] not in stated:
                actor["relation"] = None  # standing up / moving off ends "on the bench"

        set_now = set()
        for r in scene["relations"]:
            subj, obj = self.entities[r["subject"]], self.entities[r["object"]]
            if subj["id"] in set_now and r["relation"] in MIRROR and obj["id"] not in set_now:
                # "sits on a bench next to a ball": one slot per entity, so the ball goes next to her instead
                subj, obj = obj, subj
                r = dict(r, relation=MIRROR[r["relation"]])
            elif subj["id"] in set_now:
                continue
            set_now.add(subj["id"])
            if subj["relation"] != (r["relation"], obj["id"]):
                log.append(f"{subj['name']} {r['relation'].replace('_', ' ')} {obj['name']}")
            subj["relation"] = (r["relation"], obj["id"])
            if subj["held_by"] and r["relation"] in ("on", "under", "next_to", "near", "behind", "in_front_of"):
                subj["held_by"] = None
            if obj["location"] is None:
                obj["location"] = new_loc
            subj["location"] = obj["location"]

        # carried things go wherever their holder goes
        for ent in self.entities.values():
            if ent["held_by"]:
                ent["location"] = self.entities[ent["held_by"]]["location"]

        visible = [eid for eid in self.order if self.entities[eid]["location"] == new_loc]
        relations = []
        for eid in visible:
            ent = self.entities[eid]
            if ent["held_by"]:
                relations.append({"subject": eid, "relation": "held_by", "object": ent["held_by"]})
            elif ent["relation"] and self.entities[ent["relation"][1]]["location"] == new_loc \
                    and not self.entities[ent["relation"][1]]["held_by"]:
                relations.append({"subject": eid, "relation": ent["relation"][0], "object": ent["relation"][1]})

        snap = {
            "index": scene["index"],
            "env": dict(env, indoors=new_loc in INDOORS),
            "visible": visible,
            "entities": {eid: copy.deepcopy({k: self.entities[eid][k] for k in
                                             ("name", "type", "kind", "attributes", "filled", "held_by")})
                         for eid in visible},
            "relations": relations,
            "changes": log,
        }
        self.snapshots.append(snap)
        return snap


# ---------------------------------------------------------------------------------------------------------
# visualisation

def draw_graphs(snapshots, path, max_cols=4):
    """One small graph per scene: location hub, entity nodes, relation edges, and what changed."""
    n = len(snapshots)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.axis("off")
    for k, snap in enumerate(snapshots):
        ax = axes[k // cols][k % cols]
        _draw_one(ax, snap)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


_EDGE_COLORS = {"held_by": "#c9762f", "on": "#2f6fc9", "under": "#7a4fc9", "next_to": "#2f9c6a",
                "near": "#2f9c6a", "behind": "#888", "in_front_of": "#888", "left_of": "#888", "right_of": "#888"}


def _draw_one(ax, snap):
    env = snap["env"]
    ids = snap["visible"]
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.55, 1.35)
    ax.set_title(f"Scene {snap['index']}: {env['location']}" +
                 ("" if env["indoors"] or env["weather"] == "clear" else f", {env['weather']}") +
                 ("" if env["time_of_day"] == "day" else f", {env['time_of_day']}"), fontsize=11)
    ax.add_patch(plt.Circle((0, 0), 0.95, fill=False, ec="#eee", lw=1, zorder=0))
    pos = {}
    for i, eid in enumerate(ids):
        ang = np.pi / 2 + 2 * np.pi * i / max(1, len(ids)) + (np.pi / 6 if len(ids) % 2 == 0 else 0)
        pos[eid] = (0.95 * np.cos(ang), 0.95 * np.sin(ang))
    for r in snap["relations"]:
        if r["subject"] in pos and r["object"] in pos:
            (x0, y0), (x1, y1) = pos[r["subject"]], pos[r["object"]]
            col = _EDGE_COLORS.get(r["relation"], "#888")
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, shrinkA=16, shrinkB=16), zorder=2)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, r["relation"].replace("_", " "), fontsize=7, color=col,
                    ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5), zorder=4)
    for eid, (x, y) in pos.items():
        ent = snap["entities"][eid]
        is_char = ent["type"] == "character"
        a = ent["attributes"]
        face = _swatch(a.get("clothing_color") or a.get("color"))
        ax.add_patch(plt.Circle((x, y), 0.17, color=face, ec="#222" if is_char else "#777",
                                lw=2.2 if is_char else 1, zorder=3))
        ax.text(x, y - 0.27, ent["name"], ha="center", va="top", fontsize=8, fontweight="bold" if is_char else None)
        detail = f"{a.get('hair_color')} hair" if "hair_color" in a else a.get("state", "")
        if detail:
            ax.text(x, y - 0.41, detail, ha="center", va="top", fontsize=6.5, color="#555")
    changes = snap["changes"][:4]
    ax.text(-1.4, -1.5, "\n".join("+ " + c for c in changes) if changes else "(no state change)",
            fontsize=6.8, color="#444", va="bottom", ha="left", family="monospace")


def _swatch(name):
    from .generator import COLOR_BGR
    b, g, r = COLOR_BGR.get(name or "", (180, 180, 180))
    return (r / 255, g / 255, b / 255)
