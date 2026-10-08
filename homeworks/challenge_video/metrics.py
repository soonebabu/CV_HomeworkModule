"""Evaluation: baseline vs. orchestrator on the axes from the proposal.

Consistency
  character   - colour-histogram similarity of each character to its own first appearance (re-anchored only when
                the story explicitly changes their look). 1.0 = identical across all scenes.
  object      - share of objects the world state says should be visible that actually are, averaged with how
                similar each object looks to its first appearance.
  spatial     - where the narrative stays in one place: how far things jump at a scene cut, and how much the
                background (room layout) changes (cuts where the story changes weather/time are left out).
  temporal    - optical-flow warping error (Farneback from HW6) between consecutive frames, inside each clip
                and across same-location cuts. Low error = the next frame is the previous one, moved.
Adherence
  prompt      - per scene, did the clip show what that scene's own text asks for (entities, relations, place).
  narrative   - per scene, does the clip match the full expected world state from the state graph: everyone who
                should be there, every relation carried over from earlier, the right place, the locked appearance.

All of it is measured on the rendered frames. Entity masks come from the generator's label maps (standing in for
a detector/tracker), but colours, positions and relations are read from pixels.
"""
import cv2
import numpy as np

from .generator import COLOR_BGR, LOC_STYLE, H, W, background
from .planner import HUMAN_KINDS
from .state_graph import CARRYABLE

MIN_PIX = 25
THIN = {"bicycle", "lamp", "umbrella", "kite", "balloon", "guitar"}  # mostly outline, little solid colour
AXES = [("character", "Character consistency"), ("object", "Object persistence"),
        ("spatial", "Spatial consistency"), ("temporal", "Temporal consistency"),
        ("prompt", "Prompt adherence"), ("narrative", "Narrative flow")]


def _mask(lab, ids, eid):
    if eid not in ids:
        return None
    m = lab == (ids.index(eid) + 1)
    return m if m.sum() >= MIN_PIX else None


def _hist(frame, mask):
    labimg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    h = cv2.calcHist([labimg], [0, 1, 2], mask.astype(np.uint8), [6, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def _sim(h1, h2):
    return float(1.0 - cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def _mean_lab(frame, mask):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)[mask]
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    return lab.mean(axis=0)


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return {"x0": xs.min() / W, "x1": xs.max() / W, "y0": ys.min() / H, "y1": ys.max() / H,
            "cx": xs.mean() / W, "cy": ys.mean() / H}


def check_relation(rel, sb, ob):
    overlap_x = min(sb["x1"], ob["x1"]) - max(sb["x0"], ob["x0"])
    gap_x = max(sb["x0"] - ob["x1"], ob["x0"] - sb["x1"], 0)
    overlap_y = min(sb["y1"], ob["y1"]) - max(sb["y0"], ob["y0"])
    oh = ob["y1"] - ob["y0"]
    if rel == "on":
        # top of the subject above the support's top, centre above its centre (a sitter's legs may hang below)
        return overlap_x > 0 and sb["y0"] < ob["y0"] and sb["cy"] < ob["cy"]
    if rel == "under":
        return overlap_x > 0 and sb["cy"] > ob["y0"] + 0.3 * oh
    if rel in ("next_to", "near"):
        return gap_x <= (0.06 if rel == "next_to" else 0.14) and overlap_y > 0
    if rel == "left_of":
        return sb["cx"] < ob["cx"]
    if rel == "right_of":
        return sb["cx"] > ob["cx"]
    if rel in ("behind", "in_front_of"):
        return overlap_x > -0.05
    if rel == "held_by":
        return gap_x <= 0.03 and overlap_y > -0.03
    return False


_PROTO = {}


def classify_location(frame, lab, weather, time_of_day):
    """Nearest-prototype classifier on background colours (no learning involved)."""
    bgmask = (lab == 0).astype(np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], bgmask, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    h = cv2.normalize(h, h).flatten()
    best, best_sim = None, -1
    for loc in LOC_STYLE:
        key = (loc, weather, time_of_day)
        if key not in _PROTO:
            hs = []
            for seed in (1, 2, 3):
                ref = background({"location": loc, "weather": weather, "time_of_day": time_of_day, "seed": seed}, 0)
                rh = cv2.calcHist([cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [12, 6, 6],
                                  [0, 180, 0, 256, 0, 256])
                hs.append(cv2.normalize(rh, rh).flatten())
            _PROTO[key] = hs
        s = max(_sim(h, rh) for rh in _PROTO[key])
        if s > best_sim:
            best, best_sim = loc, s
    return best


def _color_share(frame, mask, name):
    """Share of an entity's pixels that are close to a named colour (plus its shaded variants)."""
    if name not in COLOR_BGR:
        return 1.0
    px = frame[mask].astype(np.float32)
    target = np.array(COLOR_BGR[name], np.float32)
    best = np.full(len(px), np.inf, np.float32)
    for f in (0.45, 0.62, 0.65, 0.78, 0.85, 0.92, 1.0, 1.15):
        d = np.linalg.norm(px - np.clip(target * f, 0, 255), axis=1)
        best = np.minimum(best, d)
    return float((best < 38).mean())


def _warp_error(f0, f1):
    from ..hw6_motion.optical_flow import farneback
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    flow = farneback(g0, g1)
    yy, xx = np.mgrid[0:g0.shape[0], 0:g0.shape[1]].astype(np.float32)
    warped = cv2.remap(g1, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return float(np.abs(g0.astype(np.float32) - warped.astype(np.float32)).mean())


def evaluate(renders, plan, snapshots):
    """renders: per scene {"frames", "labels", "ids"}. Returns axis scores (0-1) plus the evidence behind them."""
    ents = {e["id"]: e for e in plan["entities"]}
    scenes = plan["scenes"]
    n = len(scenes)
    mid = [len(r["frames"]) // 2 for r in renders]

    # --- identity: characters and objects vs. their own first appearance --------------------------------
    identity = {}
    for eid, ent in ents.items():
        anchor, rows = None, []
        for k in range(n):
            if any(c["entity"] == eid for c in scenes[k]["attribute_changes"]):
                anchor = None  # the story changed their look on purpose - start a new reference
            m = _mask(renders[k]["labels"][mid[k]], renders[k]["ids"], eid)
            if m is None:
                continue
            fr = renders[k]["frames"][mid[k]]
            h, mlab = _hist(fr, m), _mean_lab(fr, m)
            if anchor is None:
                anchor = (k, h, mlab)
                continue
            rows.append({"scene": k + 1, "vs_scene": anchor[0] + 1, "sim": _sim(anchor[1], h),
                         "delta_e": float(np.linalg.norm(anchor[2] - mlab))})
        identity[eid] = rows

    char_ids = [e for e in ents if ents[e]["type"] == "character"]
    obj_ids = [e for e in ents if ents[e]["type"] == "object"]
    char_rows = [r for e in char_ids for r in identity[e]]
    character = float(np.mean([r["sim"] for r in char_rows])) if char_rows else None

    # --- object persistence ------------------------------------------------------------------------------
    expected, shown, vanish = 0, 0, []
    for k, snap in enumerate(snapshots):
        for eid in snap["visible"]:
            if ents[eid]["type"] != "object":
                continue
            expected += 1
            here = _mask(renders[k]["labels"][mid[k]], renders[k]["ids"], eid) is not None
            shown += here
            if not here and k > 0 and eid in snapshots[k - 1]["visible"] and \
                    _mask(renders[k - 1]["labels"][mid[k - 1]], renders[k - 1]["ids"], eid) is not None:
                vanish.append(f"scene {k + 1}: {ents[eid]['name']} vanished")
    presence = shown / expected if expected else None
    obj_rows = [r for e in obj_ids for r in identity[e]]
    obj_ident = float(np.mean([r["sim"] for r in obj_rows])) if obj_rows else None
    obj_parts = [v for v in (presence, obj_ident) if v is not None]
    obj_score = float(np.mean(obj_parts)) if obj_parts else None

    # --- spatial + temporal at same-location cuts --------------------------------------------------------
    jumps, bg_stab, cut_warp = [], [], []
    for k in range(1, n):
        if snapshots[k]["env"]["location"] != snapshots[k - 1]["env"]["location"]:
            continue
        # sunset / night / rain arriving is a change the story asked for, not inconsistency
        env_change = any(snapshots[k]["env"][key] != snapshots[k - 1]["env"][key] for key in ("weather", "time_of_day"))
        a_fr, a_lab, a_ids = renders[k - 1]["frames"][-1], renders[k - 1]["labels"][-1], renders[k - 1]["ids"]
        b_fr, b_lab, b_ids = renders[k]["frames"][0], renders[k]["labels"][0], renders[k]["ids"]
        for eid in snapshots[k]["visible"]:
            ma, mb = _mask(a_lab, a_ids, eid), _mask(b_lab, b_ids, eid)
            if ma is None or mb is None:
                continue
            ba, bb = _bbox(ma), _bbox(mb)
            d = float(np.hypot(ba["cx"] - bb["cx"], (ba["cy"] - bb["cy"]) * H / W))
            jumps.append({"scene": k + 1, "entity": ents[eid]["name"], "jump": d})
        if env_change:
            continue
        both_bg = (a_lab == 0) & (b_lab == 0)
        if both_bg.sum() > 1000:
            diff = np.abs(a_fr.astype(np.float32) - b_fr.astype(np.float32)).mean(axis=2)[both_bg]
            bg_stab.append(float(max(0.0, 1 - diff.mean() / 60)))
        cut_warp.append(_warp_error(a_fr, b_fr))
    ent_cont = float(np.mean([max(0.0, 1 - j["jump"] / 0.2) for j in jumps])) if jumps else None
    bg_score = float(np.mean(bg_stab)) if bg_stab else None
    sp_parts = [v for v in (ent_cont, bg_score) if v is not None]
    spatial = float(np.mean(sp_parts)) if sp_parts else None

    in_warp = []
    for r in renders:
        fr = r["frames"]
        for i in range(0, len(fr) - 1, 3):
            in_warp.append(_warp_error(fr[i], fr[i + 1]))
    werr_in = float(np.mean(in_warp)) if in_warp else 0.0
    werr_cut = float(np.mean(cut_warp)) if cut_warp else None
    t_parts = [max(0.0, 1 - werr_in / 25)] + ([max(0.0, 1 - werr_cut / 25)] if werr_cut is not None else [])
    temporal = float(np.mean(t_parts))

    # --- adherence ---------------------------------------------------------------------------------------
    p_ok = p_tot = n_ok = n_tot = 0
    per_scene = []
    for k, (sc, snap) in enumerate(zip(scenes, snapshots)):
        r = renders[k]
        lab_end, ids = r["labels"][-1], r["ids"]
        fr_mid, lab_mid = r["frames"][mid[k]], r["labels"][mid[k]]
        loc_seen = classify_location(fr_mid, lab_mid, snap["env"]["weather"], snap["env"]["time_of_day"])
        boxes = {eid: _bbox(m) for eid in ids if (m := _mask(lab_end, ids, eid)) is not None}
        mid_shown = {eid for eid in ids if _mask(lab_mid, ids, eid) is not None}
        notes = []

        # local: the scene's own sentence
        checks = [(eid in mid_shown, f"shows {ents[eid]['name']}") for eid in sc["present"]]
        for rel in sc["relations"]:
            s, o = rel["subject"], rel["object"]
            ok = s in boxes and o in boxes and (check_relation(rel["relation"], boxes[s], boxes[o]) or
                                                check_relation(rel["relation"], boxes[o], boxes[s]))
            checks.append((ok, f"{ents[s]['name']} {rel['relation'].replace('_', ' ')} {ents[o]['name']}"))
        for a in sc["actions"]:
            if a["verb"] in ("pick_up", "hold") and a["target"] and (
                    ents[a["target"]]["type"] == "object" or ents[a["target"]]["kind"] in CARRYABLE):
                t = a["target"]
                ok = t in boxes and a["actor"] in boxes and check_relation("held_by", boxes[t], boxes[a["actor"]])
                checks.append((ok, f"{ents[a['actor']]['name']} holds {ents[t]['name']}"))
        if sc["location"]:
            checks.append((loc_seen == sc["location"], f"in the {sc['location']}"))
        checks = [(bool(ok), what) for ok, what in checks]
        p_ok += sum(c[0] for c in checks)
        p_tot += len(checks)

        # global: the full expected state for this point in the story
        facts = [(loc_seen == snap["env"]["location"], f"in the {snap['env']['location']}")]
        for eid in snap["visible"]:
            ok = eid in mid_shown
            facts.append((ok, f"{ents[eid]['name']} present"))
            if not ok:
                notes.append(f"missing {ents[eid]['name']}")
                continue
            m = _mask(lab_mid, ids, eid)
            a = snap["entities"][eid]["attributes"]
            want = a.get("clothing_color") if ents[eid]["kind"] in HUMAN_KINDS else a.get("color")
            if want:
                share = _color_share(fr_mid, m, want)
                need = 0.12 if ents[eid]["kind"] in HUMAN_KINDS else 0.15 if ents[eid]["kind"] in THIN else 0.3
                facts.append((share >= need, f"{ents[eid]['name']} is {want}"))
                if share < need:
                    notes.append(f"{ents[eid]['name']} not {want}")
        for rel in snap["relations"]:
            s, o = rel["subject"], rel["object"]
            ok = s in boxes and o in boxes and (check_relation(rel["relation"], boxes[s], boxes[o]) or
                                                (rel["relation"] in ("next_to", "near") and
                                                 check_relation(rel["relation"], boxes[o], boxes[s])))
            facts.append((ok, f"{ents[s]['name']} {rel['relation'].replace('_', ' ')} {ents[o]['name']}"))
            if not ok:
                notes.append(f"broken: {ents[s]['name']} {rel['relation'].replace('_', ' ')} {ents[o]['name']}")
        if loc_seen != snap["env"]["location"]:
            notes.insert(0, f"looks like {loc_seen}, should be {snap['env']['location']}")
        facts = [(bool(ok), what) for ok, what in facts]
        n_ok += sum(f[0] for f in facts)
        n_tot += len(facts)
        per_scene.append({"scene": k + 1, "prompt_ok": sum(c[0] for c in checks), "prompt_total": len(checks),
                          "state_ok": sum(f[0] for f in facts), "state_total": len(facts),
                          "location_seen": loc_seen, "notes": notes})

    scores = {"character": character, "object": obj_score, "spatial": spatial, "temporal": temporal,
              "prompt": p_ok / p_tot if p_tot else None, "narrative": n_ok / n_tot if n_tot else None}
    valid = [v for v in scores.values() if v is not None]
    return {
        "scores": scores,
        "overall": float(np.mean(valid)) if valid else 0.0,
        "details": {
            "character_rows": [dict(r, name=ents[e]["name"]) for e in char_ids for r in identity[e]],
            "object_presence": presence, "objects_expected": expected, "objects_shown": shown,
            "object_identity": obj_ident, "vanish_events": vanish,
            "cut_jumps": jumps, "entity_continuity": ent_cont, "background_stability": bg_score,
            "warp_error_in_clip": werr_in, "warp_error_at_cuts": werr_cut,
            "per_scene": per_scene,
        },
    }
