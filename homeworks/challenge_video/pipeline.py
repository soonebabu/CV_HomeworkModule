"""Runs the whole prototype: Prompt -> Planner -> State Graph -> Control Adapter -> Generator -> Evaluation.

Produces, for the same story, a baseline video (each scene prompted on its own) and an orchestrated video
(scenes conditioned on the persistent state graph), plus everything needed to see why they differ.
"""
import json
import os
import zlib

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..hw6_motion.video_io import VideoWriter
from . import control_adapter, generator, judge, metrics, planner, state_graph

FPS = 12
ORCH_COLOR = "#2a78d6"
BASE_COLOR = "#eb6834"


def run(story, out_dir, run_id, plan_json=None, frames_per_scene=24, planner_backend="auto", use_judge=False):
    if plan_json and plan_json.strip():
        plan = planner.normalize_plan(plan_json, story)
        plan["source"] = plan.get("source") or "edited plan (pasted JSON)"
        plan_info = {"planner": "plan supplied by the user", "fallback_reason": None}
    else:
        plan, plan_info = planner.plan(story, planner_backend)
    if not plan["scenes"]:
        raise ValueError("The planner found no scenes - write at least one full sentence.")
    if not plan["entities"]:
        raise ValueError("The planner found no characters or objects to track.")

    world = state_graph.WorldState(plan)
    n = len(plan["scenes"])
    story_seed = zlib.crc32((story or plan["title"]).encode("utf-8"))
    snaps, conds, renders = [], {"baseline": [], "orchestrated": []}, {"baseline": [], "orchestrated": []}
    prev_loc = None
    for sc in plan["scenes"]:
        snap = world.apply(sc)
        snaps.append(snap)
        co = control_adapter.orchestrated(sc, snap, world, n, prev_loc)
        cb = control_adapter.baseline(sc, plan, world.entities, story_seed + 7919 * sc["index"])
        prev_loc = snap["env"]["location"]
        for arm, c in (("baseline", cb), ("orchestrated", co)):
            frames, labels, ids = generator.render_clip(c, frames_per_scene)
            conds[arm].append(c)
            renders[arm].append({"frames": frames, "labels": labels, "ids": ids,
                                 "names": {e["id"]: e["name"] for e in c["entities"]}})

    evals = {arm: metrics.evaluate(renders[arm], plan, snaps) for arm in renders}

    files = {}
    files["video_orchestrated"] = _write_video(renders["orchestrated"], plan, out_dir, f"{run_id}_orchestrated")
    files["video_baseline"] = _write_video(renders["baseline"], plan, out_dir, f"{run_id}_baseline")
    files["video_compare"] = _write_compare(renders, plan, out_dir, f"{run_id}_compare")
    files["storyboard"] = _storyboard(renders, plan, out_dir, f"{run_id}_storyboard.png")
    files["layout"] = _layout_board(conds["orchestrated"], out_dir, f"{run_id}_layout.png")
    files["graph"] = f"{run_id}_graph.png"
    state_graph.draw_graphs(snaps, os.path.join(out_dir, files["graph"]))
    files["chart"] = _chart(evals, out_dir, f"{run_id}_chart.png")

    judged = None
    if use_judge:
        judged = judge.judge_runs({arm: [r["frames"] for r in renders[arm]] for arm in renders}, plan, snaps)

    effort = {
        "story_words": len((story or "").split()),
        "orchestrated_words": int(np.mean([len(c["prompt"].split()) for c in conds["orchestrated"]])),
        "baseline_words": int(np.mean([len(c["prompt"].split()) for c in conds["baseline"]])),
    }
    prompts = {
        "how_to_use": "Generate one clip per scene with any text-to-video model, once with the baseline prompts "
                      "and once with the orchestrated prompts (+ negative prompts), then upload both sets on "
                      "the 'Evaluate real clips' page in scene order.",
        "baseline": [c["prompt"] for c in conds["baseline"]],
        "orchestrated": [{"prompt": c["prompt"], "negative_prompt": c["negative_prompt"]}
                         for c in conds["orchestrated"]],
    }
    files["prompts"] = f"{run_id}_prompts.json"
    with open(os.path.join(out_dir, files["prompts"]), "w") as fh:
        json.dump(prompts, fh, indent=2)
    files["plan"] = f"{run_id}_plan.json"
    with open(os.path.join(out_dir, files["plan"]), "w") as fh:
        json.dump(plan, fh, indent=2)
    files["report"] = f"{run_id}_report.json"
    with open(os.path.join(out_dir, files["report"]), "w") as fh:
        json.dump({"plan": plan, "snapshots": snaps, "conditioning": conds, "evaluation": evals,
                   "judge": judged, "prompt_effort": effort}, fh, indent=2, default=str)

    return {"plan": plan, "plan_info": plan_info, "snapshots": snaps, "conds": conds, "evals": evals,
            "judge": judged, "files": files, "effort": effort,
            "plan_text": json.dumps({k: plan[k] for k in ("title", "entities", "scenes")}, indent=2)}


# ---------------------------------------------------------------------------------------------------------
# outputs

def _caption(plan, k):
    return f"Scene {k + 1}/{len(plan['scenes'])}: {plan['scenes'][k]['text']}"


def _write_video(arm_renders, plan, out_dir, stem):
    vw = VideoWriter(os.path.join(out_dir, stem), FPS, (generator.W, generator.H))
    for k, r in enumerate(arm_renders):
        for fr, lab in zip(r["frames"], r["labels"]):
            vw.write(generator.annotate(fr, lab, r["ids"], r["names"], _caption(plan, k)))
    vw.close()
    return os.path.basename(vw.path)


def _header(text, color_bgr, width):
    bar = np.full((22, width, 3), 28, np.uint8)
    cv2.rectangle(bar, (0, 18), (width, 22), color_bgr, -1)
    cv2.putText(bar, text, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return bar


def _hex_bgr(h):
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _write_compare(renders, plan, out_dir, stem):
    W, H = generator.W, generator.H
    hb = _header("Baseline: each scene prompted on its own", _hex_bgr(BASE_COLOR), W)
    ho = _header("Scene-Graph Orchestrator", _hex_bgr(ORCH_COLOR), W)
    vw = VideoWriter(os.path.join(out_dir, stem), FPS, (2 * W + 4, H + 22))
    for k in range(len(plan["scenes"])):
        rb, ro = renders["baseline"][k], renders["orchestrated"][k]
        for f in range(len(rb["frames"])):
            left = np.vstack([hb, generator.annotate(rb["frames"][f], rb["labels"][f], rb["ids"], rb["names"],
                                                     _caption(plan, k))])
            right = np.vstack([ho, generator.annotate(ro["frames"][f], ro["labels"][f], ro["ids"], ro["names"],
                                                      _caption(plan, k))])
            vw.write(np.hstack([left, np.full((H + 22, 4, 3), 18, np.uint8), right]))
    vw.close()
    return os.path.basename(vw.path)


def _storyboard(renders, plan, out_dir, fname, tile_w=300):
    tile_h = int(tile_w * generator.H / generator.W)
    rows = []
    for arm, label, col in (("baseline", "Baseline", BASE_COLOR), ("orchestrated", "Orchestrator", ORCH_COLOR)):
        tiles = []
        for k, r in enumerate(renders[arm]):
            f = len(r["frames"]) // 2
            img = generator.annotate(r["frames"][f], r["labels"][f], r["ids"], r["names"])
            img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(img, (0, 0), (74, 18), (25, 25, 25), -1)
            cv2.putText(img, f"Scene {k + 1}", (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1,
                        cv2.LINE_AA)
            tiles.append(img)
            tiles.append(np.full((tile_h, 3, 3), 255, np.uint8))
        row = np.hstack(tiles[:-1])
        side = np.full((tile_h, 34, 3), 255, np.uint8)
        cv2.rectangle(side, (0, 0), (5, tile_h), _hex_bgr(col), -1)
        txt = np.full((34, tile_h, 3), 255, np.uint8)
        cv2.putText(txt, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        side[:, 6:] = cv2.rotate(txt, cv2.ROTATE_90_COUNTERCLOCKWISE)[:, :28]
        rows.append(np.hstack([side, row]))
        rows.append(np.full((3, rows[-1].shape[1], 3), 255, np.uint8))
    board = np.vstack(rows[:-1])
    cv2.imwrite(os.path.join(out_dir, fname), board)
    return fname


def _layout_board(conds, out_dir, fname, tile_w=300):
    """What the control adapter hands the generator: boxes, identities, motion, per scene."""
    tile_h = int(tile_w * generator.H / generator.W)
    tiles = []
    for c in conds:
        img = np.full((tile_h, tile_w, 3), 34, np.uint8)
        gy = int(control_adapter.GROUND * tile_h)
        cv2.line(img, (0, gy), (tile_w, gy), (90, 90, 90), 1)
        for e in c["entities"]:
            if e["held_by"]:
                continue
            col = generator.color(e["attributes"].get("clothing_color") or e["attributes"].get("color"))
            for x, thick in ((e["x0"], 1), (e["x1"], 2)):
                x0 = int((x - e["w"] / 2) * tile_w)
                x1 = int((x + e["w"] / 2) * tile_w)
                y1 = int(e["y_base"] * tile_h) if e["pose"] != "sit" else int((e["y_base"] + e["h"] * 0.2) * tile_h)
                y0 = y1 - int(e["h"] * tile_h)
                if thick == 1 and abs(e["x1"] - e["x0"]) < 1e-3:
                    continue
                cv2.rectangle(img, (x0, y0), (x1, y1), col, thick, cv2.LINE_AA)
            if abs(e["x1"] - e["x0"]) > 1e-3:
                yy = int((e["y_base"] - e["h"] / 2) * tile_h)
                cv2.arrowedLine(img, (int(e["x0"] * tile_w), yy), (int(e["x1"] * tile_w), yy), (200, 200, 200), 1,
                                cv2.LINE_AA, tipLength=0.15)
            held = [h["name"] for h in c["entities"] if h["held_by"] == e["id"]]
            label = e["name"] + (f" +{','.join(held)}" if held else "")
            cv2.putText(img, label, (max(2, int((e["x1"] - e["w"] / 2) * tile_w)),
                                     max(10, int((e["y_base"] - e["h"]) * tile_h) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (235, 235, 235), 1, cv2.LINE_AA)
        bg = c["background"]
        cv2.putText(img, f"S{c['scene']} {bg['location']} seed={bg['seed'] % 10000}", (5, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(img)
        tiles.append(np.full((tile_h, 3, 3), 255, np.uint8))
    cv2.imwrite(os.path.join(out_dir, fname), np.hstack(tiles[:-1]))
    return fname


def _chart(evals, out_dir, fname):
    labels = [lbl for key, lbl in metrics.AXES] + ["Overall"]
    keys = [key for key, _ in metrics.AXES]
    vals = {arm: [evals[arm]["scores"][k] for k in keys] + [evals[arm]["overall"]] for arm in evals}
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(labels))[::-1]
    hgt = 0.36
    for arm, off, col, name in (("orchestrated", hgt / 2 + 0.01, ORCH_COLOR, "Scene-Graph Orchestrator"),
                                ("baseline", -hgt / 2 - 0.01, BASE_COLOR, "Baseline (re-prompting)")):
        v = [np.nan if x is None else x for x in vals[arm]]
        ax.barh(y + off, v, height=hgt, color=col, label=name, zorder=2)
        for yy, x in zip(y + off, v):
            if not np.isnan(x):
                ax.text(x + 0.01, yy, f"{x:.2f}", va="center", fontsize=8.5, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("score (0 = worst, 1 = best)", color="#555")
    ax.grid(axis="x", color="#e6e6e6", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(colors="#444", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=120)
    plt.close(fig)
    return fname
