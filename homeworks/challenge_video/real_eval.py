"""Evaluating real generated clips (from any text-to-video tool) without masks or ground truth.

The prototype run exports two prompt sets (baseline / orchestrated). Generate one clip per scene with each set,
upload them here in scene order, and both sets get the same model-free measurements:

  temporal   - Farneback warping error between consecutive frames inside each clip (HW6 optical flow)
  cut        - warping error across each scene cut (last frame of clip k -> first frame of clip k+1); only
               cuts where the plan keeps the same location are counted when a plan is given
  appearance - colour-histogram correlation of every clip's middle frame with clip 1's (global look drift)
  structure  - ORB features matched between consecutive clips (ratio test + RANSAC homography); the share of
               matches that survive says how much of the scene/characters is literally the same
plus the optional Video-LLM judge on frames sampled from every clip.
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..hw6_motion.video_io import iter_frames
from . import judge, metrics

ORCH_COLOR = "#2a78d6"
BASE_COLOR = "#eb6834"
ARM_NAMES = {"baseline": "Baseline", "orchestrated": "Orchestrator"}


def _load(path, max_frames=96, max_dim=320):
    frames = [f for _, _, f in iter_frames(path, 0.0, 20.0, max_dim=max_dim)]
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def _hsv_hist(f):
    h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _orb_share(a, b):
    orb = cv2.ORB_create(800)
    ka, da = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < 8:
        return 0.0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(inl.sum()) if inl is not None else 0
    return min(1.0, inliers / 60.0)


def _same_place(plan, n):
    """For each cut k-1 -> k: does the plan keep the location? Without a plan every cut counts."""
    if not plan:
        return [True] * n
    locs, cur = [], None
    for sc in plan["scenes"]:
        cur = sc.get("location") or cur
        locs.append(cur)
    return [k < len(locs) and k > 0 and locs[k] == locs[k - 1] for k in range(n)]


def evaluate_arm(clips, plan=None):
    """clips: list of frame lists (one per scene)."""
    in_err = []
    for fr in clips:
        step = max(1, len(fr) // 16)
        for i in range(0, len(fr) - 1, step):
            in_err.append(metrics._warp_error(fr[i], fr[i + 1]))
    same = _same_place(plan, len(clips))
    cut_err, structure = [], []
    for k in range(1, len(clips)):
        structure.append(_orb_share(clips[k - 1][len(clips[k - 1]) // 2], clips[k][len(clips[k]) // 2]))
        if same[k]:
            a, b = clips[k - 1][-1], clips[k][0]
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            cut_err.append(metrics._warp_error(a, b))
    ref = _hsv_hist(clips[0][len(clips[0]) // 2])
    drift = [max(0.0, float(cv2.compareHist(ref, _hsv_hist(c[len(c) // 2]), cv2.HISTCMP_CORREL))) for c in clips[1:]]
    werr = float(np.mean(in_err)) if in_err else None
    cerr = float(np.mean(cut_err)) if cut_err else None
    scores = {
        "temporal": None if werr is None else max(0.0, 1 - werr / 25),
        "cut": None if cerr is None else max(0.0, 1 - cerr / 40),
        "appearance": float(np.mean(drift)) if drift else None,
        "structure": float(np.mean(structure)) if structure else None,
    }
    return {"scores": scores, "warp_error": werr, "cut_error": cerr, "drift_curve": [1.0] + drift,
            "structure_per_cut": structure, "n_clips": len(clips), "n_frames": sum(len(c) for c in clips)}


REAL_AXES = [("temporal", "Temporal (in-clip warp)"), ("cut", "Cut continuity"),
             ("appearance", "Appearance stability"), ("structure", "Shared structure")]


def run(paths_by_arm, out_dir, run_id, plan=None, use_judge=False):
    clips = {arm: [_load(p) for p in paths] for arm, paths in paths_by_arm.items() if paths}
    for arm, cl in clips.items():
        bad = [i + 1 for i, c in enumerate(cl) if len(c) < 2]
        if bad:
            raise ValueError(f"{ARM_NAMES[arm]} clip(s) {bad} could not be read as video.")
    results = {arm: evaluate_arm(cl, plan) for arm, cl in clips.items()}
    files = {"sheet": _contact_sheet(clips, out_dir, f"{run_id}_sheet.png"),
             "chart": _chart(results, out_dir, f"{run_id}_chart.png")}
    judged = judge.judge_runs(clips, plan) if use_judge else None
    return {"results": results, "files": files, "judge": judged, "arms": list(clips)}


def _contact_sheet(clips, out_dir, fname, tile_w=240):
    rows = []
    width = None
    for arm, cl in clips.items():
        tiles = []
        for k, c in enumerate(cl):
            f = c[len(c) // 2]
            th = int(tile_w * f.shape[0] / f.shape[1])
            t = cv2.resize(f, (tile_w, th))
            t = cv2.resize(t, (tile_w, int(tile_w * 9 / 16))) if th != int(tile_w * 9 / 16) else t
            cv2.rectangle(t, (0, 0), (110, 16), (25, 25, 25), -1)
            cv2.putText(t, f"{ARM_NAMES[arm]} {k + 1}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1,
                        cv2.LINE_AA)
            tiles.append(t)
        row = np.hstack(tiles)
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    cv2.imwrite(os.path.join(out_dir, fname), np.vstack(rows))
    return fname


def _chart(results, out_dir, fname):
    labels = [l for _, l in REAL_AXES]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(labels))[::-1]
    arms = [a for a in ("orchestrated", "baseline") if a in results]
    hgt = 0.36 if len(arms) == 2 else 0.5
    for i, arm in enumerate(arms):
        off = (hgt / 2 + 0.01) * (1 if i == 0 else -1) if len(arms) == 2 else 0
        v = [np.nan if results[arm]["scores"][k] is None else results[arm]["scores"][k] for k, _ in REAL_AXES]
        ax.barh(y + off, v, height=hgt, color=ORCH_COLOR if arm == "orchestrated" else BASE_COLOR,
                label=ARM_NAMES[arm], zorder=2)
        for yy, x in zip(y + off, v):
            if not np.isnan(x):
                ax.text(x + 0.01, yy, f"{x:.2f}", va="center", fontsize=8.5, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("score (0 = worst, 1 = best)", color="#555")
    ax.grid(axis="x", color="#e6e6e6", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(colors="#444", length=0)
    if len(arms) > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=120)
    plt.close(fig)
    return fname
