"""Saved sample outputs, so visitors can see real results without uploading anything.

Every result page stages a snapshot of its run (images copied + numbers as JSON) under
static/samples/.pending/. Clicking "Save as sample" promotes it to static/samples/<hw>/<id>/,
which is committed to git so the deployed site shows it (Render's disk is wiped on deploy).
Saving/deleting is only allowed when running locally (debug) or with SAMPLES_EDITABLE=1.
"""
import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime

import cv2
from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for

HOMEWORKS = {
    "hw2": "Homework 2 - Camera Calibration",
    "hw3": "Homework 3 - Blurring: Space vs. Frequency",
    "hw4": "Homework 4 - Human Boundary Segmentation",
    "hw6": "Homework 6 - Optical Flow, Tracking & Structure from Motion",
    "challenge": "Challenge 1 - Consistent Long-Form Generative Video",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_PENDING_MAX_AGE_S = 24 * 3600
_MAX_IMG_DIM = 1600
_VIDEO_EXTS = {".mp4", ".webm"}

bp = Blueprint("samples", __name__)


def _root():
    return os.path.join(current_app.static_folder, "samples")


def _pending_root():
    return os.path.join(_root(), ".pending")


def editable():
    return current_app.debug or os.environ.get("SAMPLES_EDITABLE") == "1"


def _cleanup_pending():
    root = _pending_root()
    if not os.path.isdir(root):
        return
    now = time.time()
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if now - os.path.getmtime(path) > _PENDING_MAX_AGE_S:
            shutil.rmtree(path, ignore_errors=True)


def _copy_image(src, dst_dir, index):
    """Copies an output image, downscaling huge phone photos so the repo stays small."""
    ext = os.path.splitext(src)[1].lower() or ".png"
    fname = f"{index:02d}{ext}"
    if ext in _VIDEO_EXTS:
        shutil.copy(src, os.path.join(dst_dir, fname))
        return fname
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        shutil.copy(src, os.path.join(dst_dir, fname))
        return fname
    h, w = img.shape[:2]
    if max(h, w) > _MAX_IMG_DIM:
        scale = _MAX_IMG_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dst_dir, fname), img)
    return fname


def stage(hw, page, title, images=(), metrics=(), params=(), table_html=None, note=None,
          table_title=None, math=False):
    """Snapshots a finished run so it can be saved as a sample later.

    images:  [(absolute_path, caption), ...] - add a third True item for a full-width figure
             (.mp4 / .webm files are copied as-is and shown as videos)
    math:    True if table_html contains LaTeX that needs MathJax
    metrics: [(label, formatted_value), ...]  - the headline numbers
    params:  [(label, formatted_value), ...]  - the inputs used for the run
    Returns a token for the "Save as sample" form, or None when saving is disabled.
    """
    if not editable():
        return None
    _cleanup_pending()

    token = uuid.uuid4().hex[:12]
    pending_dir = os.path.join(_pending_root(), token)
    os.makedirs(pending_dir, exist_ok=True)

    saved_images = []
    for i, (path, caption, *rest) in enumerate(images):
        if path and os.path.exists(path):
            fname = _copy_image(path, pending_dir, i)
            saved_images.append({"file": fname, "caption": caption, "wide": bool(rest and rest[0]),
                                 "video": os.path.splitext(fname)[1] in _VIDEO_EXTS})

    meta = {
        "hw": hw,
        "page": page,
        "title": title,
        "images": saved_images,
        "metrics": [{"label": l, "value": v} for l, v in metrics],
        "params": [{"label": l, "value": v} for l, v in params],
        "table_html": table_html,
        "table_title": table_title,
        "math": math,
        "note": note,
    }
    with open(os.path.join(pending_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return token


def _load(hw, sid):
    path = os.path.join(_root(), hw, sid, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        meta = json.load(fh)
    meta["id"] = sid
    for img in meta["images"]:
        img["url"] = url_for("static", filename=f"samples/{hw}/{sid}/{img['file']}")
    return meta


def list_samples(hw):
    hw_dir = os.path.join(_root(), hw)
    if not os.path.isdir(hw_dir):
        return []
    samples = [_load(hw, sid) for sid in sorted(os.listdir(hw_dir), reverse=True)]
    return [s for s in samples if s]


def _check_hw(hw):
    if hw not in HOMEWORKS:
        abort(404)


def _check_id(value):
    if not value or not _SAFE_ID.match(value):
        abort(400)


@bp.app_context_processor
def _inject():
    return {"samples_editable": editable()}


@bp.route("/<hw>/")
def gallery(hw):
    _check_hw(hw)
    return render_template("samples/gallery.html", layout=f"{hw}/_layout.html", hw=hw,
                           hw_title=HOMEWORKS[hw], samples=list_samples(hw), active_page="samples")


@bp.route("/<hw>/<sid>")
def detail(hw, sid):
    _check_hw(hw)
    _check_id(sid)
    sample = _load(hw, sid)
    if sample is None:
        abort(404)
    return render_template("samples/detail.html", layout=f"{hw}/_layout.html", hw=hw,
                           sample=sample, active_page="samples")


@bp.route("/save", methods=["POST"])
def save():
    if not editable():
        abort(403)
    token = request.form.get("token")
    _check_id(token)
    pending_dir = os.path.join(_pending_root(), token)
    meta_path = os.path.join(pending_dir, "meta.json")
    if not os.path.exists(meta_path):
        flash("That run has expired - run it again, then save.")
        return redirect(request.referrer or url_for("landing"))

    with open(meta_path) as fh:
        meta = json.load(fh)
    hw = meta["hw"]
    _check_hw(hw)
    meta["caption"] = request.form.get("caption", "").strip()
    meta["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    sid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{token}"
    os.makedirs(os.path.join(_root(), hw), exist_ok=True)
    shutil.move(pending_dir, os.path.join(_root(), hw, sid))

    flash("Saved as a sample output. Commit static/samples/ and push so it shows on the live site.")
    return redirect(url_for("samples.detail", hw=hw, sid=sid))


@bp.route("/<hw>/<sid>/delete", methods=["POST"])
def delete(hw, sid):
    if not editable():
        abort(403)
    _check_hw(hw)
    _check_id(sid)
    shutil.rmtree(os.path.join(_root(), hw, sid), ignore_errors=True)
    flash("Sample deleted.")
    return redirect(url_for("samples.gallery", hw=hw))
