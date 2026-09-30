import os
import uuid

import cv2
import numpy as np
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from .. import samples
from . import metrics as mx
from . import rgb_segmentation as rgbseg
from . import thermal_segmentation as thermseg

bp = Blueprint("hw4", __name__)


def _dirs():
    static_dir = current_app.static_folder
    upload_dir = os.path.join(static_dir, "hw4", "uploads")
    output_dir = os.path.join(static_dir, "hw4", "outputs")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return upload_dir, output_dir


def _resize_cap(img, max_dim=900):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _save_png(output_dir, run_id, name, arr):
    fname = f"{run_id}_{name}.png"
    cv2.imwrite(os.path.join(output_dir, fname), arr)
    return url_for("static", filename=f"hw4/outputs/{fname}")


def _stage_sample(page, title, output_dir, run_id, images, params, sam2):
    """Snapshots a segmentation run for the sample-outputs gallery."""
    image_list = [(os.path.join(output_dir, f"{run_id}_{name}.png"), caption) for name, caption in images]
    metrics = []
    if sam2:
        metrics = [("IoU vs. SAM2", f"{sam2['iou']:.3f}"), ("Dice vs. SAM2", f"{sam2['dice']:.3f}")]
        image_list += [
            (os.path.join(output_dir, f"{run_id}_sam2_mask.png"), "SAM2 mask (uploaded)"),
            (os.path.join(output_dir, f"{run_id}_sam2_overlay.png"),
             "Agreement: white = both, green = ours only, red = SAM2 only"),
        ]
    return samples.stage("hw4", page, title, images=image_list, params=params, metrics=metrics)


def _sam2_comparison(sam2_file, upload_dir, run_id, our_mask, shape_hw, output_dir):
    """Saves the uploaded SAM2 mask, binarizes/resizes it to match our mask, and scores it."""
    sam2_path = os.path.join(upload_dir, f"{run_id}_sam2_mask_{sam2_file.filename}")
    sam2_file.save(sam2_path)
    sam2_raw = cv2.imread(sam2_path, cv2.IMREAD_UNCHANGED)
    if sam2_raw is None:
        return None

    sam2_bin = mx.to_binary_mask(sam2_raw)
    if sam2_bin.shape != shape_hw:
        sam2_bin = cv2.resize(sam2_bin, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)

    iou = mx.iou(our_mask, sam2_bin)
    dice = mx.dice(our_mask, sam2_bin)
    overlay = mx.overlay_comparison(shape_hw, our_mask, sam2_bin)

    return {
        "iou": iou,
        "dice": dice,
        "sam2_mask_url": _save_png(output_dir, run_id, "sam2_mask", mx.mask_to_display(sam2_bin)),
        "overlay_url": _save_png(output_dir, run_id, "sam2_overlay", overlay),
    }


@bp.route("/")
def overview():
    return render_template("hw4/overview.html", active_page="overview")


@bp.route("/theory")
def theory():
    return render_template("hw4/theory.html", active_page="theory")


@bp.route("/rgb", methods=["GET", "POST"])
def rgb_view():
    upload_dir, output_dir = _dirs()

    if request.method == "GET":
        return render_template("hw4/rgb.html", active_page="rgb", image_url=None, result=None)

    # Step 1: just the photo, uploaded so it can be shown for rectangle selection.
    if "image" in request.files and request.files["image"].filename:
        f = request.files["image"]
        run_id = uuid.uuid4().hex[:10]
        in_path = os.path.join(upload_dir, f"{run_id}_{f.filename}")
        f.save(in_path)

        img = cv2.imread(in_path)
        if img is None:
            flash("Could not read that image - try a JPG or PNG.")
            return redirect(url_for("hw4.rgb_view"))
        img = _resize_cap(img)
        resized_path = os.path.join(upload_dir, f"{run_id}_resized.png")
        cv2.imwrite(resized_path, img)
        image_url = url_for("static", filename=f"hw4/uploads/{run_id}_resized.png")
        return render_template("hw4/rgb.html", active_page="rgb", image_url=image_url,
                                run_id=run_id, img_w=img.shape[1], img_h=img.shape[0], result=None)

    # Step 2: run GrabCut with either a manual or automatic rectangle.
    run_id = request.form.get("run_id")
    in_path = os.path.join(upload_dir, f"{run_id}_resized.png")
    img = cv2.imread(in_path)
    if img is None:
        flash("Session expired - upload the photo again.")
        return redirect(url_for("hw4.rgb_view"))

    use_auto = request.form.get("use_auto") == "on"
    if use_auto:
        rect = rgbseg.default_rect(img.shape)
    else:
        rx, ry = int(float(request.form["rx"])), int(float(request.form["ry"]))
        rw, rh = int(float(request.form["rw"])), int(float(request.form["rh"]))
        rect = (rx, ry, max(rw, 5), max(rh, 5))

    mask, contour = rgbseg.grabcut_segment(img, rect)

    boundary_img = mx.boundary_overlay(img, contour)
    rect_img = img.copy()
    cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 200, 0), 2)

    urls = {
        "original": _save_png(output_dir, run_id, "rgb_original", img),
        "seed_rect": _save_png(output_dir, run_id, "rgb_seed_rect", rect_img),
        "mask": _save_png(output_dir, run_id, "rgb_mask", mx.mask_to_display(mask)),
        "boundary": _save_png(output_dir, run_id, "rgb_boundary", boundary_img),
    }

    sam2 = None
    sam2_file = request.files.get("sam2_mask")
    if sam2_file and sam2_file.filename:
        sam2 = _sam2_comparison(sam2_file, upload_dir, run_id, mask, img.shape[:2], output_dir)

    result = {"urls": urls, "rect": rect, "sam2": sam2}
    sample_token = _stage_sample(
        "rgb", "RGB boundary (GrabCut)", output_dir, run_id,
        images=[("rgb_seed_rect", "GrabCut seed rectangle"), ("rgb_mask", "Segmentation mask"),
                ("rgb_boundary", "Extracted boundary (largest contour)")],
        params=[("Seed box", "automatic (inset 5%)" if use_auto else "drawn by hand"),
                ("Box (x, y, w, h)", ", ".join(str(v) for v in rect))],
        sam2=sam2,
    )
    return render_template("hw4/rgb.html", active_page="rgb", image_url=urls["original"],
                            run_id=run_id, img_w=img.shape[1], img_h=img.shape[0], result=result,
                            sample_token=sample_token)


@bp.route("/thermal", methods=["GET", "POST"])
def thermal_view():
    upload_dir, output_dir = _dirs()

    if request.method == "GET":
        return render_template("hw4/thermal.html", active_page="thermal", result=None)

    f = request.files.get("image")
    if not f or not f.filename:
        flash("Choose a thermal image first.")
        return redirect(url_for("hw4.thermal_view"))

    hot_is_bright = request.form.get("hot_is_bright", "on") == "on"

    run_id = uuid.uuid4().hex[:10]
    in_path = os.path.join(upload_dir, f"{run_id}_{f.filename}")
    f.save(in_path)

    img = cv2.imread(in_path)
    if img is None:
        flash("Could not read that image - try a JPG or PNG.")
        return redirect(url_for("hw4.thermal_view"))
    img = _resize_cap(img)

    mask, contour, gray_used = thermseg.thermal_segment(img, hot_is_bright=hot_is_bright)
    boundary_img = mx.boundary_overlay(img, contour, color=(0, 140, 255))

    urls = {
        "original": _save_png(output_dir, run_id, "thermal_original", img),
        "grayscale": _save_png(output_dir, run_id, "thermal_gray", gray_used),
        "mask": _save_png(output_dir, run_id, "thermal_mask", mx.mask_to_display(mask)),
        "boundary": _save_png(output_dir, run_id, "thermal_boundary", boundary_img),
    }

    sam2 = None
    sam2_file = request.files.get("sam2_mask")
    if sam2_file and sam2_file.filename:
        sam2 = _sam2_comparison(sam2_file, upload_dir, run_id, mask, img.shape[:2], output_dir)

    result = {"urls": urls, "sam2": sam2, "hot_is_bright": hot_is_bright}
    sample_token = _stage_sample(
        "thermal", "Thermal boundary (Otsu)", output_dir, run_id,
        images=[("thermal_original", "Thermal input"),
                ("thermal_gray", "Grayscale intensity used for thresholding"),
                ("thermal_mask", "Segmentation mask (Otsu + morphology + largest blob)"),
                ("thermal_boundary", "Extracted boundary (largest contour)")],
        params=[("Image", f.filename), ("Palette", "white-hot" if hot_is_bright else "black-hot")],
        sam2=sam2,
    )
    return render_template("hw4/thermal.html", active_page="thermal", result=result,
                            sample_token=sample_token)
