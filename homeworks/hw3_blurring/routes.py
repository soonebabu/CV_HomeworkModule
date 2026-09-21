import os
import uuid

import cv2
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from . import blur

bp = Blueprint("hw3", __name__)


def _dirs():
    static_dir = current_app.static_folder
    upload_dir = os.path.join(static_dir, "hw3", "uploads")
    output_dir = os.path.join(static_dir, "hw3", "outputs")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return upload_dir, output_dir


@bp.route("/")
def overview():
    return render_template("hw3/overview.html", active_page="overview")


@bp.route("/theory")
def theory():
    return render_template("hw3/theory.html", active_page="theory")


@bp.route("/blur", methods=["GET", "POST"])
def blur_view():
    if request.method == "GET":
        return render_template("hw3/blur.html", active_page="blur", result=None)

    upload_dir, output_dir = _dirs()

    f = request.files.get("image")
    if not f or not f.filename:
        flash("Choose an image first.")
        return redirect(url_for("hw3.blur_view"))

    kernel_type = request.form.get("kernel_type", "gaussian")
    ksize = int(request.form.get("ksize", 15))
    sigma = float(request.form.get("sigma", 3.0))

    run_id = uuid.uuid4().hex[:10]
    in_path = os.path.join(upload_dir, f"{run_id}_{f.filename}")
    f.save(in_path)

    img = cv2.imread(in_path)
    if img is None:
        flash("Could not read that image - try a JPG or PNG.")
        return redirect(url_for("hw3.blur_view"))

    # keep the demo fast for large phone photos
    max_dim = 900
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    out = blur.blur_compare(img, kernel_type=kernel_type, ksize=ksize, sigma=sigma)

    comparison_fname = f"{run_id}_comparison.png"
    blur.build_comparison_figure(out, kernel_type, os.path.join(output_dir, comparison_fname))

    diff_fname = f"{run_id}_diff.png"
    cv2.imwrite(os.path.join(output_dir, diff_fname), out["diff_vis"])

    urls = {
        "comparison": url_for("static", filename=f"hw3/outputs/{comparison_fname}"),
        "diff": url_for("static", filename=f"hw3/outputs/{diff_fname}"),
    }

    result = {
        "urls": urls,
        "metrics": out["metrics"],
        "kernel_type": kernel_type,
        "ksize": ksize,
        "sigma": sigma,
    }
    return render_template("hw3/blur.html", active_page="blur", result=result)
