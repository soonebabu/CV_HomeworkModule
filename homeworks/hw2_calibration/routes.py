import os

import cv2
import numpy as np
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from .. import samples
from . import calibration
from . import validate_measurements as vm
from .measure_dimensions import pixel_length_to_real, undistort_points

bp = Blueprint("hw2", __name__)


def _dirs():
    static_dir = current_app.static_folder
    upload_calib_dir = os.path.join(static_dir, "calib_uploads")
    upload_obj_dir = os.path.join(static_dir, "object_uploads")
    calib_file = os.path.join(static_dir, "calib", "camera_calib.npz")
    results_dir = os.path.join(static_dir, "results")
    for d in [upload_calib_dir, upload_obj_dir, os.path.dirname(calib_file), results_dir]:
        os.makedirs(d, exist_ok=True)
    return upload_calib_dir, upload_obj_dir, calib_file, results_dir


def get_current_calibration():
    _, _, calib_file, _ = _dirs()
    if not os.path.exists(calib_file):
        return None
    data = np.load(calib_file)
    return {
        "camera_matrix": data["camera_matrix"],
        "dist_coeffs": data["dist_coeffs"],
        "reprojection_error": float(data["reprojection_error"]),
        "image_size": tuple(data["image_size"]),
    }


@bp.route("/")
def overview():
    calib = get_current_calibration()
    return render_template("hw2/overview.html", calib=calib, active_page="overview")


@bp.route("/calibrate", methods=["GET", "POST"])
def calibrate_view():
    upload_calib_dir, _, calib_file, results_dir = _dirs()

    if request.method == "GET":
        return render_template("hw2/calibrate.html", result=None, calib=get_current_calibration(),
                                active_page="calibrate")

    files = request.files.getlist("images")
    cols = int(request.form.get("cols", 9))
    rows = int(request.form.get("rows", 6))
    square = float(request.form.get("square", 25.0))

    if len(files) < 3:
        flash("Please upload chessboard photos (10-15+ recommended).")
        return redirect(url_for("hw2.calibrate_view"))

    for f in os.listdir(upload_calib_dir):
        os.remove(os.path.join(upload_calib_dir, f))

    saved_paths = []
    for f in files:
        path = os.path.join(upload_calib_dir, f.filename)
        f.save(path)
        saved_paths.append(path)

    object_points, image_points, image_size = calibration.find_corners(
        saved_paths, (cols, rows), square
    )

    if len(object_points) < 3:
        flash("Chessboard wasn't detected in enough images - check the cols/rows values match your "
              "printed board (inner corners, not squares) and try again.")
        return redirect(url_for("hw2.calibrate_view"))

    camera_matrix, dist_coeffs, mean_error, per_image_error = calibration.calibrate(
        object_points, image_points, image_size
    )

    np.savez(
        calib_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error=mean_error,
        image_size=np.array(image_size),
    )

    result = {
        "n_used": len(object_points),
        "n_uploaded": len(files),
        "image_size": image_size,
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "dist_coeffs": [round(float(c), 5) for c in dist_coeffs.ravel()],
        "mean_error": mean_error,
    }

    sample_images = _calibration_preview(saved_paths, (cols, rows), camera_matrix, dist_coeffs, results_dir)
    sample_token = samples.stage(
        "hw2", "calibrate", "Camera calibration",
        images=sample_images,
        params=[("Photos used", f"{result['n_used']} of {result['n_uploaded']}"),
                ("Board (inner corners)", f"{cols} x {rows}"), ("Square size", f"{square:g} mm"),
                ("Image size", f"{image_size[0]} x {image_size[1]}")],
        metrics=[("fx", f"{result['fx']:.2f}"), ("fy", f"{result['fy']:.2f}"),
                 ("cx", f"{result['cx']:.2f}"), ("cy", f"{result['cy']:.2f}"),
                 ("Mean reprojection error", f"{mean_error:.4f} px")],
        note=f"Distortion coefficients (k1, k2, p1, p2, k3): {result['dist_coeffs']}",
    )
    return render_template("hw2/calibrate.html", result=result, calib=get_current_calibration(),
                            active_page="calibrate", sample_token=sample_token)


def _calibration_preview(image_paths, pattern_size, camera_matrix, dist_coeffs, results_dir):
    """Draws detected corners on the first usable board photo, plus its undistorted version."""
    if not samples.editable():
        return []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue
        corners_img = img.copy()
        cv2.drawChessboardCorners(corners_img, pattern_size, corners, found)
        corners_path = os.path.join(results_dir, "calib_corners.jpg")
        undist_path = os.path.join(results_dir, "calib_undistorted.jpg")
        cv2.imwrite(corners_path, corners_img)
        cv2.imwrite(undist_path, cv2.undistort(img, camera_matrix, dist_coeffs))
        return [(corners_path, "Detected chessboard corners"),
                (undist_path, "Same photo undistorted with the calibrated intrinsics")]
    return []


@bp.route("/measure", methods=["GET", "POST"])
def measure_view():
    _, upload_obj_dir, _, _ = _dirs()
    calib = get_current_calibration()

    if request.method == "GET":
        return render_template("hw2/measure.html", calib=calib, image_url=None, result=None,
                                active_page="measure")

    if calib is None:
        flash("Run calibration before measuring an object.")
        return redirect(url_for("hw2.calibrate_view"))

    if "object_image" in request.files and request.files["object_image"].filename:
        f = request.files["object_image"]
        path = os.path.join(upload_obj_dir, f.filename)
        f.save(path)
        image_url = url_for("static", filename=f"object_uploads/{f.filename}")
        return render_template("hw2/measure.html", calib=calib, image_url=image_url,
                                image_name=f.filename, result=None, active_page="measure")

    image_name = request.form.get("image_name")
    distance_mm = float(request.form.get("distance_mm"))
    x1, y1 = float(request.form["x1"]), float(request.form["y1"])
    x2, y2 = float(request.form["x2"]), float(request.form["y2"])

    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]

    p1_u, p2_u = undistort_points([(x1, y1), (x2, y2)], camera_matrix, dist_coeffs)
    real_mm, dx_mm, dy_mm = pixel_length_to_real(p1_u, p2_u, distance_mm, camera_matrix)
    pixel_dist = float(np.hypot(x2 - x1, y2 - y1))

    result = {
        "pixel_dist": pixel_dist,
        "distance_mm": distance_mm,
        "real_mm": real_mm,
        "real_cm": real_mm / 10.0,
    }

    sample_token = None
    if samples.editable():
        img = cv2.imread(os.path.join(upload_obj_dir, image_name))
        annotated_path = None
        if img is not None:
            thick = max(2, int(max(img.shape[:2]) / 400))
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.line(img, p1, p2, (0, 255, 0), thick)
            cv2.circle(img, p1, thick * 3, (0, 0, 255), -1)
            cv2.circle(img, p2, thick * 3, (0, 0, 255), -1)
            annotated_path = os.path.join(upload_obj_dir, "_measured_annotated.jpg")
            cv2.imwrite(annotated_path, img)
        sample_token = samples.stage(
            "hw2", "measure", "Real-world measurement",
            images=[(annotated_path, "Measured edge (clicked endpoints)")],
            params=[("Camera-to-object distance", f"{distance_mm:g} mm"),
                    ("Endpoints (px)", f"({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")],
            metrics=[("Measured length", f"{real_mm / 10.0:.2f} cm"),
                     ("Measured length (mm)", f"{real_mm:.2f}"),
                     ("Pixel distance", f"{pixel_dist:.1f} px")],
            note="L = (pixel_distance x Z) / f, using the calibrated focal length and the entered distance.",
        )

    image_url = url_for("static", filename=f"object_uploads/{image_name}")
    return render_template("hw2/measure.html", calib=calib, image_url=image_url,
                            image_name=image_name, result=result, active_page="measure",
                            sample_token=sample_token)


@bp.route("/validate", methods=["GET", "POST"])
def validate_view():
    _, _, _, results_dir = _dirs()

    if request.method == "GET":
        return render_template("hw2/validate.html", stats=None, table=None, plot1=None, plot2=None,
                                active_page="validate")

    import pandas as pd

    f = request.files.get("csv_file")
    if not f or not f.filename:
        flash("Choose a CSV file first.")
        return redirect(url_for("hw2.validate_view"))

    df = pd.read_csv(f)
    required_cols = {"object_id", "actual_mm", "measured_mm"}
    if not required_cols.issubset(df.columns):
        flash(f"CSV needs at least these columns: {sorted(required_cols)}")
        return redirect(url_for("hw2.validate_view"))

    df_with_error, stats = vm.compute_stats(df)
    vm.make_plots(df_with_error, results_dir)

    table_html = df_with_error.round(2).to_html(index=False, classes="result-table")
    plot1 = url_for("static", filename="results/error_by_object.png")
    plot2 = url_for("static", filename="results/percent_error.png")

    sample_token = samples.stage(
        "hw2", "validate", "Validation (ground-truth objects)",
        images=[(os.path.join(results_dir, "error_by_object.png"), "Error by object", True),
                (os.path.join(results_dir, "percent_error.png"), "Percent error", True)],
        params=[("CSV", f.filename)],
        metrics=[("Objects", str(stats["n_objects"])),
                 ("Mean abs. error", f"{stats['mean_abs_error_mm']:.2f} mm"),
                 ("RMSE", f"{stats['rmse_mm']:.2f} mm"),
                 ("Std dev of error", f"{stats['std_error_mm']:.2f} mm"),
                 ("Mean % error", f"{stats['mean_pct_error']:.2f}%"),
                 ("Max abs. error", f"{stats['max_abs_error_mm']:.2f} mm")],
        table_html=table_html,
    )
    return render_template("hw2/validate.html", stats=stats, table=table_html, plot1=plot1, plot2=plot2,
                            active_page="validate", sample_token=sample_token)
