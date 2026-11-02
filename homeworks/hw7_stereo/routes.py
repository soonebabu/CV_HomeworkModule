"""Homework 7: object size with calibrated stereo, and room dimensions from one video.

The chessboard photos in calibration_images/ give the camera intrinsics, using the same code as HW2.
"""
import glob
import json
import os
import re
import uuid

import cv2
import numpy as np
from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for

from .. import samples
from ..hw2_calibration import calibration
from . import room as rm
from . import stereo as st

bp = Blueprint("hw7", __name__)

BOARD_SIZE = (10, 7)  # inner corners of the board in calibration_images/
SQUARE_MM = 25.0  # only scales the board's pose, it has no effect on the intrinsics
ROOM_MAX_DIM = 960

_RUN_ID = re.compile(r"^[a-f0-9]{10}$")
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}


def _dirs():
    static_dir = current_app.static_folder
    dirs = [os.path.join(static_dir, "hw7", name) for name in ("uploads", "outputs", "calib")]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def _intrinsics_file():
    return os.path.join(_dirs()[2], "camera_intrinsics.npz")


def get_intrinsics():
    path = _intrinsics_file()
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "camera_matrix": data["camera_matrix"],
        "dist_coeffs": data["dist_coeffs"],
        "reprojection_error": float(data["reprojection_error"]),
        "image_size": tuple(data["image_size"]),
        "n_used": int(data["n_used"]),
        "n_photos": int(data["n_photos"]),
    }


@bp.route("/")
def overview():
    return render_template("hw7/overview.html", active_page="overview")


# ---------------- Camera intrinsics ----------------

@bp.route("/intrinsics", methods=["GET", "POST"])
def intrinsics_view():
    if request.method == "POST":
        folder = os.path.join(current_app.root_path, "calibration_images")
        paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(folder, ext)))
        object_points, image_points, image_size = calibration.find_corners(paths, BOARD_SIZE, SQUARE_MM)
        if len(object_points) < 3:
            flash(f"The chessboard was found in only {len(object_points)} of {len(paths)} photos.")
            return redirect(url_for("hw7.intrinsics_view"))
        camera_matrix, dist_coeffs, mean_error, _ = calibration.calibrate(object_points, image_points, image_size)
        np.savez(_intrinsics_file(), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
                 reprojection_error=mean_error, image_size=np.array(image_size),
                 n_used=len(object_points), n_photos=len(paths))
        sample_token = samples.stage(
            "hw7", "intrinsics", "Camera intrinsics",
            images=_intrinsics_preview(paths, camera_matrix, dist_coeffs),
            params=[("Photos used", f"{len(object_points)} of {len(paths)}"),
                    ("Board (inner corners)", f"{BOARD_SIZE[0]} x {BOARD_SIZE[1]}"),
                    ("Image size", f"{image_size[0]} x {image_size[1]}")],
            metrics=[("fx", f"{camera_matrix[0, 0]:.2f}"), ("fy", f"{camera_matrix[1, 1]:.2f}"),
                     ("cx", f"{camera_matrix[0, 2]:.2f}"), ("cy", f"{camera_matrix[1, 2]:.2f}"),
                     ("Mean reprojection error", f"{mean_error:.4f} px")],
        )
        return render_template("hw7/intrinsics.html", calib=get_intrinsics(), active_page="intrinsics",
                               sample_token=sample_token)
    return render_template("hw7/intrinsics.html", calib=get_intrinsics(), active_page="intrinsics")


def _intrinsics_preview(paths, camera_matrix, dist_coeffs):
    """Detected corners on the first usable board photo, plus that photo undistorted."""
    if not samples.editable():
        return []
    output_dir = _dirs()[1]
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        found, corners = cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), BOARD_SIZE)
        if not found:
            continue
        corners_img = img.copy()
        cv2.drawChessboardCorners(corners_img, BOARD_SIZE, corners, found)
        corners_path = os.path.join(output_dir, "intrinsics_corners.jpg")
        undistorted_path = os.path.join(output_dir, "intrinsics_undistorted.jpg")
        cv2.imwrite(corners_path, corners_img)
        cv2.imwrite(undistorted_path, cv2.undistort(img, camera_matrix, dist_coeffs))
        return [(corners_path, "Detected chessboard corners"), (undistorted_path, "Same photo undistorted")]
    return []


def _make_rig(pose_left, pose_right, size, baseline_mm):
    """Calibrated camera moved or mounted sideways: pose from the larger photos, scale from the measured
    baseline. The rig itself is for images of the given (working) size."""
    calib = get_intrinsics()
    if calib is None:
        raise st.StereoError("Compute the camera intrinsics first.")
    if baseline_mm <= 0:
        raise st.StereoError("Enter the baseline in mm.")
    calib_size = tuple(int(v) for v in calib["image_size"])
    pose_size = (pose_left.shape[1], pose_left.shape[0])
    K_pose = st.fit_camera_to_image(calib["camera_matrix"], calib_size, pose_size)
    K = st.fit_camera_to_image(calib["camera_matrix"], calib_size, size)
    dist = calib["dist_coeffs"]
    R, t, _ = st.pose_from_images(pose_left, pose_right, K_pose, dist, K_pose, dist)
    return st.make_rig(K, dist, K, dist, R, t * baseline_mm)


# ---------------- Object size from a calibrated pair ----------------

def _prepare_pair(left, right, rig):
    """Rectifies a pair and stores what the click step needs. Returns the run id."""
    upload_dir = _dirs()[0]
    if rig["swapped"]:
        left, right = right, left
    rect = st.rectify_maps(rig, (left.shape[1], left.shape[0]), alpha=1)
    rect_l, rect_r = st.apply_rectify(left, right, rect)
    disp = st.compute_disparity(rect_l, rect_r, search=st.disparity_range(rect_l, rect_r, plain_scene=True))

    run_id = uuid.uuid4().hex[:10]
    cv2.imwrite(os.path.join(upload_dir, f"stereo_{run_id}_left.jpg"), rect_l, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(upload_dir, f"stereo_{run_id}_right.jpg"), rect_r, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.savez_compressed(os.path.join(upload_dir, f"stereo_{run_id}.npz"), disp=disp, f=rect["f"], cx=rect["cx"],
                        cy=rect["cy"], baseline=rect["baseline"])
    return run_id


def _stereo_run(run_id):
    if not run_id or not _RUN_ID.match(run_id):
        abort(400)
    path = os.path.join(_dirs()[0], f"stereo_{run_id}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {"disp": data["disp"], "f": float(data["f"]), "cx": float(data["cx"]), "cy": float(data["cy"]),
            "baseline": float(data["baseline"])}


def _stereo_page(run_id, run, **extra):
    urls = {name: url_for("static", filename=f"hw7/uploads/stereo_{run_id}_{name}.jpg") for name in ("left", "right")}
    return render_template("hw7/stereo.html", active_page="stereo", run_id=run_id, run=run, urls=urls, **extra)


def _annotate(left_path, kind, pts, edges, out_path):
    img = cv2.imread(left_path)
    thick = max(2, int(max(img.shape[:2]) / 500))
    outline = np.round(pts).astype(np.int32)
    if kind != "circle":
        cv2.polylines(img, [outline], True, (0, 255, 0), thick)
    for i, p in enumerate(outline):
        cv2.circle(img, tuple(int(v) for v in p), thick * 2, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    for i, length in enumerate(edges):
        mid = ((pts[i] + pts[(i + 1) % len(pts)]) / 2).astype(np.int32)
        cv2.putText(img, f"{length:.0f} mm", (int(mid[0]) + 4, int(mid[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(out_path, img)


def _stereo_upload():
    left_file, right_file = request.files.get("left"), request.files.get("right")
    if not (left_file and left_file.filename and right_file and right_file.filename):
        raise st.StereoError("Choose both photos.")
    decode = lambda f: cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    left, right = decode(left_file), decode(right_file)
    if left is None or right is None:
        raise st.StereoError("One of those files could not be read as an image.")
    if left.shape != right.shape:
        raise st.StereoError("The two photos must have the same resolution and orientation.")
    pose_left, pose_right = st.resize_max(left, st.POSE_DIM), st.resize_max(right, st.POSE_DIM)
    left, right = st.resize_max(left), st.resize_max(right)
    rig = _make_rig(pose_left, pose_right, (left.shape[1], left.shape[0]),
                    request.form.get("baseline_mm", type=float) or 0.0)
    run_id = _prepare_pair(left, right, rig)
    run = _stereo_run(run_id)

    rect_left, rect_right = (cv2.imread(os.path.join(_dirs()[0], f"stereo_{run_id}_{name}.jpg")) for name in ("left", "right"))
    found, found_right = st.detect_object(rect_left), st.detect_object(rect_right)
    if found is None:
        flash("No object could be picked out automatically. Click its points on the left image.")
        return _stereo_page(run_id, run)
    kind, points, contour = found
    partners = st.match_outline(points, contour, found_right[2]) if found_right else None
    try:
        return _measure_points(run_id, run, kind, points.tolist(), [] if partners is None else partners.tolist())
    except st.StereoError as err:
        flash(str(err))
        return _stereo_page(run_id, run)


def _stereo_measure():
    run_id = request.form.get("run_id")
    run = _stereo_run(run_id)
    if run is None:
        raise st.StereoError("That stereo pair has expired. Upload it again.")
    kind = request.form.get("kind")
    if kind not in ("rectangle", "circle", "polygon"):
        abort(400)
    try:
        points = json.loads(request.form.get("points") or "[]")
        right_points = json.loads(request.form.get("right_points") or "[]")
    except ValueError:
        abort(400)
    if not points:
        raise st.StereoError("Click the object's points on the left image first.")
    if right_points and len(right_points) != len(points):
        raise st.StereoError("Click the same number of points on both images, or none on the right one.")
    return _measure_points(run_id, run, kind, points, right_points)


def _measure_points(run_id, run, kind, points, right_points):
    upload_dir = _dirs()[0]
    left_path = os.path.join(upload_dir, f"stereo_{run_id}_left.jpg")
    left, right = cv2.imread(left_path), cv2.imread(os.path.join(upload_dir, f"stereo_{run_id}_right.jpg"))
    pts = np.array(points, dtype=np.float64)
    manual = np.array(right_points, dtype=np.float64) if right_points else None

    disparities = st.point_disparities(pts, left, right, run["disp"], manual)
    xyz = st.to_3d(pts, disparities, run)
    headline, edges = st.measure_shape(kind, xyz, pts)
    rows = [{"n": i + 1, "d": float(disparities[i]), "X": xyz[i, 0], "Y": xyz[i, 1], "Z": xyz[i, 2]}
            for i in range(len(pts))]

    annotated = os.path.join(upload_dir, f"stereo_{run_id}_result.jpg")
    _annotate(left_path, kind, pts, edges, annotated)
    result = {"headline": [("Distance Z", float(xyz[:, 2].mean()))] + headline, "edges": edges, "rows": rows,
              "image_url": url_for("static", filename=f"hw7/uploads/stereo_{run_id}_result.jpg")}

    metrics = [(label, f"{value:.1f} mm") for label, value in headline]
    metrics += [(f"Edge {i + 1}", f"{e:.1f} mm") for i, e in enumerate(edges)]
    metrics.append(("Mean distance Z", f"{xyz[:, 2].mean():.1f} mm"))
    sample_token = samples.stage(
        "hw7", "stereo", f"Calibrated stereo measurement ({kind})",
        images=[(annotated, "Measured outline")],
        params=[("Focal length f", f"{run['f']:.1f} px"), ("Baseline B", f"{run['baseline']:.1f} mm")],
        metrics=metrics,
    )
    return _stereo_page(run_id, run, result=result, sample_token=sample_token, form_kind=kind, form_points=points)


@bp.route("/stereo", methods=["GET", "POST"])
def stereo_view():
    if request.method == "GET":
        return render_template("hw7/stereo.html", active_page="stereo", run=None)
    try:
        if request.form.get("action") == "measure":
            return _stereo_measure()
        return _stereo_upload()
    except st.StereoError as err:
        flash(str(err))
    return redirect(url_for("hw7.stereo_view"))


# ---------------- Room dimensions from one video ----------------

def _measure_room(path, run_id):
    calib = get_intrinsics()
    if calib is None:
        raise rm.RoomError("Compute the camera intrinsics first.")
    slide_mm = request.form.get("slide_mm", type=float) or 0.0
    slide_end_s = request.form.get("slide_end_s", type=float) or 0.0
    wall_mm = request.form.get("wall_mm", type=float) or 0.0
    if slide_mm <= 0 or slide_end_s <= 0 or wall_mm <= 0:
        raise rm.RoomError("Enter the slide distance, the time it ends and the distance to the wall.")
    n_frames = min(300, max(10, request.form.get("frames", 150, type=int)))

    frames = rm.read_frames(path, n_frames, ROOM_MAX_DIM)
    h, w = frames[0][1].shape[:2]
    K = st.fit_camera_to_image(calib["camera_matrix"], tuple(int(v) for v in calib["image_size"]), (w, h))
    frames = [(t, cv2.undistort(img, K, calib["dist_coeffs"])) for t, img in frames]
    show = lambda v: f"{v / 1000:.2f} m" if v else "not measured"
    try:
        tracker, cloud = rm.run(frames, K, slide_mm, slide_end_s)
        fit = rm.fit_room(cloud, tracker)
        plan = os.path.join(_dirs()[1], f"room_{run_id}_plan.png")
        rm.plot_floor_plan(cloud, tracker, fit, plan)
        stats = [("Length", show(fit["a"]["size"])), ("Width", show(fit["b"]["size"])),
                 ("Height", show(fit["height"]["size"]))]
        title, caption, note = "Room dimensions from one video", "Floor plan from the 3D points", None
    except (rm.RoomError, st.StereoError, cv2.error):
        plan = os.path.join(_dirs()[1], f"room_{run_id}_wall.jpg")
        wall = rm.measure_wall(frames, K, wall_mm, plan)
        stats = [("Wall width" if wall["complete"] else "Wall width (at least)", show(wall["width"])),
                 ("Height", show(wall["height"])), ("Distance to wall", show(wall_mm))]
        title, caption = "Wall dimensions from one video", "First frame, last frame, and the ceiling and floor lines"
        note = "Rough estimate of one wall, scaled by the distance entered. The opposite wall was not measured."

    result = {"stats": stats, "note": note, "plan_url": url_for("static", filename=f"hw7/outputs/{os.path.basename(plan)}")}
    sample_token = samples.stage(
        "hw7", "room", title, images=[(plan, caption)],
        params=[("Slide", f"{slide_mm:g} mm in {slide_end_s:g} s"), ("Distance to wall", f"{wall_mm:g} mm"),
                ("Frames", str(len(frames)))],
        metrics=stats,
    )
    return render_template("hw7/room.html", active_page="room", result=result, sample_token=sample_token)


@bp.route("/room", methods=["GET", "POST"])
def room_view():
    if request.method == "GET":
        return render_template("hw7/room.html", active_page="room", result=None)
    f = request.files.get("video")
    ext = os.path.splitext(f.filename)[1].lower() if f and f.filename else ""
    if ext not in _VIDEO_EXTS:
        flash("Choose a video file.")
        return redirect(url_for("hw7.room_view"))
    run_id = uuid.uuid4().hex[:10]
    path = os.path.join(_dirs()[0], f"room_{run_id}{ext}")
    f.save(path)
    try:
        return _measure_room(path, run_id)
    except (rm.RoomError, st.StereoError) as err:
        flash(str(err))
    finally:
        os.remove(path)
    return redirect(url_for("hw7.room_view"))
