import glob
import json
import os
import re
import uuid

import cv2
import numpy as np
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from .. import samples
from . import lucas_kanade as lk
from . import optical_flow as of
from . import sfm_planar as sfm
from . import video_io

bp = Blueprint("hw6", __name__)

_RUN_ID = re.compile(r"^[a-f0-9]{10}$")
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}


def _dirs():
    static_dir = current_app.static_folder
    upload_dir = os.path.join(static_dir, "hw6", "uploads")
    output_dir = os.path.join(static_dir, "hw6", "outputs")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return upload_dir, output_dir


def _out_url(fname):
    return url_for("static", filename=f"hw6/outputs/{fname}")


def _float(name, default):
    try:
        return float(request.form.get(name, default))
    except (TypeError, ValueError):
        return default


def _save_video(upload_dir, f):
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in _VIDEO_EXTS:
        return None, None
    run_id = uuid.uuid4().hex[:10]
    path = os.path.join(upload_dir, f"{run_id}_video{ext}")
    f.save(path)
    return run_id, path


def _find_video(upload_dir, run_id):
    if not run_id or not _RUN_ID.match(run_id):
        return None
    hits = glob.glob(os.path.join(upload_dir, f"{run_id}_video.*"))
    return hits[0] if hits else None


@bp.route("/")
def overview():
    return render_template("hw6/overview.html", active_page="overview")


@bp.route("/theory")
def theory():
    return render_template("hw6/theory.html", active_page="theory")


# ---------------- Part 1: optical flow video ----------------

@bp.route("/flow", methods=["GET", "POST"])
def flow_view():
    upload_dir, output_dir = _dirs()
    if request.method == "GET":
        return render_template("hw6/flow.html", active_page="flow", result=None)

    f = request.files.get("video")
    if not f or not f.filename:
        flash("Choose a video first.")
        return redirect(url_for("hw6.flow_view"))
    run_id, path = _save_video(upload_dir, f)
    if path is None or video_io.clip_info(path) is None:
        flash("Couldn't read that video - MP4 or MOV straight off the phone works best.")
        return redirect(url_for("hw6.flow_view"))

    label = request.form.get("label", "").strip() or f.filename
    start = max(0.0, _float("start", 0.0))
    duration = min(60.0, max(1.0, _float("duration", 30.0)))
    max_dim = int(_float("max_dim", 480))
    step = max(1, int(_float("step", 1)))
    thr = max(0.1, _float("threshold", 1.0))

    res = of.run(path, output_dir, run_id, start, duration, max_dim, thr, step)
    if res is None:
        flash("No frames were read from that part of the video - check the start time.")
        return redirect(url_for("hw6.flow_view"))

    files, summ = res["files"], res["summary"]
    urls = {k: _out_url(v) for k, v in files.items() if isinstance(v, str)}
    snaps = [{"t": files[f"snap{i}_t"], "arrows": urls[f"snap{i}_arrows"], "color": urls[f"snap{i}_color"],
              "mask": urls[f"snap{i}_mask"]} for i in range(3) if f"snap{i}_t" in files]
    result = {"run_id": run_id, "label": label, "urls": urls, "s": summ, "snaps": snaps,
              "peak_t": files.get("peak_t", 0.0)}

    p = lambda name: os.path.join(output_dir, files[name])
    obs_html = "<ul>" + "".join(f"<li>{o}</li>" for o in summ["observations"]) + "</ul>"
    sample_token = samples.stage(
        "hw6", "flow", f"Optical flow - {label}",
        images=[(p("video"), "Flow video: arrows on the frame (left), colour-coded flow (right)", True),
                (p("timeline"), "Motion statistics over the clip (dashed line = busiest frame)", True),
                (p("peak_arrows"), f"Busiest frame (t = {result['peak_t']:.2f} s) with flow vectors"),
                (p("peak_color"), "Same frame, colour-coded flow (hue = direction, brightness = speed)"),
                (p("peak_mask"), "Pixels moving independently of the camera (red)"),
                (p("heatmap"), "Average flow magnitude per pixel over the clip"),
                (p("heatmap_obj"), "Average object motion (camera motion removed)"),
                (p("rose"), "Direction histogram")],
        metrics=[("Mean speed", f"{summ['mean_mag']:.2f} px/frame"),
                 ("Peak speed", f"{summ['peak_mag']:.2f} px/frame"),
                 ("Pixels moving (avg)", f"{summ['moving_pct']:.1f}%"),
                 ("Independent motion (avg)", f"{summ['object_pct']:.1f}%")],
        params=[("Video", label), ("Clip", f"{summ['start']:.1f} s + {summ['duration']:.1f} s"),
                ("Frame rate", f"{summ['fps']:.1f} fps"),
                ("Processed at", f"{summ['proc_w']} x {summ['proc_h']} (from {summ['orig_w']} x {summ['orig_h']})")],
        table_html=obs_html, table_title="What the flow tells us",
    )
    return render_template("hw6/flow.html", active_page="flow", result=result, sample_token=sample_token)


# ---------------- Part 2: two-frame tracking validation ----------------

def _parse_manual(raw, limit=10):
    pairs = []
    try:
        for item in json.loads(raw or "[]")[:limit]:
            pairs.append(tuple(float(v) for v in item[:4]))
    except (ValueError, TypeError):
        pass
    return pairs


@bp.route("/tracking", methods=["GET", "POST"])
def tracking_view():
    upload_dir, output_dir = _dirs()
    if request.method == "GET":
        return render_template("hw6/tracking.html", active_page="tracking", result=None,
                               run_id=request.args.get("run_id", ""), t=request.args.get("t", ""))

    run_id = request.form.get("run_id", "").strip()
    frames_ready = False
    if run_id and _RUN_ID.match(run_id) and request.form.get("reuse_frames") == "1":
        p1 = os.path.join(output_dir, f"{run_id}_trk_f1.png")
        p2 = os.path.join(output_dir, f"{run_id}_trk_f2.png")
        if os.path.exists(p1) and os.path.exists(p2):
            f1, f2 = cv2.imread(p1), cv2.imread(p2)
            frames_ready = f1 is not None and f2 is not None

    if frames_ready:
        meta = json.loads(request.form.get("meta", "{}"))
    else:
        f = request.files.get("video")
        if f and f.filename:
            run_id, path = _save_video(upload_dir, f)
        else:
            path = _find_video(upload_dir, run_id)
        if not path:
            flash("Upload a video (or open this page from a finished optical-flow run).")
            return redirect(url_for("hw6.tracking_view"))
        pair = video_io.read_frame_pair(path, max(0.0, _float("time", 1.0)))
        if pair is None:
            flash("Couldn't read two frames at that time - try an earlier time.")
            return redirect(url_for("hw6.tracking_view", run_id=run_id))
        f1, f2, n, fps = pair
        cv2.imwrite(os.path.join(output_dir, f"{run_id}_trk_f1.png"), f1)
        cv2.imwrite(os.path.join(output_dir, f"{run_id}_trk_f2.png"), f2)
        label = request.form.get("label", "").strip() or (f.filename if f and f.filename else "video")
        meta = {"frame": n, "fps": fps, "t": n / fps, "label": label}

    manual = _parse_manual(request.form.get("manual_points"))
    res = lk.validate(f1, f2, n_features=int(_float("n_features", 40)), manual_pairs=manual)

    a, b = lk.draw_tracks(f1, f2, res["rows"], res["manual"])
    names = {"tracks1": a, "tracks2": b, "diff": lk.difference_image(res["gray1"], res["gray2"])}
    zoom = lk.zoom_grid(res["gray1"], res["gray2"], res["rows"])
    if zoom is not None:
        names["zoom"] = zoom
    for name, img in names.items():
        cv2.imwrite(os.path.join(output_dir, f"{run_id}_trk_{name}.png"), img)
    has_plot = lk.plot_agreement(res["rows"], os.path.join(output_dir, f"{run_id}_trk_agree.png"))

    urls = {name: _out_url(f"{run_id}_trk_{name}.png") for name in list(names) + ["f1", "f2"]}
    if has_plot:
        urls["agree"] = _out_url(f"{run_id}_trk_agree.png")

    result = {"run_id": run_id, "meta": meta, "rows": res["rows"], "manual": res["manual"],
              "worked": res["worked"], "s": res["summary"], "urls": urls,
              "img_w": f1.shape[1], "img_h": f1.shape[0], "manual_json": json.dumps(manual)}

    sm = res["summary"]
    tables = render_template("hw6/_tracking_tables.html", result=result, lk=lk)
    p = lambda name: os.path.join(output_dir, f"{run_id}_trk_{name}.png")
    images = [(p("tracks1"), "Frame t: features and their Lucas-Kanade displacement (arrows x4)"),
              (p("tracks2"), "Frame t+1: green circle = LK prediction, red cross = actual (NCC)"),
              (p("diff"), "|I(t+1) - I(t)|: where the two frames differ")]
    if zoom is not None:
        images.append((p("zoom"), "Zoomed pixels (10x): blue = start, green = predicted, red = actual", True))
    if has_plot:
        images.append((p("agree"), "Predicted vs. actual displacement for every reliable feature", True))
    sample_token = samples.stage(
        "hw6", "tracking", f"Two-frame tracking - {meta.get('label', 'video')}",
        images=images,
        metrics=[("Median error vs. actual", f"{sm['median_err']:.3f} px"),
                 ("Within 0.5 px", f"{sm['pct_half']:.0f}%"),
                 ("Within 1 px", f"{sm['pct_one']:.0f}%"),
                 ("Our LK vs. OpenCV LK", f"{sm['median_err_cv']:.4f} px")],
        params=[("Frames", f"{meta['frame']} and {meta['frame'] + 1} (t = {meta['t']:.2f} s)"),
                ("Features", f"{sm['n_reliable']} reliable of {sm['n']}"),
                ("Window / levels", f"{2 * lk.HALF_WIN + 1} x {2 * lk.HALF_WIN + 1}, {lk.LEVELS} levels")],
        table_html=tables, table_title="Worked example and per-feature validation", math=True,
    )
    result["tables"] = tables
    return render_template("hw6/tracking.html", active_page="tracking", result=result, sample_token=sample_token,
                           run_id=run_id, t=meta.get("t", ""), meta_json=json.dumps(meta))


# ---------------- Part 3: structure from motion ----------------

def _camera_defaults():
    """Prefer a calibration made on the HW2 page, fall back to the numbers baked into sfm_planar."""
    calib_file = os.path.join(current_app.static_folder, "calib", "camera_calib.npz")
    if os.path.exists(calib_file):
        data = np.load(calib_file)
        K = data["camera_matrix"]
        w, h = (int(v) for v in data["image_size"])
        return {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "width": w, "height": h,
                "dist": [float(v) for v in data["dist_coeffs"].ravel()[:5]],
                "source": "HW2 calibration page (static/calib/camera_calib.npz)"}
    return dict(sfm.DEFAULT_CAMERA)


def _exif(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
    except Exception:
        return {}
    info = {}
    for key, tag, src in (("Make", 0x010F, ex), ("Model", 0x0110, ex), ("Taken", 0x0132, ex),
                          ("Focal length (mm)", 0x920A, sub), ("35 mm equivalent (mm)", 0xA405, sub),
                          ("f-number", 0x829D, sub), ("Exposure (s)", 0x829A, sub), ("ISO", 0x8827, sub)):
        val = src.get(tag)
        if val is not None:
            info[key] = f"{float(val):.4g}" if not isinstance(val, str) else val.strip("\x00 ")
    return info


def _sfm_dir(upload_dir, run_id):
    return os.path.join(upload_dir, f"sfm_{run_id}")


@bp.route("/sfm", methods=["GET", "POST"])
def sfm_view():
    upload_dir, output_dir = _dirs()
    if request.method == "GET":
        return render_template("hw6/sfm.html", active_page="sfm", stage="upload", cam=_camera_defaults())

    if request.form.get("stage") == "upload":
        files = [f for f in request.files.getlist("images") if f and f.filename]
        if len(files) != 4:
            flash("Pick exactly four photos, one per viewpoint.")
            return redirect(url_for("hw6.sfm_view"))
        run_id = uuid.uuid4().hex[:10]
        d = _sfm_dir(upload_dir, run_id)
        os.makedirs(d, exist_ok=True)
        views = []
        for i, f in enumerate(sorted(files, key=lambda f: f.filename)):
            raw = os.path.join(d, f"raw{i + 1}{os.path.splitext(f.filename)[1].lower()}")
            f.save(raw)
            img = cv2.imread(raw)  # applies the EXIF rotation, same as the phone's gallery shows it
            if img is None:
                flash(f"Couldn't read {f.filename}.")
                return redirect(url_for("hw6.sfm_view"))
            h, w = img.shape[:2]
            s = min(1.0, 1600 / max(h, w))
            small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
            cv2.imwrite(os.path.join(d, f"view{i + 1}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            views.append({"name": f.filename, "orig_w": w, "orig_h": h,
                          "w": small.shape[1], "h": small.shape[0], "exif": _exif(raw)})

        cam = {k: _float(k, v) for k, v in _camera_defaults().items() if k in ("fx", "fy", "cx", "cy", "width", "height")}
        try:
            cam["dist"] = [float(v) for v in request.form.get("dist", "").replace(",", " ").split()][:5]
        except ValueError:
            cam["dist"] = []
        cam["dist"] = (cam["dist"] + [0.0] * 5)[:5]
        cam["source"] = request.form.get("cam_source", "entered on the form")
        with open(os.path.join(d, "setup.json"), "w") as fh:
            json.dump({"views": views, "cam": cam, "object": request.form.get("object", "").strip()}, fh, indent=2)
        return redirect(url_for("hw6.sfm_points", run_id=run_id))

    return _sfm_compute(upload_dir, output_dir)


@bp.route("/sfm/<run_id>/points")
def sfm_points(run_id):
    upload_dir, _ = _dirs()
    if not _RUN_ID.match(run_id):
        return redirect(url_for("hw6.sfm_view"))
    d = _sfm_dir(upload_dir, run_id)
    if not os.path.exists(os.path.join(d, "setup.json")):
        flash("That upload has expired - upload the photos again.")
        return redirect(url_for("hw6.sfm_view"))
    with open(os.path.join(d, "setup.json")) as fh:
        setup = json.load(fh)
    urls = [url_for("static", filename=f"hw6/uploads/sfm_{run_id}/view{i + 1}.jpg") for i in range(4)]
    return render_template("hw6/sfm.html", active_page="sfm", stage="points", run_id=run_id, setup=setup,
                           view_urls=urls)


def _sfm_compute(upload_dir, output_dir):
    run_id = request.form.get("run_id", "")
    if not _RUN_ID.match(run_id):
        return redirect(url_for("hw6.sfm_view"))
    d = _sfm_dir(upload_dir, run_id)
    with open(os.path.join(d, "setup.json")) as fh:
        setup = json.load(fh)

    try:
        pts = json.loads(request.form.get("points", "[]"))
        pts = [np.array(p, float).reshape(-1, 2) for p in pts]
    except (ValueError, TypeError):
        pts = []
    if len(pts) != 4 or len({len(p) for p in pts}) != 1 or len(pts[0]) < 4:
        flash("Click the same number of points (at least 4, ideally 6-10) in all four photos.")
        return redirect(url_for("hw6.sfm_points", run_id=run_id))

    n = len(pts[0])
    ki, kj = int(_float("known_a", 1)) - 1, int(_float("known_b", 2)) - 1
    if not (0 <= ki < n and 0 <= kj < n and ki != kj):
        ki, kj = 0, 1
    known_len = _float("known_len", 10.0)
    actual = None
    raw_actual = request.form.get("actual_lengths", "").replace(",", " ").split()
    if raw_actual:
        try:
            actual = [float(v) for v in raw_actual]
        except ValueError:
            actual = None

    v0 = setup["views"][0]
    K, dist, swapped = sfm.scaled_K(setup["cam"], v0["w"], v0["h"])
    try:
        rec = sfm.reconstruct(pts, K, dist, known=(ki, kj, known_len), actual_lengths=actual)
    except (ValueError, np.linalg.LinAlgError) as e:
        flash(f"Reconstruction failed: {e}")
        return redirect(url_for("hw6.sfm_points", run_id=run_id))

    imgs = [cv2.imread(os.path.join(d, f"view{i + 1}.jpg")) for i in range(4)]
    files = {}
    for i, img in enumerate(imgs):
        R, T = rec["poses"][i]
        reproj = cv2.projectPoints(rec["X"], cv2.Rodrigues(R)[0], T, K, dist)[0].reshape(-1, 2)
        fname = f"{run_id}_sfm_view{i + 1}.jpg"
        cv2.imwrite(os.path.join(output_dir, fname), sfm.draw_view(img, rec["obs"][i], reproj, i + 1))
        files[f"view{i + 1}"] = fname
    files["scene"] = f"{run_id}_sfm_scene.png"
    sfm.plot_scene(rec, (v0["w"], v0["h"]), os.path.join(output_dir, files["scene"]))
    files["boundary"] = f"{run_id}_sfm_boundary.png"
    sfm.plot_boundary(rec, os.path.join(output_dir, files["boundary"]))
    files["rectified"] = f"{run_id}_sfm_rectified.jpg"
    cv2.imwrite(os.path.join(output_dir, files["rectified"]), sfm.rectify_view1(imgs[0], rec))

    notes = [request.form.get(f"note{i + 1}", "").strip() for i in range(4)]
    report = _sfm_report(setup, rec, notes, swapped)
    files["report"] = f"{run_id}_sfm_report.json"
    with open(os.path.join(output_dir, files["report"]), "w") as fh:
        json.dump(report, fh, indent=2)

    urls = {k: _out_url(v) for k, v in files.items()}
    workout = render_template("hw6/_sfm_workout.html", rec=rec, setup=setup, notes=notes, swapped=swapped,
                              tex=sfm.tex, texc=sfm.tex_col, np=np)
    p = lambda k: os.path.join(output_dir, files[k])
    err_ba = np.mean([e.mean() for e in rec["err_ba"]])
    obj = setup.get("object") or "planar object"
    sample_token = samples.stage(
        "hw6", "sfm", f"Structure from motion - {obj}",
        images=[(p("scene"), "Recovered object and camera positions", True),
                (p("boundary"), "Recovered boundary in the object plane (cm)"),
                (p("rectified"), "View 1 re-rendered top-down using the recovered plane")]
               + [(p(f"view{i + 1}"), f"View {i + 1}: red = clicked, green = reprojected") for i in range(4)],
        metrics=[("Area", f"{rec['area']:.1f} cm²"), ("Perimeter", f"{rec['perimeter']:.1f} cm"),
                 ("Mean reprojection error", f"{err_ba:.2f} px"),
                 ("Out-of-plane RMS", f"{rec['flat_rms_cm']:.2f} cm")],
        params=[("Object", obj), ("Points per view", str(n)),
                ("Scale from", f"points {ki + 1}-{kj + 1} = {known_len:g} cm"),
                ("fx, fy (px)", f"{K[0, 0]:.1f}, {K[1, 1]:.1f}")],
        table_html=workout, table_title="Mathematical workout", math=True,
    )
    return render_template("hw6/sfm.html", active_page="sfm", stage="result", run_id=run_id, setup=setup,
                           rec=rec, urls=urls, workout=workout, err_ba=err_ba, notes=notes,
                           sample_token=sample_token)


def _sfm_report(setup, rec, notes, swapped):
    """Everything needed to reproduce the reconstruction, as plain JSON (for the submission)."""
    r = lambda a, k=6: np.round(np.asarray(a, float), k).tolist()
    return {
        "object": setup.get("object"),
        "camera": {"K_used": r(rec["K"]), "distortion": r(rec["dist"]), "calibration": setup["cam"],
                   "axes_swapped_for_orientation": bool(swapped)},
        "views": [{"file": v["name"], "original_size": [v["orig_w"], v["orig_h"]], "used_size": [v["w"], v["h"]],
                   "exif": v["exif"], "your_notes": notes[i], "clicked_points_px": r(rec["obs"][i], 2),
                   "R": r(cam["R"]), "t_plane_units": r(cam["T"]),
                   "camera_center_cm_object_frame": r(cam["C_obj"], 2),
                   "distance_to_object_center_cm": round(cam["dist_to_center"], 2),
                   "tilt_from_plane_normal_deg": round(cam["tilt"], 2),
                   "mean_reprojection_error_px": round(float(rec["err_ba"][i].mean()), 3)}
                  for i, (v, cam) in enumerate(zip(setup["views"], rec["cameras"]))],
        "homographies_view1_to_i": [r(h["H_px"]) for h in rec["homogs"]],
        "plane_normal_camera1": r(rec["plane_n"]),
        "scale_cm_per_unit": rec["scale"],
        "points_cm_object_frame": r(rec["uvz"], 3),
        "edges_cm": [{k: e[k] for k in ("a", "b", "length", "actual")} for e in rec["edges"]],
        "area_cm2": rec["area"], "perimeter_cm": rec["perimeter"],
    }
