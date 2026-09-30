"""Dense optical flow over a video clip (Farneback), visualised as a video, plus the numbers
that back up what we claim the flow tells us (where/when things move, direction, speed,
camera pan / zoom / roll vs. objects moving on their own).
"""
import csv
import math
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import video_io

FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def farneback(prev_gray, gray):
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK)


def flow_to_color(flow, max_mag):
    """Hue = direction, brightness = speed (black = not moving)."""
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    ang = (np.degrees(np.arctan2(-fy, fx)) + 360) % 360  # screen angle, 90 = up
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(mag / max_mag, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def color_wheel(size=70, max_mag=1.0):
    r = size // 2
    ys, xs = np.mgrid[-r:r, -r:r].astype(np.float32)
    flow = np.dstack([xs, ys]) / r * max_mag
    wheel = flow_to_color(flow, max_mag)
    wheel[xs * xs + ys * ys > r * r] = 40
    return wheel


def draw_arrows(frame, flow, step=16, scale=3.0, min_mag=0.5, color=(60, 255, 60)):
    out = frame.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            if dx * dx + dy * dy < min_mag * min_mag:
                continue
            tip = (int(round(x + dx * scale)), int(round(y + dy * scale)))
            cv2.arrowedLine(out, (x, y), tip, color, 1, cv2.LINE_AA, tipLength=0.35)
    return out


def texture_mask(gray, min_grad=2.0):
    """Pixels with enough gradient for the flow there to mean anything (aperture problem)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 3) > min_grad


def fit_global_motion(flow, grid=8, thr=1.0, valid=None):
    """Least-squares affine flow u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y (x, y from image centre).

    This is the part of the flow the camera can explain (pan -> a0,b0, zoom/forward -> divergence,
    roll -> curl). Only textured pixels are used, with one re-fit without the worst residuals so
    a big moving object doesn't drag it.
    """
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[grid // 2:h:grid, grid // 2:w:grid]
    if valid is not None and valid[ys, xs].sum() > 30:
        pick = valid[ys, xs]
        ys, xs = ys[pick], xs[pick]
    u = flow[ys, xs, 0].ravel()
    v = flow[ys, xs, 1].ravel()
    x = (xs.ravel() - w / 2.0) / w
    y = (ys.ravel() - h / 2.0) / w
    A = np.column_stack([np.ones_like(x), x, y])

    keep = np.ones(len(u), bool)
    for _ in range(2):
        pu, *_ = np.linalg.lstsq(A[keep], u[keep], rcond=None)
        pv, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        res = np.hypot(u - A @ pu, v - A @ pv)
        keep = res < max(thr, 2.5 * np.median(res))

    # x, y were divided by w, so derivatives come out per image-width; convert back to per-pixel
    a0, a1, a2 = pu[0], pu[1] / w, pu[2] / w
    b0, b1, b2 = pv[0], pv[1] / w, pv[2] / w
    return {"tx": a0, "ty": b0, "div": a1 + b2, "curl": b1 - a2, "params": (a0, a1, a2, b0, b1, b2)}


def global_flow_field(shape, params):
    h, w = shape
    a0, a1, a2, b0, b1, b2 = params
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xs - w / 2.0, ys - h / 2.0
    return np.dstack([a0 + a1 * x + a2 * y, b0 + b1 * x + b2 * y]).astype(np.float32)


def _estimate_color_scale(path, start_s, duration_s, max_dim, samples=12):
    """Rough 99th-percentile speed from a few frame pairs, so the colour scale doesn't flicker."""
    cap = cv2.VideoCapture(path)
    vals = []
    for i in range(samples):
        t = start_s + duration_s * (i + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok1, a = cap.read()
        ok2, b = cap.read()
        if not (ok1 and ok2):
            continue
        a = cv2.cvtColor(video_io.resize_to(a, max_dim), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(video_io.resize_to(b, max_dim), cv2.COLOR_BGR2GRAY)
        mag = np.linalg.norm(farneback(a, b), axis=2)
        vals.append(np.percentile(mag, 99))
    cap.release()
    if not vals:
        return 4.0
    return float(max(1.0, np.median(vals) * 1.3))


def _label(img, text, pos=(8, 20)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def run(video_path, out_dir, run_id, start_s=0.0, duration_s=30.0, max_dim=480, motion_thr=1.0, step=1):
    t0 = time.time()
    info = video_io.clip_info(video_path)
    fps = info["fps"]
    duration_s = min(duration_s, max(0.0, info["duration"] - start_s)) or duration_s
    color_max = _estimate_color_scale(video_path, start_s, duration_s, max_dim) * step

    stats = []
    writer = None
    prev_gray = None
    heat = heat_obj = None
    rose = np.zeros(36)
    keyframes = {}
    best_peak = -1.0
    n_expected = max(2, int(duration_s * fps / step))
    snapshot_at = {int(n_expected * f) for f in (1 / 6, 1 / 2, 5 / 6)}
    mid_frame = None

    # with step > 1 the flow is between frames `step` apart, so speeds are per `step` frames
    for k, t, frame in video_io.iter_frames(video_path, start_s, duration_s, max_dim, step):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            h, w = gray.shape
            heat = np.zeros((h, w), np.float64)
            heat_obj = np.zeros((h, w), np.float64)
            wheel = color_wheel(70, color_max)
            writer = video_io.VideoWriter(os.path.join(out_dir, f"{run_id}_flow"), fps / step, (2 * w, h))
            continue

        flow = farneback(prev_gray, gray)
        prev_gray = gray
        mag = np.linalg.norm(flow, axis=2)
        g = fit_global_motion(flow, thr=motion_thr, valid=texture_mask(gray))
        residual = np.linalg.norm(flow - global_flow_field(flow.shape[:2], g["params"]), axis=2)

        moving = mag > motion_thr
        obj_moving = residual > motion_thr
        heat += mag
        heat_obj += residual
        if moving.any():
            ang = (np.degrees(np.arctan2(-flow[..., 1], flow[..., 0])) + 360) % 360
            rose += np.histogram(ang[moving], bins=36, range=(0, 360), weights=mag[moving])[0]

        stats.append({
            "t": t, "mean_mag": float(mag.mean()), "p95_mag": float(np.percentile(mag, 95)),
            "moving_frac": float(moving.mean()), "object_frac": float(obj_moving.mean()),
            "tx": float(g["tx"]), "ty": float(g["ty"]), "div": float(g["div"]), "curl": float(g["curl"]),
        })

        arrows = draw_arrows(frame, flow, min_mag=motion_thr * 0.5)
        colored = flow_to_color(flow, color_max)
        colored[-wheel.shape[0] - 6:-6, -wheel.shape[1] - 6:-6] = wheel
        _label(arrows, f"t = {t:5.2f} s   mean |v| = {mag.mean():.2f} px/frame")
        _label(colored, f"moving: {100 * moving.mean():4.1f}%   pan ({g['tx']:+.1f}, {g['ty']:+.1f})")
        writer.write(np.hstack([arrows, colored]))

        # evidence frames: the busiest frame, plus three spread through the clip
        mask_vis = frame.copy()
        mask_vis[obj_moving] = (0.45 * mask_vis[obj_moving] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        panel = {"arrows": arrows, "color": colored, "mask": mask_vis}
        score = stats[-1]["mean_mag"]
        if score > best_peak:
            best_peak = score
            keyframes["peak"] = dict(panel, t=t)
        if k in snapshot_at:
            keyframes[f"snap{len([s for s in keyframes if s.startswith('snap')])}"] = dict(panel, t=t)
        if k == n_expected // 2 or mid_frame is None:
            mid_frame = frame

    if writer is None or not stats:
        return None
    writer.close()

    n = len(stats)
    heat /= n
    heat_obj /= n
    files = {"video": os.path.basename(writer.path)}
    files.update(_save_keyframes(keyframes, out_dir, run_id))
    files["heatmap"] = _save_heat(mid_frame, heat, out_dir, f"{run_id}_heat_all.png")
    files["heatmap_obj"] = _save_heat(mid_frame, heat_obj, out_dir, f"{run_id}_heat_obj.png")
    files["timeline"] = _plot_timeline(stats, keyframes.get("peak", {}).get("t"), motion_thr, out_dir, run_id)
    files["rose"] = _plot_rose(rose, out_dir, run_id)
    files["csv"] = _write_csv(stats, out_dir, run_id)

    frame_h, frame_w = heat.shape
    summary = summarize(stats, fps / step, motion_thr, rose)
    summary.update({
        "fps": fps, "frames": n + 1, "proc_w": frame_w, "proc_h": frame_h,
        "orig_w": info["width"], "orig_h": info["height"], "start": start_s, "duration": duration_s,
        "color_max": color_max, "elapsed": time.time() - t0, "motion_thr": motion_thr, "step": step,
    })
    return {"files": files, "summary": summary}


def _save_keyframes(keyframes, out_dir, run_id):
    saved = {}
    for name, kf in keyframes.items():
        for part in ("arrows", "color", "mask"):
            fname = f"{run_id}_{name}_{part}.png"
            cv2.imwrite(os.path.join(out_dir, fname), kf[part])
            saved[f"{name}_{part}"] = fname
        saved[f"{name}_t"] = kf["t"]
    return saved


def _save_heat(frame, heat, out_dir, fname):
    norm = heat / (np.percentile(heat, 99.5) + 1e-9)
    colored = cv2.applyColorMap((np.clip(norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    out = cv2.addWeighted(frame, 0.35, colored, 0.65, 0)
    cv2.imwrite(os.path.join(out_dir, fname), out)
    return fname


def _plot_timeline(stats, peak_t, thr, out_dir, run_id):
    t = np.array([s["t"] for s in stats])
    get = lambda k: np.array([s[k] for s in stats])
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, get("mean_mag"), label="mean |v|", color="#c9762f")
    axes[0].plot(t, get("p95_mag"), label="95th percentile |v|", color="#555", lw=0.8)
    axes[0].set_ylabel("px / frame")
    axes[0].set_title("How much is moving")

    axes[1].plot(t, 100 * get("moving_frac"), label=f"|v| > {thr:g} px (all motion)", color="#1f77b4")
    axes[1].plot(t, 100 * get("object_frac"), label="left after removing camera motion", color="#d62728")
    axes[1].set_ylabel("% of pixels")
    axes[1].set_title("Where it is moving")

    axes[2].plot(t, get("tx"), label="tx (+ = content moves right)", color="#2ca02c")
    axes[2].plot(t, get("ty"), label="ty (+ = content moves down)", color="#9467bd")
    axes[2].axhline(0, color="#999", lw=0.6)
    axes[2].set_ylabel("px / frame")
    axes[2].set_title("Global translation (camera pan / tilt)")

    axes[3].plot(t, get("div"), label="divergence (+ = expanding / approaching)", color="#ff7f0e")
    axes[3].plot(t, get("curl"), label="curl (+ = clockwise on screen)", color="#17becf")
    axes[3].axhline(0, color="#999", lw=0.6)
    axes[3].set_ylabel("1 / frame")
    axes[3].set_xlabel("time in video (s)")
    axes[3].set_title("Zoom / forward motion and roll")

    for ax in axes:
        if peak_t is not None:
            ax.axvline(peak_t, color="#c9762f", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fname = f"{run_id}_timeline.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=110)
    plt.close(fig)
    return fname


def _plot_rose(rose, out_dir, run_id):
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_subplot(projection="polar")
    theta = np.radians(np.arange(36) * 10 + 5)
    total = rose.sum() or 1.0
    ax.bar(theta, rose / total * 100, width=np.radians(10), color="#c9762f", edgecolor="#7a4415")
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["right", "up", "left", "down"])
    ax.set_title("Direction of motion on screen\n(% of total flow magnitude)", fontsize=10)
    fig.tight_layout()
    fname = f"{run_id}_rose.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=110)
    plt.close(fig)
    return fname


def _write_csv(stats, out_dir, run_id):
    fname = f"{run_id}_flow_stats.csv"
    with open(os.path.join(out_dir, fname), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
        wr.writeheader()
        for s in stats:
            wr.writerow({k: round(v, 5) for k, v in s.items()})
    return fname


def _direction_word(deg):
    names = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return names[int(((deg % 360) + 22.5) // 45) % 8]


def summarize(stats, fps, thr, rose):
    """Turns the per-frame numbers into plain-language observations (each one cites its number)."""
    get = lambda k: np.array([s[k] for s in stats])
    t, mm, mv, obj = get("t"), get("mean_mag"), get("moving_frac"), get("object_frac")
    tx, ty, div, curl = get("tx"), get("ty"), get("div"), get("curl")
    obs = []

    i = int(np.argmax(mm))
    obs.append(f"Strongest motion at t = {t[i]:.2f} s: mean speed {mm[i]:.2f} px/frame "
               f"({mm[i] * fps:.0f} px/s), {100 * mv[i]:.0f}% of pixels above {thr:g} px/frame.")

    quiet = np.mean(mv < 0.02)
    if quiet < 0.01:
        obs.append("Every frame contains motion (more than 2% of pixels moving in each one).")
    else:
        obs.append(f"{100 * (1 - quiet):.0f}% of the frames contain motion (more than 2% of pixels moving); "
                   f"the other {100 * quiet:.0f}% are essentially still.")

    if rose.sum() > 0:
        peak_dir = (np.argmax(rose) * 10 + 5)
        share = rose.max() / rose.sum()
        obs.append(f"Dominant direction of motion on screen: {_direction_word(peak_dir)} (~{peak_dir:.0f} deg, "
                   f"{100 * share:.0f}% of all flow magnitude falls in that 10-degree bin).")

    pan = np.hypot(tx, ty)
    panning = pan > 0.3
    if panning.mean() < 0.15:
        obs.append(f"The global (camera) translation is tiny - median {np.median(pan):.2f} px/frame, above "
                   f"0.3 px/frame in only {100 * panning.mean():.0f}% of frames - so the camera is basically still "
                   f"and the flow comes from objects moving in the scene (on average {100 * obj.mean():.1f}% of "
                   f"pixels move independently).")
    else:
        mx, my = tx[panning].mean(), ty[panning].mean()
        cam_dir = _direction_word(math.degrees(math.atan2(my, -mx)))  # camera turns opposite to the content
        obs.append(f"In {100 * panning.mean():.0f}% of the frames (between t = {t[panning].min():.1f} s and "
                   f"{t[panning].max():.1f} s) the whole field moves together - average global translation "
                   f"({mx:+.2f}, {my:+.2f}) px/frame, content drifting "
                   f"{_direction_word(math.degrees(math.atan2(-my, mx)))}. That is camera motion: the camera is "
                   f"panning/tilting {cam_dir}. After removing it, {100 * obj.mean():.1f}% of pixels still move on "
                   f"their own (independent objects, or near objects showing parallax).")

    if np.abs(np.median(div)) > 0.003 or np.percentile(np.abs(div), 90) > 0.008:
        j = int(np.argmax(np.abs(div)))
        kind = "expanding (camera moving forward or something approaching)" if div[j] > 0 else \
            "contracting (camera moving backward or something receding)"
        ttc = 2.0 / div[j] / fps if div[j] > 0 else None
        line = f"Largest divergence {div[j]:+.4f}/frame at t = {t[j]:.2f} s: the field is {kind}."
        if ttc:
            line += f" For a surface facing the camera that means time-to-contact of about 2/div = {ttc:.1f} s."
        obs.append(line)
    else:
        obs.append(f"Divergence stays near zero (median {np.median(div):+.4f}/frame), so there is no sustained "
                   f"zoom or forward/backward camera motion.")

    if np.percentile(np.abs(curl), 90) > 0.004:
        j = int(np.argmax(np.abs(curl)))
        obs.append(f"Noticeable rotation about the viewing axis at t = {t[j]:.2f} s (curl {curl[j]:+.4f}/frame, "
                   f"{'clockwise' if curl[j] > 0 else 'counter-clockwise'} on screen) - camera roll or a turning object.")

    return {
        "observations": obs,
        "peak_t": float(t[i]), "peak_mag": float(mm[i]), "mean_mag": float(mm.mean()),
        "moving_pct": float(100 * mv.mean()), "object_pct": float(100 * obj.mean()),
        "median_pan": float(np.median(pan)), "median_div": float(np.median(div)),
    }
