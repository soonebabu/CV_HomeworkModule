"""Two-frame feature tracking, written out by hand so it follows the derivation on the theory page.

For a feature at u in frame I we look for the displacement d that minimises
    eps(d) = sum_{x in W} (I(x) - J(x + d))^2
Linearising J(x + d) around the current guess gives the 2x2 system G * delta = b with
    G = sum [Ix^2  IxIy; IxIy  Iy^2],   b = sum (I(x) - J(x + d_k)) [Ix, Iy]^T
which is iterated (Newton-Raphson) and run coarse-to-fine on an image pyramid. J(x + d) sits between
pixels, so it is sampled with bilinear interpolation (bilinear() below).

The "actual" location in frame 2 is measured independently by normalised cross-correlation
template matching with a parabolic sub-pixel peak, and OpenCV's calcOpticalFlowPyrLK is run on the
same points as a second reference.
"""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HALF_WIN = 10      # 21 x 21 window, same as OpenCV's default
LEVELS = 4         # pyramid levels 0..3
MAX_ITER = 30
EPS = 0.01         # stop when the update is below 1/100 px
NCC_SEARCH = 48    # +-48 px search window for the template-matching ground truth
MIN_EIG = 0.01     # smallest eigenvalue of G per window pixel before we call it untrackable
MIN_NCC = 0.9      # below this the template match itself is doubtful, so it isn't used as ground truth


def bilinear(img, x, y):
    """f(x, y) = (1-a)(1-b) f00 + a(1-b) f10 + (1-a) b f01 + a b f11, with a, b the fractional parts."""
    h, w = img.shape
    x = np.clip(np.asarray(x, np.float64), 0, w - 1.000001)
    y = np.clip(np.asarray(y, np.float64), 0, h - 1.000001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    a = x - x0
    b = y - y0
    f00 = img[y0, x0]
    f10 = img[y0, x0 + 1]
    f01 = img[y0 + 1, x0]
    f11 = img[y0 + 1, x0 + 1]
    return (1 - a) * (1 - b) * f00 + a * (1 - b) * f10 + (1 - a) * b * f01 + a * b * f11


def bilinear_breakdown(img, x, y):
    """Same as bilinear() for one point, but returns every intermediate number for the write-up."""
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    a, b = x - x0, y - y0
    f = {k: float(img[y0 + dy, x0 + dx]) for k, (dx, dy) in
         {"f00": (0, 0), "f10": (1, 0), "f01": (0, 1), "f11": (1, 1)}.items()}
    w = {"w00": (1 - a) * (1 - b), "w10": a * (1 - b), "w01": (1 - a) * b, "w11": a * b}
    top = (1 - a) * f["f00"] + a * f["f10"]
    bottom = (1 - a) * f["f01"] + a * f["f11"]
    value = (1 - b) * top + b * bottom
    reference = float(cv2.getRectSubPix(img.astype(np.float32), (1, 1), (float(x), float(y)))[0, 0])
    return dict(x=x, y=y, x0=x0, y0=y0, a=a, b=b, top=top, bottom=bottom, value=value,
                reference=reference, **f, **w)


def gradients(img):
    # Scharr / 32 is a smoothed central difference, i.e. an estimate of dI/dx in intensity per pixel
    return cv2.Scharr(img, cv2.CV_32F, 1, 0) / 32.0, cv2.Scharr(img, cv2.CV_32F, 0, 1) / 32.0


class PyramidLK:
    def __init__(self, gray1, gray2, levels=LEVELS, half=HALF_WIN):
        self.half = half
        self.I = [gray1.astype(np.float32)]
        self.J = [gray2.astype(np.float32)]
        for _ in range(levels - 1):
            self.I.append(cv2.pyrDown(self.I[-1]))
            self.J.append(cv2.pyrDown(self.J[-1]))
        self.grads = [gradients(im) for im in self.I]
        oy, ox = np.mgrid[-half:half + 1, -half:half + 1]
        self.ox, self.oy = ox.astype(np.float64), oy.astype(np.float64)

    def track(self, pt, record=False):
        """Returns (d, info). d is the displacement at full resolution, info has the level-0 details."""
        g = np.zeros(2)
        info = {"ok": True}
        for L in reversed(range(len(self.I))):
            px, py = pt[0] / 2 ** L, pt[1] / 2 ** L
            xs, ys = px + self.ox, py + self.oy
            I = bilinear(self.I[L], xs, ys)
            Ix = bilinear(self.grads[L][0], xs, ys)
            Iy = bilinear(self.grads[L][1], xs, ys)

            G = np.array([[np.sum(Ix * Ix), np.sum(Ix * Iy)],
                          [np.sum(Ix * Iy), np.sum(Iy * Iy)]])
            eig = np.linalg.eigvalsh(G)
            if eig[0] / (2 * self.half + 1) ** 2 < MIN_EIG:
                info["ok"] = False  # flat patch or pure edge - aperture problem, G not invertible
                break

            d = np.zeros(2)
            iters = []
            for k in range(MAX_ITER):
                J = bilinear(self.J[L], xs + g[0] + d[0], ys + g[1] + d[1])
                diff = I - J
                b = np.array([np.sum(diff * Ix), np.sum(diff * Iy)])
                step = np.linalg.solve(G, b)
                if record and L == 0:
                    iters.append({"k": k, "dx": g[0] + d[0], "dy": g[1] + d[1], "bx": b[0], "by": b[1],
                                  "sx": step[0], "sy": step[1], "rms": float(np.sqrt(np.mean(diff ** 2)))})
                d += step
                if np.hypot(*step) < EPS:
                    break

            if L == 0:
                J_final = bilinear(self.J[0], xs + g[0] + d[0], ys + g[1] + d[1])
                info.update({
                    "G": G, "eig": eig, "guess": g.copy(), "iters": iters, "n_iter": k + 1,
                    "res_before": float(np.mean(np.abs(I - bilinear(self.J[0], xs, ys)))),
                    "res_after": float(np.mean(np.abs(I - J_final))),
                })
                g = g + d
            else:
                g = 2 * (g + d)
        return g, info


def ncc_actual(gray1, gray2, pt, half=HALF_WIN, search=NCC_SEARCH):
    """Where the patch around integer point pt actually is in frame 2 (NCC peak + parabola fit)."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = gray1.shape
    tpl = gray1[y - half:y + half + 1, x - half:x + half + 1]
    x_lo, x_hi = max(0, x - half - search), min(w, x + half + search + 1)
    y_lo, y_hi = max(0, y - half - search), min(h, y + half + search + 1)
    region = gray2[y_lo:y_hi, x_lo:x_hi]
    if tpl.shape != (2 * half + 1, 2 * half + 1) or region.shape[0] <= tpl.shape[0] or region.shape[1] <= tpl.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(region.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(res)

    def parabola(r_minus, r0, r_plus):
        den = r_minus - 2 * r0 + r_plus
        return 0.0 if abs(den) < 1e-12 else 0.5 * (r_minus - r_plus) / den

    sx = parabola(res[my, mx - 1], res[my, mx], res[my, mx + 1]) if 0 < mx < res.shape[1] - 1 else 0.0
    sy = parabola(res[my - 1, mx], res[my, mx], res[my + 1, mx]) if 0 < my < res.shape[0] - 1 else 0.0
    cx = x_lo + mx + half + sx
    cy = y_lo + my + half + sy
    return np.array([cx - x, cy - y]), float(score)


def pick_features(gray1, gray2, n=40):
    """Shi-Tomasi corners, favouring ones on things that actually move between the two frames."""
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    moving = (np.linalg.norm(flow, axis=2) > 0.7).astype(np.uint8) * 255
    border = HALF_WIN + 3
    moving[:border] = moving[-border:] = 0
    moving[:, :border] = moving[:, -border:] = 0
    anywhere = np.zeros_like(moving)
    anywhere[border:-border, border:-border] = 255

    pts = []
    first = cv2.goodFeaturesToTrack(gray1, int(n * 0.75), 0.01, 12, mask=moving)
    if first is not None:
        pts += [tuple(p) for p in first.reshape(-1, 2)]
    rest = cv2.goodFeaturesToTrack(gray1, n, 0.01, 12, mask=anywhere)
    if rest is not None:
        for p in rest.reshape(-1, 2):
            if len(pts) >= n:
                break
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 12 for q in pts):
                pts.append(tuple(p))
    # integer positions so the NCC template (which sits on the pixel grid) is centred on the same point
    return [(float(round(x)), float(round(y))) for x, y in pts]


def validate(frame1, frame2, n_features=40, manual_pairs=()):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    lk = PyramidLK(gray1, gray2)
    pts = pick_features(gray1, gray2, n_features)

    cv_pts = None
    if pts:
        p0 = np.array(pts, np.float32).reshape(-1, 1, 2)
        cv_pts, cv_status, _ = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, p0, None, winSize=(2 * HALF_WIN + 1, 2 * HALF_WIN + 1), maxLevel=LEVELS - 1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, MAX_ITER, EPS))

    rows = []
    for i, p in enumerate(pts):
        d, info = lk.track(p)
        actual, score = ncc_actual(gray1, gray2, p)
        if not info["ok"] or actual is None:
            continue
        cv_d = cv_pts[i, 0] - np.array(p) if cv_status[i, 0] else np.array([np.nan, np.nan])
        rows.append({
            "id": len(rows) + 1, "x": p[0], "y": p[1],
            "lk_dx": d[0], "lk_dy": d[1], "pred_x": p[0] + d[0], "pred_y": p[1] + d[1],
            "act_dx": actual[0], "act_dy": actual[1], "act_x": p[0] + actual[0], "act_y": p[1] + actual[1],
            "cv_dx": cv_d[0], "cv_dy": cv_d[1], "ncc": score, "lam_min": info["eig"][0],
            "err": float(np.hypot(*(d - actual))), "err_cv": float(np.hypot(*(d - cv_d))),
            "res_before": info["res_before"], "res_after": info["res_after"], "n_iter": info["n_iter"],
            "reliable": score > MIN_NCC,
        })

    manual = []
    for (x1, y1, x2, y2) in manual_pairs:
        d, info = lk.track((x1, y1))
        if not info["ok"]:
            manual.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "ok": False})
            continue
        manual.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "ok": True,
                       "pred_x": x1 + d[0], "pred_y": y1 + d[1],
                       "err": float(np.hypot(x1 + d[0] - x2, y1 + d[1] - y2))})

    # worked example: the biggest mover among features where the two measurements agree
    worked = None
    good = [r for r in rows if r["reliable"]]
    if good:
        pool = [r for r in good if r["err"] < 0.5] or good
        r = max(pool, key=lambda r: np.hypot(r["act_dx"], r["act_dy"]))
        d, info = lk.track((r["x"], r["y"]), record=True)
        worked = {"row": r, "info": info, "half": HALF_WIN, "levels": LEVELS,
                  "bilinear": bilinear_breakdown(gray2.astype(np.float32), r["x"] + d[0], r["y"] + d[1]),
                  "patch_value": float(gray1[int(r["y"]), int(r["x"])])}

    errs = np.array([r["err"] for r in good]) if good else np.array([np.nan])
    summary = {
        "n": len(rows), "n_reliable": len(good),
        "median_err": float(np.median(errs)), "mean_err": float(np.mean(errs)), "max_err": float(np.max(errs)),
        "pct_half": float(100 * np.mean(errs < 0.5)), "pct_one": float(100 * np.mean(errs < 1.0)),
        "median_err_cv": float(np.nanmedian([r["err_cv"] for r in good])) if good else float("nan"),
        "mean_motion": float(np.mean([np.hypot(r["act_dx"], r["act_dy"]) for r in good])) if good else 0.0,
        "res_before": float(np.mean([r["res_before"] for r in good])) if good else 0.0,
        "res_after": float(np.mean([r["res_after"] for r in good])) if good else 0.0,
    }
    return {"rows": rows, "manual": manual, "worked": worked, "summary": summary,
            "gray1": gray1, "gray2": gray2}


# ---------- figures ----------

def draw_tracks(frame1, frame2, rows, manual, arrow_scale=4.0):
    a = frame1.copy()
    b = frame2.copy()
    for r in rows:
        color = (60, 220, 60) if r["reliable"] else (150, 150, 150)
        p = (int(r["x"]), int(r["y"]))
        tip = (int(round(r["x"] + arrow_scale * r["lk_dx"])), int(round(r["y"] + arrow_scale * r["lk_dy"])))
        cv2.circle(a, p, 3, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(a, p, tip, color, 1, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(a, str(r["id"]), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.circle(b, (int(round(r["pred_x"])), int(round(r["pred_y"]))), 4, color, 1, cv2.LINE_AA)
        cv2.drawMarker(b, (int(round(r["act_x"])), int(round(r["act_y"]))), (40, 40, 255),
                       cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
    for m in manual:
        cv2.circle(a, (int(m["x1"]), int(m["y1"])), 5, (255, 180, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(b, (int(m["x2"]), int(m["y2"])), (255, 180, 0), cv2.MARKER_TILTED_CROSS, 9, 2)
        if m["ok"]:
            cv2.circle(b, (int(round(m["pred_x"])), int(round(m["pred_y"]))), 5, (60, 220, 60), 2, cv2.LINE_AA)
    return a, b


def zoom_grid(gray1, gray2, rows, count=4, radius=12, mag=10):
    """Magnified crops (nearest-neighbour, so the pixel grid is visible) for the biggest movers."""
    picks = sorted([r for r in rows if r["reliable"]], key=lambda r: -np.hypot(r["act_dx"], r["act_dy"]))[:count]
    tiles = []
    for r in picks:
        cx, cy = int(r["x"]), int(r["y"])
        pair = []
        for img, pts in ((gray1, [((r["x"], r["y"]), (255, 180, 0), "o")]),
                         (gray2, [((r["x"], r["y"]), (255, 180, 0), "o"),
                                  ((r["pred_x"], r["pred_y"]), (60, 220, 60), "o"),
                                  ((r["act_x"], r["act_y"]), (40, 40, 255), "+")])):
            crop = cv2.getRectSubPix(img, (2 * radius + 1, 2 * radius + 1), (cx, cy))
            big = cv2.resize(crop, None, fx=mag, fy=mag, interpolation=cv2.INTER_NEAREST)
            big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
            for (px, py), color, kind in pts:
                # pixel centre (cx, cy) maps to the middle of the magnified cell
                zx = int(round((px - cx + radius + 0.5) * mag))
                zy = int(round((py - cy + radius + 0.5) * mag))
                if kind == "o":
                    cv2.circle(big, (zx, zy), 7 if color[0] == 255 else 12, color, 2, cv2.LINE_AA)
                else:
                    cv2.drawMarker(big, (zx, zy), color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            pair.append(big)
        tile = np.hstack([pair[0], np.full((pair[0].shape[0], 6, 3), 30, np.uint8), pair[1]])
        cv2.putText(tile, f"#{r['id']}  frame t", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(tile, "frame t+1", (pair[0].shape[1] + 14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    if not tiles:
        return None
    gap = np.full((8, tiles[0].shape[1], 3), 30, np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        out = np.vstack([out, gap, t])
    return out


def plot_agreement(rows, path):
    good = [r for r in rows if r["reliable"]]
    if not good:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, key, name in ((axes[0], "dx", "horizontal"), (axes[1], "dy", "vertical")):
        act = np.array([r["act_" + key] for r in good])
        lk = np.array([r["lk_" + key] for r in good])
        lo, hi = min(act.min(), lk.min()) - 0.5, max(act.max(), lk.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], color="#999", lw=1, label="perfect agreement")
        ax.scatter(act, lk, s=22, color="#c9762f", edgecolor="#7a4415", zorder=3, label="features")
        ax.set_xlabel(f"actual {key} from NCC (px)")
        ax.set_ylabel(f"Lucas-Kanade {key} (px)")
        ax.set_title(f"{name} displacement")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    errs = [r["err"] for r in good]
    axes[2].hist(errs, bins=min(20, max(5, len(errs) // 2)), color="#c9762f", edgecolor="#7a4415")
    axes[2].axvline(np.median(errs), color="k", ls="--", lw=1, label=f"median {np.median(errs):.2f} px")
    axes[2].set_xlabel("|predicted - actual| (px)")
    axes[2].set_ylabel("features")
    axes[2].set_title("tracking error")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def difference_image(gray1, gray2):
    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(diff, cv2.COLORMAP_MAGMA)
