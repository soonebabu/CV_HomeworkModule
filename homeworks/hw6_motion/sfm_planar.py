"""Structure from motion for a flat object photographed from four viewpoints.

The same N points on the object (e.g. its corners and a few points along its edges, clicked in the
same order in every photo) are the only input besides the camera intrinsics K. Because the object is
planar, the 8-point / essential-matrix route is degenerate, so the reconstruction goes through the
plane-induced homography instead:

    view 1 -> view i :  lambda_i m_i = H_i lambda_1 m_1,   H_i = R_i + T_i N^T   (T_i = t_i / d)

1. normalised DLT for each H_i (views 2, 3, 4 against view 1)
2. SVD decomposition of each H_i into {R_i, T_i, N}: 4 candidates each, reduced by positive depth
   and by requiring all three homographies to agree on the plane normal N
3. linear triangulation of every point from all four views, then bundle adjustment
4. plane fit -> boundary polygon in the object's own 2-D frame, scaled to cm by one known length
"""
import itertools
import math

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares

# Samsung Galaxy S21 Ultra (SM-G998U) main camera, 1x, portrait 3000 x 4000 - from the HW2
# calibration run on the chessboard photos in calibration_images/ (10 x 7 inner corners).
DEFAULT_CAMERA = {
    "fx": 2767.12, "fy": 2775.44, "cx": 1477.98, "cy": 2051.34, "width": 3000, "height": 4000,
    "dist": [0.09570796, -0.30767867, -0.00221489, -0.0005458, 0.38901332],
    "source": "HW2 calibration of the SM-G998U main camera (checked-in chessboard photos)",
}


# ---------- small helpers ----------

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def to_h(p):
    p = np.asarray(p, float)
    return np.hstack([p, np.ones((len(p), 1))])


def scaled_K(cam, img_w, img_h):
    """K for an image of size img_w x img_h, given calibration at cam['width'] x cam['height'].

    If the photo is rotated relative to the calibration images (portrait vs landscape) the axes are
    swapped - fine for a principal point near the centre, which is what phones have.
    """
    fx, fy, cx, cy, W, H = cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["width"], cam["height"]
    dist = np.array(cam.get("dist") or [0, 0, 0, 0, 0], float)
    swapped = (img_w > img_h) != (W > H)
    if swapped:
        fx, fy, cx, cy, W, H = fy, fx, cy, cx, H, W
        dist = dist.copy()
        dist[2:4] = 0  # tangential terms don't survive the axis swap
    sx, sy = img_w / W, img_h / H
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
    return K, dist, swapped


def normalized_coords(pts_px, K, dist):
    """m = K^-1 x after removing lens distortion, as homogeneous 3-vectors."""
    pts = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    return to_h(und)


# ---------- 1. homography by normalised DLT ----------

def hartley_normalize(p):
    c = p.mean(axis=0)
    s = math.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return T, (to_h(p) @ T.T)[:, :2]


def dlt_homography(p1, p2):
    """H with p2 ~ H p1. Each correspondence gives two rows of A h = 0 (from p2 x (H p1) = 0)."""
    T1, a = hartley_normalize(p1)
    T2, b = hartley_normalize(p2)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(rows)
    _, S, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    return H / H[2, 2], S


# ---------- 2. decomposition H = R + T N^T ----------

def normalize_homography(H, m1):
    """Fix the scale (middle singular value = 1) and the sign (positive depth: (H m1)_z > 0)."""
    H = H / np.linalg.svd(H, compute_uv=False)[1]
    if np.sum((m1 @ H.T)[:, 2] > 0) < len(m1) / 2:
        H = -H
    return H


def decompose_homography(H):
    """The four {R, T, N} solutions of H = R + T N^T (Ma, Soatto, Kosecka & Sastry, sec. 5.3)."""
    S2, V = np.linalg.eigh(H.T @ H)          # ascending
    S2, V = S2[::-1], V[:, ::-1]             # sigma1^2 >= sigma2^2 (=1) >= sigma3^2
    if np.linalg.det(V) < 0:
        V = -V
    s1, s3 = S2[0], S2[2]
    v1, v2, v3 = V[:, 0], V[:, 1], V[:, 2]
    if s1 - s3 < 1e-9:
        return [], S2  # pure rotation: no translation, the plane can't be recovered from this view
    k = math.sqrt(s1 - s3)
    u1 = (math.sqrt(max(1 - s3, 0)) * v1 + math.sqrt(max(s1 - 1, 0)) * v3) / k
    u2 = (math.sqrt(max(1 - s3, 0)) * v1 - math.sqrt(max(s1 - 1, 0)) * v3) / k

    sols = []
    for u in (u1, u2):
        U = np.column_stack([v2, u, skew(v2) @ u])
        W = np.column_stack([H @ v2, H @ u, skew(H @ v2) @ (H @ u)])
        R = W @ U.T
        N = skew(v2) @ u
        T = (H - R) @ N
        sols.append((R, T, N))
        sols.append((R, -T, -N))
    return sols, S2


def choose_solutions(decomps, m1):
    """Keep candidates with every point in front of camera 1 (N^T m1 > 0), then pick the combination
    whose plane normals agree best across the three homographies - the true normal is shared."""
    kept = []
    for sols in decomps:
        ok = [i for i, (R, T, N) in enumerate(sols) if np.all(m1 @ N > 0)]
        kept.append(ok or list(range(len(sols))))
    best, best_cost = None, np.inf
    for combo in itertools.product(*kept):
        normals = [decomps[v][i][2] / np.linalg.norm(decomps[v][i][2]) for v, i in enumerate(combo)]
        cost = sum(math.acos(np.clip(a @ b, -1, 1)) for a, b in itertools.combinations(normals, 2))
        if cost < best_cost:
            best, best_cost = combo, cost
    return list(best), kept, math.degrees(best_cost)


# ---------- 3. triangulation + bundle adjustment ----------

def triangulate(ms, poses):
    """Linear triangulation: m x ([R|T] X) = 0 gives 2 rows per view; X = null vector of A."""
    rows = []
    for m, (R, T) in zip(ms, poses):
        P = np.hstack([R, T.reshape(3, 1)])
        rows.append(m[0] * P[2] - P[0])
        rows.append(m[1] * P[2] - P[1])
    A = np.array(rows)
    _, S, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    return Xh[:3] / Xh[3], A, S, Xh


def project(K, R, T, X):
    x = (K @ (X @ R.T + T).T).T
    return x[:, :2] / x[:, 2:3]


def reprojection_errors(K, poses, X, obs_px):
    return [np.linalg.norm(project(K, R, T, X) - obs, axis=1) for (R, T), obs in zip(poses, obs_px)]


def bundle_adjust(K, poses, X, obs_px):
    """Minimise total reprojection error over the poses of views 2-4 and all 3-D points.
    View 1 stays at [I | 0]; a soft constraint on |T_2| pins the overall scale."""
    n_views, n_pts = len(poses), len(X)
    t2_norm = np.linalg.norm(poses[1][1])

    def unpack(p):
        out = [(np.eye(3), np.zeros(3))]
        for v in range(n_views - 1):
            rv, t = p[6 * v:6 * v + 3], p[6 * v + 3:6 * v + 6]
            out.append((cv2.Rodrigues(rv)[0], t))
        return out, p[6 * (n_views - 1):].reshape(n_pts, 3)

    def residuals(p):
        ps, Xs = unpack(p)
        r = [(project(K, R, T, Xs) - obs).ravel() for (R, T), obs in zip(ps, obs_px)]
        r.append([100.0 * (np.linalg.norm(ps[1][1]) - t2_norm)])
        return np.concatenate(r)

    p0 = []
    for R, T in poses[1:]:
        p0 += list(cv2.Rodrigues(R)[0].ravel()) + list(T)
    p0 = np.concatenate([p0, X.ravel()])
    res = least_squares(residuals, p0, method="trf", x_scale="jac")
    new_poses, new_X = unpack(res.x)
    return new_poses, new_X, res


# ---------- 4. plane, boundary, scale ----------

def fit_plane(X):
    """Least-squares plane through X. n points away from camera 1, like N in N^T X = d (d > 0)."""
    c = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - c)
    n = Vt[-1]
    if n @ c < 0:
        n = -n
    return c, n, S[-1] / math.sqrt(len(X))


def object_frame(X, n, i=0, j=1):
    """Right-handed frame on the object: origin at point i, x along edge i->j, z towards the camera.
    Seen from the camera side that is x right, y up, so plots of (x, y) aren't mirrored."""
    z = -n
    e1 = X[j] - X[i]
    e1 = e1 - (e1 @ z) * z
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(z, e1)
    return np.column_stack([e1, e2, z]), X[i]


def shoelace(uv):
    u, v = uv[:, 0], uv[:, 1]
    return 0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))


def reconstruct(points_px, K, dist, known=(0, 1, 10.0), actual_lengths=None):
    """points_px: list of 4 arrays (N x 2) of clicked pixel positions, same point order in each view."""
    obs = [np.asarray(p, float) for p in points_px]
    n_pts = len(obs[0])
    ms = [normalized_coords(p, K, dist) for p in obs]
    # undistorted pixel positions - what an ideal pinhole with this K would have recorded
    obs_ud = [(m @ K.T)[:, :2] for m in ms]
    m1 = ms[0]

    homogs, decomps = [], []
    for i in range(1, 4):
        H_raw, dlt_sv = dlt_homography(m1[:, :2], ms[i][:, :2])
        H = normalize_homography(H_raw, m1)
        sols, sig2 = decompose_homography(H)
        if not sols:
            raise ValueError(f"View {i + 1} looks like a pure rotation of view 1 (no translation), so depth "
                             f"can't be recovered from it. Move the camera, don't just turn it.")
        transfer = np.linalg.norm(project(K, H, np.zeros(3), m1) - obs_ud[i], axis=1)
        homogs.append({"H_raw": H_raw, "H": H, "H_px": K @ H @ np.linalg.inv(K), "dlt_sv": dlt_sv,
                       "sig2": sig2, "transfer_err": float(transfer.mean())})
        decomps.append(sols)

    choice, kept, normal_spread = choose_solutions(decomps, m1)
    N = np.mean([decomps[v][c][2] for v, c in enumerate(choice)], axis=0)
    poses = [(np.eye(3), np.zeros(3))] + [decomps[v][c][:2] for v, c in enumerate(choice)]

    X_lin, tri = [], []
    for j in range(n_pts):
        X, A, S, Xh = triangulate([m[j] for m in ms], poses)
        X_lin.append(X)
        tri.append({"A": A, "S": S, "Xh": Xh})
    X_lin = np.array(X_lin)
    err_lin = reprojection_errors(K, poses, X_lin, obs_ud)

    poses_ba, X_ba, ba = bundle_adjust(K, poses, X_lin, obs_ud)
    err_ba = reprojection_errors(K, poses_ba, X_ba, obs_ud)

    # express everything with the plane at distance d = 1 from camera 1 (the scale the homography gave us)
    c, n, flat_rms = fit_plane(X_ba)
    d_plane = float(n @ c)
    X_ba = X_ba / d_plane
    poses_ba = [(R, T / d_plane) for R, T in poses_ba]
    c, n, flat_rms = fit_plane(X_ba)

    i, j, L = known
    scale = L / np.linalg.norm(X_ba[i] - X_ba[j])  # cm per reconstruction unit
    B, origin = object_frame(X_ba, n, i, j)
    uvz = (X_ba - origin) @ B * scale

    edges = []
    for k in range(n_pts):
        a, b = k, (k + 1) % n_pts
        length = float(np.linalg.norm(uvz[b, :2] - uvz[a, :2]))
        actual = actual_lengths[k] if actual_lengths and k < len(actual_lengths) else None
        edges.append({"a": a + 1, "b": b + 1, "length": length, "actual": actual,
                      "err": None if actual is None else length - actual,
                      "err_pct": None if not actual else 100 * (length - actual) / actual})

    cameras = []
    for v, (R, T) in enumerate(poses_ba):
        C = -R.T @ T                           # camera centre in camera-1 coordinates
        C_obj = (C - origin) @ B * scale
        axis_obj = B.T @ (R.T @ np.array([0, 0, 1.0]))
        centroid_obj = uvz.mean(axis=0)
        cameras.append({
            "view": v + 1, "C_obj": C_obj, "R": R, "T": T,
            "dist_to_center": float(np.linalg.norm(C_obj - centroid_obj)),
            "height": float(C_obj[2]),
            "tilt": float(math.degrees(math.acos(np.clip(-axis_obj[2], -1, 1)))),
            "axis_obj": axis_obj,
        })

    # numbers for the step-by-step write-up, following point 1 through the whole pipeline
    X1 = X_lin[0]
    reproj1 = []
    for (R, T), x_obs in zip(poses, obs_ud):
        xh = K @ (R @ X1 + T)
        reproj1.append({"cam": R @ X1 + T, "xh": xh, "px": xh[:2] / xh[2], "obs": x_obs[0]})
    worked = {
        "depth_ok": [[int(np.sum(m1 @ N_ > 0)) for _, _, N_ in sols] for sols in decomps],
        # rotation angle from the trace: tr(R) = 1 + 2 cos(theta)
        "rot_deg": [[math.degrees(math.acos(np.clip((np.trace(R_) - 1) / 2, -1, 1))) for R_, _, _ in sols]
                    for sols in decomps],
        "X_plane": m1[0] / (N @ m1[0]) * np.linalg.norm(N),
        "N_dot_X": float(N @ X1 / np.linalg.norm(N)),
        "reproj1": reproj1,
        "Xa": X_ba[i], "Xb": X_ba[j], "ab_units": float(np.linalg.norm(X_ba[i] - X_ba[j])),
        "shoelace": [(uvz[k, 0] * uvz[(k + 1) % n_pts, 1] - uvz[(k + 1) % n_pts, 0] * uvz[k, 1]) for k in range(n_pts)],
    }

    return {
        "worked": worked,
        "n_pts": n_pts, "K": K, "dist": dist, "obs": obs, "obs_ud": obs_ud, "ms": ms,
        "homogs": homogs, "decomps": decomps, "choice": choice, "kept": kept,
        "normal_spread": normal_spread, "N_consensus": N / np.linalg.norm(N),
        "poses_lin": poses, "X_lin": X_lin, "err_lin": err_lin, "tri": tri,
        "poses": poses_ba, "X": X_ba, "err_ba": err_ba, "ba_cost": float(ba.cost), "ba_nfev": int(ba.nfev),
        "plane_c": c, "plane_n": n, "flat_rms_cm": float(flat_rms * scale),
        "scale": float(scale), "known": known, "B": B, "origin": origin, "uvz": uvz,
        "edges": edges, "perimeter": float(sum(e["length"] for e in edges)), "area": float(shoelace(uvz[:, :2])),
        "cameras": cameras,
    }


# ---------- figures ----------

def draw_view(img, obs, reproj, idx):
    out = img.copy()
    s = max(1.0, max(img.shape[:2]) / 900)
    pts = np.round(obs).astype(int)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 220, 255), max(1, int(2 * s)), cv2.LINE_AA)
    for k, (p, q) in enumerate(zip(obs, reproj)):
        cv2.circle(out, tuple(np.round(p).astype(int)), int(5 * s), (40, 40, 255), -1, cv2.LINE_AA)
        cv2.circle(out, tuple(np.round(q).astype(int)), int(9 * s), (60, 220, 60), max(1, int(2 * s)), cv2.LINE_AA)
        cv2.putText(out, str(k + 1), (int(p[0] + 8 * s), int(p[1] - 8 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(out, f"view {idx}", (int(12 * s), int(34 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s,
                (255, 255, 255), max(2, int(3 * s)), cv2.LINE_AA)
    return out


def plot_scene(rec, img_size, path):
    uvz = rec["uvz"]
    K = rec["K"]
    w, h = img_size
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    poly = np.vstack([uvz, uvz[:1]])
    ax.add_collection3d(Poly3DCollection([uvz], alpha=0.25, facecolor="#c9762f"))
    ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color="#c9762f", lw=2)
    ax.scatter(uvz[:, 0], uvz[:, 1], uvz[:, 2], color="#7a4415", s=18)
    for k, p in enumerate(uvz):
        ax.text(p[0], p[1], p[2], f" {k + 1}", fontsize=8)

    size = np.ptp(uvz[:, :2], axis=0).max()
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    corners_px = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
    all_pts = [uvz]
    for cam, col in zip(rec["cameras"], colors):
        C = cam["C_obj"]
        R = cam["R"]
        depth = 0.25 * size
        rays = (np.linalg.inv(K) @ corners_px.T).T @ R          # camera -> camera-1 frame (R^T d)
        rays = rays @ rec["B"]                                  # camera-1 frame -> object frame
        rays = rays / (rays @ cam["axis_obj"])[:, None] * depth
        ends = C + rays
        for e in ends:
            ax.plot(*zip(C, e), color=col, lw=0.8)
        ring = np.vstack([ends, ends[:1]])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=col, lw=1)
        ax.scatter(*C, color=col, s=30)
        ax.text(*C, f"  cam {cam['view']}", color=col, fontsize=9)
        all_pts.append(np.vstack([C, ends]))

    pts = np.vstack(all_pts)
    mid = (pts.max(0) + pts.min(0)) / 2
    r = np.ptp(pts, axis=0).max() / 2
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title("Reconstructed object and the four camera positions (object frame)")
    ax.view_init(elev=28, azim=-60)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_boundary(rec, path):
    uv = rec["uvz"][:, :2]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    poly = np.vstack([uv, uv[:1]])
    ax.fill(poly[:, 0], poly[:, 1], color="#c9762f", alpha=0.2)
    ax.plot(poly[:, 0], poly[:, 1], "-o", color="#c9762f", mec="#7a4415")
    for k, p in enumerate(uv):
        ax.annotate(str(k + 1), p, textcoords="offset points", xytext=(6, 6), fontsize=9)
    for e in rec["edges"]:
        a, b = uv[e["a"] - 1], uv[e["b"] - 1]
        label = f"{e['length']:.1f}"
        if e["actual"]:
            label += f"\n({e['actual']:.1f})"
        ax.annotate(label, (a + b) / 2, ha="center", va="center", fontsize=8, color="#333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc"))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x along edge 1-2 (cm)")
    ax.set_ylabel("y in the object plane (cm)")
    sub = "  (actual in brackets)" if any(e["actual"] for e in rec["edges"]) else ""
    ax.set_title(f"Recovered boundary - area {rec['area']:.1f} cm², perimeter {rec['perimeter']:.1f} cm{sub}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def rectify_view1(img, rec, px_per_cm=20, margin_cm=2.0):
    """Top-down (fronto-parallel) picture of the object, warped out of view 1 with the recovered plane.
    pixel ~ K (origin + u/s e1 + v/s e2) = K [e1/s  e2/s  origin] [u v 1]^T"""
    K, B, s = rec["K"], rec["B"], rec["scale"]
    undist = cv2.undistort(img, K, rec["dist"])
    uv = rec["uvz"][:, :2]
    lo = uv.min(axis=0) - margin_cm
    hi = uv.max(axis=0) + margin_cm
    size = np.ceil((hi - lo) * px_per_cm).astype(int)
    if size.max() > 2400:
        px_per_cm *= 2400 / size.max()
        size = np.ceil((hi - lo) * px_per_cm).astype(int)
    # output pixel (p, q) -> plane coords u = lo_u + p / px, v = hi_v - q / px (image rows go down, v goes up)
    out_to_uv = np.array([[1 / px_per_cm, 0, lo[0]], [0, -1 / px_per_cm, hi[1]], [0, 0, 1]])
    uv_to_img = K @ np.column_stack([B[:, 0] / s, B[:, 1] / s, rec["origin"]])
    Hmap = uv_to_img @ out_to_uv
    top = cv2.warpPerspective(undist, Hmap, (int(size[0]), int(size[1])), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    poly = np.round(np.column_stack([uv[:, 0] - lo[0], hi[1] - uv[:, 1]]) * px_per_cm).astype(int)
    cv2.polylines(top, [poly.reshape(-1, 1, 2)], True, (0, 220, 255), 2, cv2.LINE_AA)
    return top


# ---------- LaTeX for the write-up ----------

def tex(M, digits=4):
    M = np.atleast_2d(np.asarray(M, float))
    fmt = lambda v: f"{v:.{digits}f}" if abs(v) >= 10 ** -digits or v == 0 else f"{v:.2e}"
    body = r" \\ ".join(" & ".join(fmt(v) for v in row) for row in M)
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"


def tex_col(v, digits=4):
    return tex(np.asarray(v, float).reshape(-1, 1), digits)
