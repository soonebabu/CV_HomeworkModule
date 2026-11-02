"""Room dimensions from one video.

A moving phone sees the room from many positions, so any two frames act as a stereo pair.

  1. Track the phone through the clip: SIFT matches between frames, triangulation for new 3D points, and PnP
     against the points found so far. One moving camera only knows the scene up to scale, so the sideways
     slide at the start of the clip, of known length, sets the real size.
  2. Build a dense cloud: pairs of frames are rectified and Z = f B / d gives depth, with B the distance the
     phone moved between them. Points seen only once are dropped.
  3. Fit the room: find its three axes from the biggest planes, then the two outermost surfaces along each.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ..hw6_motion import video_io
from . import stereo as st

MIN_Z_MM, MAX_Z_MM = 300.0, 12000.0
VOXEL_MM = 40.0
MIN_ROOM_SPAN_MM = 800.0  # two surfaces closer than this are not opposite walls


class RoomError(ValueError):
    """The clip can't be turned into a room. The message is shown to the user."""


# ---------------- reading the clip ----------------

def read_frames(path, n_frames, max_dim):
    """Evenly spaced frames: a list of (time_s, frame)."""
    info = video_io.clip_info(path)
    if info is None or not info["duration"]:
        raise RoomError("Could not read that video file.")
    step = max(1, round(info["duration"] * info["fps"] / n_frames))
    frames = [(t, frame) for _, t, frame in video_io.iter_frames(path, 0.0, info["duration"], max_dim, step)]
    if len(frames) < 3:
        raise RoomError("The video gave fewer than 3 frames.")
    return frames


# ---------------- step 1: camera tracking ----------------

class Tracker:
    """Incremental structure from motion. Poses are world-to-camera (R, t); K has no distortion left."""

    def __init__(self, K):
        self.K = K
        # the low contrast threshold keeps the faint texture of painted walls, often all there is to see
        self.sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.012)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.keyframes = []
        self.points = np.empty((0, 3))

    def _detect(self, img):
        kp, des = self.sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        if des is None:
            return np.empty((0, 2), np.float32), None
        return np.array([k.pt for k in kp], np.float32), des

    def _match(self, des1, des2):
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return np.empty((0, 2), np.int32)
        pairs = self.matcher.knnMatch(des1, des2, k=2)
        good = [(p[0].queryIdx, p[0].trainIdx) for p in pairs
                if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        return np.array(good, np.int32).reshape(-1, 2)

    def _add_points(self, xyz):
        first = len(self.points)
        self.points = np.vstack([self.points, xyz])
        return np.arange(first, first + len(xyz))

    def _keyframe(self, index, t, R, tvec, pts, des, ids, inliers):
        self.keyframes.append({"index": index, "t": t, "R": R, "tvec": tvec, "pts": pts, "des": des,
                               "ids": ids, "inliers": inliers})

    def _locate(self, pts, des):
        """PnP against the 3D points held by the last few keyframes. Returns pose, ids and inlier count."""
        ids = np.full(len(pts), -1, np.int64)
        for ref in reversed(self.keyframes[-3:]):
            for q, r in self._match(des, ref["des"]):
                if ids[q] < 0 and ref["ids"][r] >= 0:
                    ids[q] = ref["ids"][r]
        have = np.nonzero(ids >= 0)[0]
        if len(have) < 12:
            return None
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            self.points[ids[have]], pts[have], self.K, None, iterationsCount=300, reprojectionError=2.5,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 12:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(self.points[ids[have[inl]]], pts[have[inl]], self.K, None, rvec, tvec)
        outliers = np.setdiff1d(have, have[inl])
        ids[outliers] = -1
        return cv2.Rodrigues(rvec)[0], tvec.ravel(), ids, len(inl)

    def start(self, frames, slide_end_s):
        """Starts the map from the two ends of the opening slide, the widest baseline that has a known length.

        pose_from_matches copes with a flat wall in view, which is the usual sight at the start of a clip.
        """
        no_dist = np.zeros(5)
        pts0, des0 = self._detect(frames[0][1])
        last = int(np.argmin([abs(t - slide_end_s) for t, _ in frames]))
        for j in range(last, 0, -1):
            pts1, des1 = self._detect(frames[j][1])
            m = self._match(des0, des1)
            if len(m) < 40:
                continue
            p0, p1 = pts0[m[:, 0]].astype(np.float64), pts1[m[:, 1]].astype(np.float64)
            try:
                R, t, _ = st.pose_from_matches(p0, p1, self.K, no_dist, self.K, no_dist)
            except st.StereoError:
                continue
            xyz, angles = self._triangulate(np.eye(3), np.zeros(3), R, t, p0, p1)
            ok = angles > 0.5
            if abs(t[0]) < 0.7 or ok.sum() < 30:
                continue
            ids0, ids1 = np.full(len(pts0), -1, np.int64), np.full(len(pts1), -1, np.int64)
            new = self._add_points(xyz[ok])
            ids0[m[ok, 0]] = new
            ids1[m[ok, 1]] = new
            self._keyframe(0, frames[0][0], np.eye(3), np.zeros(3), pts0, des0, ids0, len(new))
            self._keyframe(j, frames[j][0], R, t, pts1, des1, ids1, len(new))
            return j
        raise RoomError("Could not start tracking: the video has to open with a slow straight sideways slide, "
                        "not a turn on the spot.")

    def _triangulate(self, R1, t1, R2, t2, p1, p2):
        """3D points and the angle each one subtends. Bad points get angle 0: behind a camera or far off."""
        h = cv2.triangulatePoints(self.K @ np.column_stack([R1, t1]), self.K @ np.column_stack([R2, t2]), p1.T, p2.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = (h[:3] / h[3]).T
        xyz = np.nan_to_num(xyz)
        v1, v2 = xyz + R1.T @ t1, xyz + R2.T @ t2
        cosang = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
        angles = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        for R, t, p in ((R1, t1, p1), (R2, t2, p2)):
            cam = xyz @ R.T + t
            with np.errstate(divide="ignore", invalid="ignore"):
                proj = (cam / cam[:, 2:3]) @ self.K.T
            good = (cam[:, 2] > 0) & (np.linalg.norm(proj[:, :2] - p, axis=1) < 2.5)
            angles = np.where(good, angles, 0.0)
        return xyz, angles

    def add(self, index, t, img):
        pts, des = self._detect(img)
        located = self._locate(pts, des)
        if located is None:
            return False
        R, tvec, ids, n = located
        self._keyframe(index, t, R, tvec, pts, des, ids, n)
        cur = self.keyframes[-1]
        # older keyframes have the wider baseline, which is what a turn needs to triangulate anything
        for ref in self.keyframes[-9:-1:2]:
            m = self._match(des, ref["des"])
            if not len(m):
                continue
            m = m[(cur["ids"][m[:, 0]] < 0) & (ref["ids"][m[:, 1]] < 0)]
            if not len(m):
                continue
            xyz, angles = self._triangulate(ref["R"], ref["tvec"], R, tvec, ref["pts"][m[:, 1]].astype(np.float64),
                                            pts[m[:, 0]].astype(np.float64))
            ok = angles > 2.0  # weaker points pull the scale down over time
            if ok.any():
                new = self._add_points(xyz[ok])
                cur["ids"][m[ok, 0]] = new
                ref["ids"][m[ok, 1]] = new
        return True

    def refine_recent(self, window=16, fixed=3, max_points=900):
        """Bundle adjustment over the last few keyframes, oldest ones held still.

        Each new pose is built on points that came from earlier poses, so one camera slowly loses its scale,
        most of all while turning. Optimising recent cameras and points together keeps it in check.
        """
        kfs = self.keyframes[-window:]
        if len(kfs) <= fixed:
            return
        seen = np.zeros(len(self.points), np.int32)
        for kf in kfs:
            np.add.at(seen, kf["ids"][kf["ids"] >= 0], 1)
        chosen = np.nonzero(seen >= 2)[0]
        if len(chosen) < 20:
            return
        chosen = chosen[np.argsort(-seen[chosen], kind="stable")[:max_points]]
        slot = np.full(len(self.points), -1, np.int64)
        slot[chosen] = np.arange(len(chosen))

        cam_i, pt_i, obs = [], [], []
        for c, kf in enumerate(kfs):
            rows = np.nonzero(kf["ids"] >= 0)[0]
            rows = rows[slot[kf["ids"][rows]] >= 0]
            cam_i.append(np.full(len(rows), c))
            pt_i.append(slot[kf["ids"][rows]])
            obs.append(kf["pts"][rows].astype(np.float64))
        cam_i, pt_i, obs = np.concatenate(cam_i), np.concatenate(pt_i), np.concatenate(obs)

        poses = np.array([np.r_[cv2.Rodrigues(kf["R"])[0].ravel(), kf["tvec"]] for kf in kfs])
        n_free = len(kfs) - fixed
        x0 = np.r_[poses[fixed:].ravel(), self.points[chosen].ravel()]
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        def residuals(x):
            cams = np.vstack([poses[:fixed], x[:6 * n_free].reshape(-1, 6)])[cam_i]
            p = x[6 * n_free:].reshape(-1, 3)[pt_i]
            theta = np.linalg.norm(cams[:, :3], axis=1, keepdims=True)
            axis = cams[:, :3] / np.maximum(theta, 1e-12)
            rotated = p * np.cos(theta) + np.cross(axis, p) * np.sin(theta) \
                + axis * (axis * p).sum(axis=1, keepdims=True) * (1 - np.cos(theta))
            cam = rotated + cams[:, 3:]
            z = np.where(np.abs(cam[:, 2]) < 1e-9, 1e-9, cam[:, 2])
            return np.r_[fx * cam[:, 0] / z + cx - obs[:, 0], fy * cam[:, 1] / z + cy - obs[:, 1]]

        rows = np.arange(len(obs))
        sparsity = lil_matrix((2 * len(obs), len(x0)), dtype=np.uint8)
        free = np.nonzero(cam_i >= fixed)[0]
        for k in range(6):
            col = 6 * (cam_i[free] - fixed) + k
            sparsity[free, col] = 1
            sparsity[len(obs) + free, col] = 1
        for k in range(3):
            col = 6 * n_free + 3 * pt_i + k
            sparsity[rows, col] = 1
            sparsity[len(obs) + rows, col] = 1

        # step sizes for a scene a few units across: x_scale="jac" starts with steps far too long to be accepted
        x_scale = np.r_[np.tile([0.02, 0.02, 0.02, 0.05, 0.05, 0.05], n_free), np.full(3 * len(chosen), 0.2)]
        sol = least_squares(residuals, x0, jac_sparsity=sparsity, loss="huber", f_scale=2.0, x_scale=x_scale,
                            max_nfev=15, tr_solver="lsmr")
        cams = sol.x[:6 * n_free].reshape(-1, 6)
        for kf, cam in zip(kfs[fixed:], cams):
            kf["R"], kf["tvec"] = cv2.Rodrigues(cam[:3])[0], cam[3:].copy()
        self.points[chosen] = sol.x[6 * n_free:].reshape(-1, 3)

    def rescale(self, s):
        self.points = self.points * s
        for kf in self.keyframes:
            kf["tvec"] = kf["tvec"] * s

    def centres(self):
        return np.array([-kf["R"].T @ kf["tvec"] for kf in self.keyframes])


def check_tracking(tracker, n_frames):
    """Refuses a partial track. It would give a confident size for only the part of the room it saw."""
    if len(tracker.keyframes) < 0.8 * n_frames:
        raise RoomError(f"Tracking was lost: only {len(tracker.keyframes)} of {n_frames} frames could be placed, so "
                        "most of the room is missing. Use more frames, move more slowly, or add texture to "
                        "plain walls.")


def close_loop(tracker, min_inliers=25):
    """Removes the drift of a lap around the room.

    Small errors add up along the path, so by the end of a lap the first wall is seen again some way from
    where it started and shows up as two walls. When the last frame sees the start of the map again,
    locating it against those early points says how far the track has drifted. That correction is then
    shared out along the path, growing from nothing at the start to all of it at the end.
    """
    kfs = tracker.keyframes
    last = kfs[-1]
    best = None
    for ref in kfs[:max(1, len(kfs) // 3)]:
        m = tracker._match(last["des"], ref["des"])
        if not len(m):
            continue
        m = m[ref["ids"][m[:, 1]] >= 0]
        if len(m) < min_inliers:
            continue
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            tracker.points[ref["ids"][m[:, 1]]], last["pts"][m[:, 0]], tracker.K, None,
            iterationsCount=300, reprojectionError=2.5, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if ok and inl is not None and len(inl) >= min_inliers and (best is None or len(inl) > best[0]):
            best = (len(inl), cv2.Rodrigues(rvec)[0], tvec.ravel())
    if best is None:
        return False

    _, R_true, t_true = best
    R_drift = R_true.T @ last["R"]
    t_drift = R_true.T @ (last["tvec"] - t_true)
    rvec_drift = cv2.Rodrigues(R_drift)[0].ravel()
    n = len(kfs) - 1
    for i, kf in enumerate(kfs):
        share = i / n
        R_i = cv2.Rodrigues(rvec_drift * share)[0]
        kf["R"] = kf["R"] @ R_i.T
        kf["tvec"] = kf["tvec"] - kf["R"] @ (t_drift * share)
    return True




# ---------------- step 2: dense cloud ----------------

def _cloud_from_disparity(gray, disp, rect, step=2):
    mask = np.isfinite(disp) & (disp > 0) & st.textured_mask(gray)
    mask[:, :int(gray.shape[1] * 0.04)] = False
    sel = np.zeros_like(mask)
    sel[::step, ::step] = True
    ys, xs = np.nonzero(mask & sel)
    if not len(xs):
        return np.empty((0, 3))
    cam = st.to_3d(np.column_stack([xs, ys]), disp[ys, xs], rect)
    return cam[(cam[:, 2] > MIN_Z_MM) & (cam[:, 2] < MAX_Z_MM)]


def voxel_filter(points, labels, voxel=VOXEL_MM, min_views=2):
    """Keeps one point per voxel, for voxels seen from at least min_views keyframes.

    Matching mistakes land somewhere different from each viewpoint; real surfaces land in the same place.
    """
    if not len(points):
        return points
    cells = np.floor(points / voxel).astype(np.int64)
    pack = (cells[:, 0] + 2 ** 20) * 2 ** 42 + (cells[:, 1] + 2 ** 20) * 2 ** 21 + (cells[:, 2] + 2 ** 20)
    order = np.lexsort((labels, pack))
    pack, labels = pack[order], labels[order]
    pts = points[order]
    starts = np.r_[0, np.nonzero(np.diff(pack))[0] + 1]
    ends = np.r_[starts[1:], len(pack)]
    new_label = np.r_[True, (np.diff(pack) != 0) | (np.diff(labels) != 0)]
    views = np.add.reduceat(new_label.astype(np.int32), starts)
    keep = views >= min_views
    means = np.add.reduceat(pts, starts, axis=0) / (ends - starts)[:, None]
    return means[keep]


def dense_cloud(tracker, images, max_pairs=60):
    """Uses pairs of keyframes as stereo pairs: B is how far the phone moved between them."""
    no_dist = np.zeros(5)
    K, kfs = tracker.K, tracker.keyframes
    size = (images[0].shape[1], images[0].shape[0])
    stride = max(1, len(kfs) // max_pairs)
    clouds, labels = [], []
    for a in range(0, len(kfs), stride):
        for b in range(a + 1, min(a + 8, len(kfs))):
            R = kfs[b]["R"] @ kfs[a]["R"].T
            T = kfs[b]["tvec"] - R @ kfs[a]["tvec"]
            baseline = float(np.linalg.norm(T))
            t = T / max(baseline, 1e-9)
            if baseline < 120.0 or abs(t[0]) < max(abs(t[1]), 0.6) \
                    or np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])) > 14.0:
                continue
            rig = st.make_rig(K, no_dist, K, no_dist, R, T)
            first = b if rig["swapped"] else a
            second = a if rig["swapped"] else b
            rect = st.rectify_maps(rig, size)
            left, right = st.apply_rectify(images[first], images[second], rect)
            disp = st.compute_disparity(left, right)
            cam = _cloud_from_disparity(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), disp, rect) @ rect["R1"]
            clouds.append((cam - kfs[first]["tvec"]) @ kfs[first]["R"])
            labels.append(np.full(len(cam), a))
            break
    if not clouds:
        return np.empty((0, 3))
    return voxel_filter(np.vstack(clouds), np.concatenate(labels))


# ---------------- step 3: fitting the box ----------------

def extract_planes(points, n_planes=7, tol=35.0, iters=500, min_support=80, seed=1):
    """Sequential RANSAC: repeatedly take the biggest plane and remove its points."""
    rng = np.random.default_rng(seed)
    rest = points
    planes = []
    for _ in range(n_planes):
        if len(rest) < min_support:
            break
        i = rng.integers(0, len(rest), (iters, 3))
        a, b, c = rest[i[:, 0]], rest[i[:, 1]], rest[i[:, 2]]
        normals = np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        ok = lengths > 1e-6
        normals, a = normals[ok] / lengths[ok, None], a[ok]
        sample = rest if len(rest) < 4000 else rest[rng.choice(len(rest), 4000, replace=False)]
        counts = (np.abs(sample @ normals.T - (a * normals).sum(axis=1)) < tol).sum(axis=0)
        k = int(np.argmax(counts))
        n, offset = normals[k], float(a[k] @ normals[k])
        inl = np.abs(rest @ n - offset) < tol
        if inl.sum() < min_support:
            break
        centre = rest[inl].mean(axis=0)
        n = np.linalg.svd(rest[inl] - centre, full_matrices=False)[2][2]
        planes.append((n, int(inl.sum())))
        rest = rest[~inl]
    return planes


def room_axes(planes, down):
    """Three orthogonal room directions: the strongest pair of perpendicular planes, then their cross product.

    Returns (vertical, horizontal_a, horizontal_b); vertical is the axis nearest the camera's down direction.
    """
    best, score = None, -1
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            if abs(planes[i][0] @ planes[j][0]) < np.sin(np.radians(10)):
                s = planes[i][1] + planes[j][1]
                if s > score:
                    best, score = (planes[i][0], planes[j][0]), s
    if best is None:
        raise RoomError("Could not find two perpendicular surfaces. The clip needs to show at least two "
                        "walls, or a wall and the floor.")
    a = best[0]
    b = best[1] - (best[1] @ a) * a
    b /= np.linalg.norm(b)
    axes = [a, b, np.cross(a, b)]
    v = int(np.argmax([abs(x @ down) for x in axes]))
    vertical = axes.pop(v)
    return (vertical if vertical @ down > 0 else -vertical), axes[0], axes[1]


def surface_peaks(values, bin_mm=40.0, min_share=0.18, min_points=40):
    """Candidate surfaces along one axis, from the histogram of point positions, lowest position first.

    Planes show up as sharp peaks. A peak counts if it holds a fair share of the tallest one, which ignores
    the scattered stray points that can lie beyond the real walls.
    """
    lo, hi = np.percentile(values, 0.1), np.percentile(values, 99.9)
    edges = np.arange(lo - bin_mm, hi + 2 * bin_mm, bin_mm)
    hist = np.histogram(values, edges)[0].astype(np.float64)
    smooth = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    limit = max(min_points, smooth.max() * min_share)
    centres = (edges[:-1] + edges[1:]) / 2
    peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] >= limit and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            peaks.append(float(np.median(values[np.abs(values - centres[i]) < 1.5 * bin_mm])))
    return peaks, centres, hist


def _span(values):
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def _axis_result(axis, peaks, centres, hist):
    pair = (peaks[0], peaks[-1]) if len(peaks) >= 2 and peaks[-1] - peaks[0] >= MIN_ROOM_SPAN_MM else None
    return {"axis": axis, "pair": pair, "centres": centres, "hist": hist,
            "size": pair[1] - pair[0] if pair else None}


def _broad_surfaces(cloud, vertical, peaks, axis_a, axis_b, size_a, size_b, tol=60.0, min_cover=0.3):
    """Keeps height peaks wide enough to be floor or ceiling.

    The top of a cupboard is a perfectly good horizontal plane. If the ceiling was never in view it would
    be taken for the ceiling and the height would come out wrong without any warning.
    """
    heights = cloud @ vertical
    keep = []
    for p in peaks:
        on = cloud[np.abs(heights - p) < tol]
        if _span(on @ axis_a) >= min_cover * size_a and _span(on @ axis_b) >= min_cover * size_b:
            keep.append(p)
    return keep


def fit_room(cloud, tracker):
    if len(cloud) < 300:
        raise RoomError(f"Only {len(cloud)} reliable 3D points were recovered. Blank walls give nothing "
                        "to match: add texture (posters, tape, sticky notes) and move more slowly.")
    planes = extract_planes(cloud)
    down = tracker.keyframes[0]["R"][1]  # image y points down when the phone is held upright
    vertical, axis_a, axis_b = room_axes(planes, down)

    a = _axis_result(axis_a, *surface_peaks(cloud @ axis_a))
    b = _axis_result(axis_b, *surface_peaks(cloud @ axis_b))
    if a["size"] is None and b["size"] is None:
        raise RoomError("Opposite walls were not found in either direction. Pan across the whole room.")
    if (b["size"] or 0) > (a["size"] or 0):
        a, b = b, a

    peaks, centres, hist = surface_peaks(cloud @ vertical)
    if a["size"] and b["size"]:
        peaks = _broad_surfaces(cloud, vertical, peaks, a["axis"], b["axis"], a["size"], b["size"])
    return {"a": a, "b": b, "height": _axis_result(vertical, peaks, centres, hist)}


# ---------------- output figure ----------------

def plot_floor_plan(cloud, tracker, fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length, width, height = fit["a"], fit["b"], fit["height"]
    centres = tracker.centres()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(cloud @ length["axis"], cloud @ width["axis"], s=1, c=cloud @ height["axis"], cmap="viridis", alpha=0.5)
    ax.plot(centres @ length["axis"], centres @ width["axis"], "-o", c="#c9762f", ms=3, lw=1, label="camera path")
    if length["pair"] and width["pair"]:
        (u0, u1), (v0, v1) = length["pair"], width["pair"]
        ax.plot([u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0], "r-", lw=2, label="fitted walls")
    ax.set_aspect("equal")
    ax.set_xlabel("length axis (mm)")
    ax.set_ylabel("width axis (mm)")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------- plain walls: one wall from a known distance ----------------

def _follow_wall(grays):
    """Each frame's motion back to the first one, as 3x3 matrices.

    Plain walls have too little for 3D tracking but enough to follow from one frame to the next. Fitting
    one motion to all the tracks keeps to the wall: things seen through a window move differently and drop out.
    """
    total = np.eye(3)
    motion = [total.copy()]
    flow = {"winSize": (31, 31), "maxLevel": 3}
    for a, b in zip(grays, grays[1:]):
        p = cv2.goodFeaturesToTrack(a, 400, 0.002, 12, blockSize=9)
        if p is not None and len(p) >= 4:
            q, found, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **flow)
            back, found_back, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **flow)
            ok = (found.ravel() == 1) & (found_back.ravel() == 1) & (np.linalg.norm((p - back)[:, 0], axis=1) < 0.5)
            if ok.sum() >= 4:
                step, _ = cv2.estimateAffinePartial2D(q[ok], p[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
                if step is not None:
                    total = total @ np.vstack([step, [0, 0, 1]])
        motion.append(total.copy())
    return motion


def _long_lines(gray):
    """Long straight edges as (near-vertical, near-horizontal), each [(position mid-image, length), ...].

    Horizontal ones must span most of the frame, which is what tells a floor line from a window sill.
    """
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 8, 25)
    found = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60, minLineLength=int(0.2 * h), maxLineGap=25)
    upright, level = [], []
    for x1, y1, x2, y2 in ([] if found is None else found[:, 0, :].astype(np.float64)):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if abs(x2 - x1) < 0.35 * abs(y2 - y1) and 0.05 * w < (x1 + x2) / 2 < 0.95 * w and length > 0.3 * h:
            upright.append(((x1 + x2) / 2, length))
        elif abs(y2 - y1) < 0.25 * abs(x2 - x1) and 0.04 * h < (y1 + y2) / 2 < 0.95 * h and length > 0.7 * w:
            level.append(((y1 + y2) / 2, length))
    return upright, level


def _ceiling_and_floor(grays):
    """The lines where the wall meets ceiling and floor, from the first frame showing both, or None."""
    h = grays[0].shape[0]
    for i in range(0, len(grays), max(1, len(grays) // 30)):
        _, level = _long_lines(grays[i])
        top = [y for y, _ in level if y < 0.35 * h]
        bottom = [y for y, _ in level if y > 0.65 * h]
        if top and bottom:
            return i, min(top), max(bottom)
    return None


def measure_wall(frames, K, wall_mm, path):
    """Wall width and height when the walls are too plain for the full room: one pixel on the wall is Z / f mm.

    The camera's view is followed along the wall, and the wall's two ends are the room corners in the first
    and last frames. Without a corner in view the wall carries on past the frame, so the width is a minimum.
    """
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for _, img in frames]
    motion = _follow_wall(grays)
    h, w = grays[0].shape
    mm_per_px = wall_mm / K[0, 0]

    forwards = motion[-1][0, 2] > 0  # the view moved right, so the scene slid left
    corners_first, corners_last = _long_lines(grays[0])[0], _long_lines(grays[-1])[0]
    pick = (min, max) if forwards else (max, min)
    start = pick[0]((x for x, _ in corners_first), default=None)
    end = pick[1]((x for x, _ in corners_last), default=None)
    start_x = (0.0 if forwards else float(w)) if start is None else start
    end_x = (float(w) if forwards else 0.0) if end is None else end
    width = abs((motion[-1] @ [end_x, h / 2, 1])[0] - start_x) * mm_per_px

    levels = _ceiling_and_floor(grays)
    height = (levels[2] - levels[1]) * mm_per_px if levels else None

    first, last = frames[0][1].copy(), frames[-1][1].copy()
    cv2.line(first, (int(start_x), 0), (int(start_x), h), (0, 0, 255), 3)
    cv2.line(last, (int(end_x), 0), (int(end_x), h), (0, 0, 255), 3)
    if levels:
        shown = frames[levels[0]][1].copy()
        for y in levels[1:]:
            cv2.line(shown, (0, int(y)), (w, int(y)), (0, 255, 0), 3)
    else:
        shown = np.zeros_like(first)
    cv2.imwrite(path, np.hstack([first, last, shown]))
    return {"width": width, "height": height, "complete": start is not None and end is not None}


# ---------------- the whole pipeline ----------------

def run(frames, K, slide_mm, slide_end_s):
    """frames: [(t, img), ...] already undistorted, so K describes them directly."""
    tracker = Tracker(K)
    start = tracker.start(frames, slide_end_s)
    for index in range(start + 1, len(frames)):
        if tracker.add(index, *frames[index]) and len(tracker.keyframes) % 6 == 0:
            tracker.refine_recent()
    check_tracking(tracker, len(frames) - start)
    tracker.rescale(slide_mm * (frames[start][0] - frames[0][0]) / max(slide_end_s - frames[0][0], 1e-6))
    close_loop(tracker)
    images = [frames[kf["index"]][1] for kf in tracker.keyframes]
    return tracker, dense_cloud(tracker, images)
