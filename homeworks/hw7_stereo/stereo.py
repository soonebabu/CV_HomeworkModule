"""Object size from calibrated stereo.

For a rectified pair every clicked point gets its own distance, Z = f * B / d, and with it a 3D position:

    X = (x - cx) * Z / f      Y = (y - cy) * Z / f

Sizes are then plain distances between 3D points.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

WORK_DIM = 1280  # phone photos are 4000 px; this is plenty and keeps SGBM fast
POSE_DIM = 2400  # larger for the pose: plain scenes need the detail. SIFT at 4000 px takes several GB of RAM
POSE_STARTS = 10
MAX_FEATURES = 12000  # a busy photo can give 100k+ with the lenient detector, too many to match in memory


class StereoError(ValueError):
    """Something about the input stops the pair being used. The message is shown to the user."""


def resize_max(img, max_dim=WORK_DIM):
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def fit_camera_to_image(camera_matrix, calib_size, image_size):
    """Adapts K to another resolution or aspect ratio, assuming the sensor was scaled then centre-cropped.

    This is what phones do for 16:9 video from a 4:3 sensor. Plain per-axis scaling (scale_camera_matrix)
    would squash the pixels, so it only suits images of the same aspect ratio.
    """
    calib_w, calib_h = calib_size
    w, h = image_size
    if (calib_w > calib_h) != (w > h):
        raise StereoError(f"The camera was calibrated at {calib_w} x {calib_h} but this image is {w} x {h}. "
                          "Rotate the image to the same orientation (portrait or landscape).")
    s = max(w / calib_w, h / calib_h)
    K = np.array(camera_matrix, dtype=np.float64)
    K[:2] *= s
    K[0, 2] -= (calib_w * s - w) / 2
    K[1, 2] -= (calib_h * s - h) / 2
    return K


def _sift_matches(gray1, gray2, ratio=0.75, nfeatures=4000, contrast=0.04):
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    return pts1, pts2


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _normalize(pts, K, dist):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist).reshape(-1, 2)


def pose_from_images(img1, img2, K1, dist1, K2, dist2, min_inliers=25):
    """R and unit t with X2 = R X1 + t, from the essential matrix of the two views.

    Takes the photos at POSE_DIM with a lenient detector: plain walls and desks leave too few features
    once shrunk, and the pose that comes out of those few can be badly wrong.

    With so few good matches a single RANSAC run is unreliable. So the RANSAC runs, and a plain sideways move
    as well, are each polished against all the matches and the one most matches agree with is kept,
    preferring a sideways result since that is the only motion this method allows.
    """
    pts1, pts2 = _sift_matches(_gray(img1), _gray(img2), ratio=0.85, nfeatures=MAX_FEATURES, contrast=0.01)
    if len(pts1) < min_inliers:
        raise StereoError(f"Only {len(pts1)} feature matches between the two images - they need to overlap "
                          "and show some texture.")
    direction = 1.0 if np.median(pts2[:, 0] - pts1[:, 0]) > 0 else -1.0  # scene moves opposite to the camera
    starts = [(np.eye(3), np.array([direction, 0.0, 0.0]))]
    rng = np.random.default_rng(0)
    for _ in range(POSE_STARTS):
        order = rng.permutation(len(pts1))
        try:
            starts.append(pose_from_matches(pts1[order], pts2[order], K1, dist1, K2, dist2, min_inliers)[:2])
        except StereoError:
            pass

    best, best_score = None, None
    for R, t in starts:
        R, t, agree = _polish_pose(R, t, pts1, pts2, K1, dist1, K2, dist2)
        score = (abs(t[0]) > 0.9, agree)
        if best is None or score > best_score:
            best, best_score = (R, t, agree), score
    if best[2] < min_inliers:
        raise StereoError(f"Only {best[2]} matches agree on the camera motion - try a pair with more "
                          "overlap or a slightly larger sideways move.")
    return best


def _sampson(R, t, rays1, rays2):
    """Distance of each match from its epipolar line, in normalized image units."""
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    a, b = rays1 @ E.T, rays2 @ E
    return np.sum(rays2 * a, axis=1) / np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2 + b[:, 0] ** 2 + b[:, 1] ** 2)


def _polish_pose(R, t, pts1, pts2, K1, dist1, K2, dist2):
    """Robust least squares on R and the direction of t. Returns R, unit t and the matches within 1.5 px."""
    rays1 = np.column_stack([_normalize(pts1, K1, dist1), np.ones(len(pts1))])
    rays2 = np.column_stack([_normalize(pts2, K2, dist2), np.ones(len(pts2))])
    pixel = 1.0 / K1[0, 0]
    rvec = cv2.Rodrigues(R)[0].ravel()
    tangents = np.linalg.svd(t[None])[2][1:]  # t only has a direction, so it moves in two ways

    def pose(x):
        moved = t + tangents.T @ x[3:]
        return cv2.Rodrigues(rvec + x[:3])[0], moved / np.linalg.norm(moved)

    x = least_squares(lambda x: _sampson(*pose(x), rays1, rays2) / pixel, np.zeros(5),
                      loss="cauchy", x_scale=1e-2).x
    R, t = pose(x)
    return R, t, int(np.sum(np.abs(_sampson(R, t, rays1, rays2)) < 1.5 * pixel))


def pose_from_matches(pts1, pts2, K1, dist1, K2, dist2, min_inliers=25):
    """The same, for matches that are already found: pts1[i] in view 1 is pts2[i] in view 2."""
    if len(pts1) < min_inliers:
        raise StereoError(f"Only {len(pts1)} feature matches between the two images - they need to overlap "
                          "and show some texture.")
    n1, n2 = _normalize(pts1, K1, dist1), _normalize(pts2, K2, dist2)
    thresh = 1.0 / K1[0, 0]  # about one pixel, in normalized image units
    E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, thresh)
    if E is None or E.shape != (3, 3):
        raise StereoError("Could not estimate the camera motion between the two images.")
    inliers, R, t, _ = cv2.recoverPose(E, n1, n2, np.eye(3), mask=mask)
    if inliers < min_inliers:
        raise StereoError(f"Only {inliers} matches agree on the camera motion - try a pair with more "
                          "overlap or a slightly larger sideways move.")
    planar = _planar_pose(n1, n2, thresh, inliers)
    if planar is not None:
        R, t = planar
    return R, t.ravel(), int(inliers)


def _planar_pose(n1, n2, thresh, e_inliers):
    """Motion for a flat scene, or None if the scene isn't flat.

    All matches on one plane (a wall, a tabletop) make the essential matrix ambiguous: there are two
    valid answers, and recoverPose can return one with the wrong direction. A homography explains such a
    scene just as well, so when it does, decompose it and keep the sideways solution, which is the only
    motion this method allows.
    """
    H, mask = cv2.findHomography(n1, n2, cv2.RANSAC, thresh)
    if H is None or mask.sum() < 0.9 * e_inliers:
        return None
    _, rotations, translations, normals = cv2.decomposeHomographyMat(H, np.eye(3))
    rays = np.column_stack([n1[mask.ravel() > 0], np.ones(int(mask.sum()))])
    best, best_score = None, -1.0
    for R, t, n in zip(rotations, translations, normals):
        norm = np.linalg.norm(t)
        if norm < 1e-9 or np.mean(rays @ n > 0) < 0.9:  # the plane must be in front of the camera
            continue
        score = abs(t[0, 0]) / norm
        if score > best_score:
            best, best_score = (R, t.ravel() / norm), score
    return best


def make_rig(K1, dist1, K2, dist2, R, T):
    """Puts the pair in left/right order. stereoRectify needs camera 2 on the right, i.e. T[0] < 0."""
    R, T = np.asarray(R, dtype=np.float64), np.asarray(T, dtype=np.float64).ravel()
    t = T / np.linalg.norm(T)
    if abs(t[1]) > abs(t[0]) or abs(t[2]) > 0.8:
        raise StereoError("The camera moved mostly up/down or forwards between the views. Stereo here "
                          "needs a sideways move: keep the phone at the same height and facing the same way.")
    swapped = T[0] > 0
    if swapped:
        K1, dist1, K2, dist2 = K2, dist2, K1, dist1
        R, T = R.T, -R.T @ T
    return {"K1": K1, "dist1": dist1, "K2": K2, "dist2": dist2, "R": R, "T": T, "swapped": bool(swapped)}


def rectify_maps(rig, size, alpha=0):
    """alpha=0 crops to the area both views share, alpha=1 keeps every pixel of both photos."""
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        rig["K1"], rig["dist1"], rig["K2"], rig["dist2"], size, rig["R"], rig["T"],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    maps1 = cv2.initUndistortRectifyMap(rig["K1"], rig["dist1"], R1, P1, size, cv2.CV_16SC2)
    maps2 = cv2.initUndistortRectifyMap(rig["K2"], rig["dist2"], R2, P2, size, cv2.CV_16SC2)
    f, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
    return {"maps1": maps1, "maps2": maps2, "f": float(f), "cx": float(cx), "cy": float(cy),
            "baseline": float(-P2[0, 3] / f), "R1": R1}


def apply_rectify(left, right, rect):
    return (cv2.remap(left, *rect["maps1"], cv2.INTER_LINEAR),
            cv2.remap(right, *rect["maps2"], cv2.INTER_LINEAR))


def disparity_range(left, right, plain_scene=False):
    """Search range for SGBM, read off sparse matches so the user never has to guess it.

    plain_scene looks harder for matches and searches further, because in a photo of a plain object the
    features sit on the background and the nearer object lies beyond them.
    """
    if plain_scene:
        pts1, pts2 = _sift_matches(_gray(left), _gray(right), ratio=0.85, nfeatures=MAX_FEATURES, contrast=0.01)
    else:
        pts1, pts2 = _sift_matches(_gray(left), _gray(right), nfeatures=2000)
    if len(pts1):
        same_row = np.abs(pts1[:, 1] - pts2[:, 1]) < 2.0
        d = (pts1[:, 0] - pts2[:, 0])[same_row]
        d = d[d > 0]
    else:
        d = np.empty(0)
    if len(d) < 12:
        return 0, 128
    lo = max(0, int(np.percentile(d, 1) * 0.8) - 8)
    hi = int(np.percentile(d, 99) * (1.5 if plain_scene else 1.2)) + (16 if plain_scene else 8)
    return lo, min(hi, lo + left.shape[1] // 2)


def compute_disparity(left, right, block=5, search=None):
    """Sub-pixel SGBM disparity on a rectified pair. Invalid pixels are NaN."""
    lo, hi = search or disparity_range(left, right)
    num = max(16, int(np.ceil((hi - lo) / 16.0)) * 16)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=lo, numDisparities=num, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=120, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(_gray(left), _gray(right)).astype(np.float32) / 16.0
    disp[disp < lo] = np.nan
    return disp


def textured_mask(gray, min_std=3.0, win=7):
    """SGBM happily invents depth on blank paint; keep only pixels with some local contrast."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (win, win))
    var = cv2.blur(g * g, (win, win)) - mean * mean
    return var > min_std * min_std


def match_along_row(left, right, x, y, half=12, search=(0, 400)):
    """Finds (x, y) from the left image on the same row of the right one by normalized correlation."""
    gl, gr = _gray(left), _gray(right)
    h, w = gl.shape
    xi, yi = int(round(x)), int(round(y))
    if not (half <= xi < w - half and half <= yi < h - half):
        return None, 0.0
    patch = gl[yi - half:yi + half + 1, xi - half:xi + half + 1]
    if patch.std() < 2.0:
        return None, 0.0
    x_hi = min(w, xi + half + 1 - search[0])
    x_lo = max(0, xi - half - search[1])
    if x_hi - x_lo < 2 * half + 2:
        return None, 0.0
    strip = gr[yi - half:yi + half + 1, x_lo:x_hi]
    scores = cv2.matchTemplate(strip, patch, cv2.TM_CCOEFF_NORMED).ravel()
    k = int(np.argmax(scores))
    offset = 0.0
    if 0 < k < len(scores) - 1:  # parabola through the peak gives the sub-pixel position
        denom = scores[k - 1] - 2 * scores[k] + scores[k + 1]
        if denom < 0:
            offset = 0.5 * (scores[k - 1] - scores[k + 1]) / denom
    x_right = x_lo + half + k + offset
    return float(x - x_right), float(scores[k])


def detect_object(img):
    """Finds the main object in a rectified image: returns (kind, outline points, contour) or None.

    The object is the largest dark or bright region that stays clear of the frame. Walls and tables run off
    the edge of the photo, so they drop out by themselves.
    """
    gray = _gray(img)
    h, w = gray.shape
    in_frame = cv2.morphologyEx((gray > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    in_frame[[0, -1], :] = 0
    in_frame[:, [0, -1]] = 0
    clear = cv2.erode(in_frame, np.ones((15, 15), np.uint8)).astype(bool)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    level, _ = cv2.threshold(blurred[in_frame > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best, best_area = None, 0.0
    for region in (blurred < level, blurred >= level):
        mask = cv2.morphologyEx((region & (in_frame > 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            xy = contour[:, 0, :]
            if 0.005 * h * w < area < 0.7 * h * w and area > best_area and clear[xy[:, 1], xy[:, 0]].all():
                best, best_area = contour, area
    if best is None:
        return None

    perimeter = cv2.arcLength(best, True)
    corners = cv2.approxPolyDP(best, 0.02 * perimeter, True)[:, 0, :].astype(np.float64)
    if 4 * np.pi * best_area / perimeter ** 2 > 0.87 and len(corners) > 4:
        step = max(1, len(best) // 24)
        return "circle", best[::step, 0, :].astype(np.float64), best
    if len(corners) == 4 and cv2.isContourConvex(corners.astype(np.int32).reshape(-1, 1, 2)):
        sums, diffs = corners.sum(axis=1), corners[:, 0] - corners[:, 1]
        return "rectangle", corners[[sums.argmin(), diffs.argmax(), sums.argmax(), diffs.argmin()]], best
    return "polygon", corners, best


def match_outline(points, contour_left, contour_right):
    """Each outline point's partner on the same object in the right image, or None if it isn't the same object.

    The whole shape moved by about the shift of its centre, so undo that shift and take the nearest point
    of the right outline. This needs no texture on the object, only that it stands out in both images.
    """
    def centre(contour):
        m = cv2.moments(contour)
        return np.array([m["m10"], m["m01"]]) / m["m00"], m["m00"]

    (c_left, area_left), (c_right, area_right) = centre(contour_left), centre(contour_right)
    shift = c_left - c_right
    if shift[0] <= 0 or abs(shift[1]) > 12 or not 0.6 < area_left / area_right < 1.6:
        return None
    outline = contour_right[:, 0, :].astype(np.float64)
    nearest = [outline[np.argmin(np.sum((outline - (p - shift)) ** 2, axis=1))] for p in points]
    return np.array(nearest)


def _polygon_mask(shape, pts):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def fit_disparity_plane(x, y, d, tol=1.0, iters=300, seed=0):
    """A flat surface has disparity linear in the image: d = a x + b y + c. RANSAC, then least squares."""
    if len(d) < 3:
        return None, 0
    rng = np.random.default_rng(seed)
    A = np.column_stack([x, y, np.ones_like(x)])
    best_in, best_n = None, 0
    for _ in range(iters):
        idx = rng.choice(len(d), 3, replace=False)
        try:
            coef = np.linalg.solve(A[idx], d[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(A @ coef - d) < tol
        n = int(inl.sum())
        if n > best_n:
            best_in, best_n = inl, n
    if best_in is None or best_n < 3:
        return None, 0
    coef, *_ = np.linalg.lstsq(A[best_in], d[best_in], rcond=None)
    return coef, best_n


def point_disparities(points, left, right, disp, manual_right=None):
    """Disparity for each clicked point.

    Clicked corners sit right on the object's edge, the worst place for block matching because half the
    window sees the background. So the object's face is treated as flat: fit the disparity plane to the
    textured pixels inside the outline, plus the corners that matched well, and read each corner off the
    plane. A blank face (a switched-off screen, plain paint) gives SGBM nothing to work with, so its pixels
    are not used. Clicks on the right image, when given, are used as they are.
    """
    pts = np.asarray(points, dtype=np.float64)
    if manual_right is not None:
        return pts[:, 0] - np.asarray(manual_right, dtype=np.float64)[:, 0]

    valid = disp[np.isfinite(disp)]
    search = (max(0, int(np.percentile(valid, 1)) - 10), int(np.percentile(valid, 99)) + 10) \
        if valid.size > 50 else (0, 400)
    xs, ds = [], []
    for x, y in pts:
        d, score = match_along_row(left, right, x, y, search=search)
        if d is not None and score > 0.75:
            xs.append((x, y))
            ds.append(d)

    inside = _polygon_mask(disp.shape, pts) if len(pts) >= 3 else np.zeros(disp.shape, bool)
    inside &= np.isfinite(disp) & textured_mask(_gray(left))
    yy, xx = np.nonzero(inside)
    if len(xx) > 4000:
        keep = np.random.default_rng(0).choice(len(xx), 4000, replace=False)
        yy, xx = yy[keep], xx[keep]
    x_all = np.concatenate([xx.astype(np.float64), np.array([p[0] for p in xs], dtype=np.float64)])
    y_all = np.concatenate([yy.astype(np.float64), np.array([p[1] for p in xs], dtype=np.float64)])
    d_all = np.concatenate([disp[yy, xx].astype(np.float64), np.array(ds, dtype=np.float64)])

    coef, n_in = fit_disparity_plane(x_all, y_all, d_all)
    if coef is not None and n_in >= 30:
        return coef[0] * pts[:, 0] + coef[1] * pts[:, 1] + coef[2]
    if ds:
        return np.full(len(pts), float(np.median(ds)))
    raise StereoError("No reliable match for these points. Click the same points on the right image too.")


def to_3d(points, disparities, rect):
    pts = np.asarray(points, dtype=np.float64)
    d = np.asarray(disparities, dtype=np.float64)
    if np.any(d <= 0):
        raise StereoError("A point got a zero or negative disparity, so its depth is undefined. Check the "
                          "matches, or click the points on the right image too.")
    Z = rect["f"] * rect["baseline"] / d
    X = (pts[:, 0] - rect["cx"]) * Z / rect["f"]
    Y = (pts[:, 1] - rect["cy"]) * Z / rect["f"]
    return np.column_stack([X, Y, Z])


def fit_circle_3d(xyz):
    """Plane through the rim points by SVD, then a circle fit (Kasa) inside that plane. Returns the radius."""
    centre = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - centre)
    u, v = vt[0], vt[1]
    p = np.column_stack([(xyz - centre) @ u, (xyz - centre) @ v])
    A = np.column_stack([2 * p, np.ones(len(p))])
    sol, *_ = np.linalg.lstsq(A, (p ** 2).sum(axis=1), rcond=None)
    return float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))


def measure_shape(kind, xyz, points_px):
    """The dimensions the shape asks for: returns (headline [(label, mm), ...], edge lengths in mm)."""
    n = len(xyz)
    edges = [float(np.linalg.norm(xyz[(i + 1) % n] - xyz[i])) for i in range(n)]

    if kind == "circle":
        if n < 3:
            raise StereoError("A circle needs at least 3 points on its rim.")
        return [("Diameter", 2 * fit_circle_3d(xyz))], []

    if kind == "rectangle":
        if n != 4:
            raise StereoError("A rectangle needs exactly 4 corners, clicked in order.")
        side_a, side_b = (edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2
        # whichever pair of edges runs more horizontally in the image is the width
        d0 = np.asarray(points_px[1], dtype=np.float64) - np.asarray(points_px[0], dtype=np.float64)
        width, height = (side_a, side_b) if abs(d0[0]) >= abs(d0[1]) else (side_b, side_a)
        return [("Width", width), ("Height / length", height)], edges

    if n < 3:
        raise StereoError("A polygon needs at least 3 vertices.")
    return [], edges
