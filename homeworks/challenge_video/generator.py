"""Stage 4b - Video generator (offline stand-in).

A real diffusion video model needs a GPU, which neither this laptop nor the hosted app has. To still test the
pipeline end to end, this module is a procedural generator with the same contract as a conditioned
text-to-video model:

  * it draws exactly what the conditioning describes (setting, identities, appearance, boxes, motion), and
  * whatever the conditioning leaves open is filled in from the clip's own random seed, the way a diffusion
    model invents unspecified details for every new sample.

The baseline and the orchestrator call this same function. The only thing that changes between them is the
conditioning, which is the variable the prototype is meant to isolate.

Alongside the frames it returns a per-pixel instance label map for every frame (which entity id drew each
pixel). The evaluation uses those maps where a real pipeline would run a detector / tracker such as
Grounding-DINO + SAM2. Pixel colours, positions and relations are all measured from the rendered frames.
"""
import math
import random
import zlib

import cv2
import numpy as np

from .planner import ANIMAL_KINDS, HUMAN_KINDS

W, H = 480, 270

COLOR_BGR = {
    "red": (55, 55, 205), "blue": (205, 115, 45), "green": (70, 165, 70), "yellow": (45, 205, 240),
    "orange": (35, 140, 245), "purple": (160, 70, 135), "pink": (185, 135, 245), "black": (40, 38, 38),
    "white": (238, 238, 238), "brown": (45, 85, 140), "gray": (145, 145, 145), "teal": (150, 150, 35),
}
SKIN = [(150, 190, 230), (110, 160, 210), (80, 120, 170), (60, 90, 130)]
LOC_STYLE = {  # sky/wall top, sky/wall bottom, ground, horizon (fraction of H)
    "park": ((235, 205, 150), (245, 230, 200), (80, 160, 95), 0.58),
    "beach": ((240, 210, 140), (250, 235, 200), (150, 210, 235), 0.6),
    "street": ((225, 205, 175), (240, 230, 215), (110, 110, 115), 0.62),
    "forest": ((150, 170, 120), (170, 190, 150), (50, 100, 70), 0.6),
    "kitchen": ((185, 215, 235), (175, 205, 228), (150, 165, 180), 0.62),
    "room": ((200, 210, 225), (190, 200, 215), (75, 110, 150), 0.63),
    "office": ((215, 205, 195), (205, 195, 185), (115, 115, 120), 0.63),
    "cafe": ((135, 175, 215), (125, 165, 205), (60, 90, 120), 0.63),
}


def color(name, fallback=(150, 150, 150)):
    return COLOR_BGR.get(name, fallback)


def shade(c, f):
    return tuple(int(max(0, min(255, v * f))) for v in c)


class Painter:
    """Draws every primitive twice: anti-aliased colour into the frame, flat instance id into the label map."""

    def __init__(self, img, lab):
        self.img, self.lab = img, lab
        self.c, self.i = (0, 0, 0), 0

    def use(self, c, i=None):
        self.c = tuple(int(v) for v in c)
        if i is not None:
            self.i = i
        return self

    def circle(self, p, r):
        p, r = _pt(p), max(1, int(round(r)))
        cv2.circle(self.img, p, r, self.c, -1, cv2.LINE_AA)
        cv2.circle(self.lab, p, r, self.i, -1)

    def ellipse(self, p, ax, ang=0, a0=0, a1=360):
        p, ax = _pt(p), (max(1, int(round(ax[0]))), max(1, int(round(ax[1]))))
        cv2.ellipse(self.img, p, ax, ang, a0, a1, self.c, -1, cv2.LINE_AA)
        cv2.ellipse(self.lab, p, ax, ang, a0, a1, self.i, -1)

    def rect(self, p0, p1):
        self.poly([(p0[0], p0[1]), (p1[0], p0[1]), (p1[0], p1[1]), (p0[0], p1[1])])

    def poly(self, pts):
        pts = np.array([_pt(p) for p in pts], np.int32)
        cv2.fillPoly(self.img, [pts], self.c, cv2.LINE_AA)
        cv2.fillPoly(self.lab, [pts], self.i)

    def line(self, p0, p1, t):
        t = max(1, int(round(t)))
        cv2.line(self.img, _pt(p0), _pt(p1), self.c, t, cv2.LINE_AA)
        cv2.line(self.lab, _pt(p0), _pt(p1), self.i, t)


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


# ---------------------------------------------------------------------------------------------------------
# backgrounds

def background(bg, frame_idx):
    loc = bg["location"] if bg["location"] in LOC_STYLE else "park"
    top, bottom, ground, hz = LOC_STYLE[loc]
    rng = random.Random(bg["seed"])
    img = np.zeros((H, W, 3), np.uint8)
    hy = int(H * hz)
    for y in range(hy):
        t = y / max(1, hy - 1)
        img[y, :] = [int(top[k] * (1 - t) + bottom[k] * t) for k in range(3)]
    img[hy:, :] = ground
    indoors = loc in ("kitchen", "room", "office", "cafe")
    night = bg["time_of_day"] == "night"
    rainy = bg["weather"] in ("rain", "snow") and not indoors

    if loc == "park":
        for _ in range(rng.randint(2, 4)):
            cx, cy = rng.randint(30, W - 30), rng.randint(20, hy - 70)
            for dx in (-18, 0, 18):
                cv2.ellipse(img, (cx + dx, cy), (22, 12), 0, 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.ellipse(img, (rng.randint(0, W), hy + 4), (rng.randint(160, 260), 45), 0, 180, 360, (95, 170, 110), -1,
                    cv2.LINE_AA)
        for _ in range(rng.randint(2, 4)):
            bx = rng.randint(10, W - 10)
            cv2.ellipse(img, (bx, hy), (rng.randint(18, 30), rng.randint(14, 22)), 0, 180, 360, (60, 130, 60), -1,
                        cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([(W // 2 - 30, hy), (W // 2 + 30, hy), (W // 2 + 120, H), (W // 2 - 120, H)])],
                     (150, 190, 205), cv2.LINE_AA)
    elif loc == "beach":
        cv2.rectangle(img, (0, hy - 30), (W, hy), (190, 150, 60), -1)
        for _ in range(6):
            x = rng.randint(0, W)
            cv2.line(img, (x, hy - 20), (x + 25, hy - 20), (230, 200, 140), 2, cv2.LINE_AA)
        if rng.random() < 0.7:
            px = rng.randint(30, W - 30)
            cv2.line(img, (px, hy + 10), (px + 8, hy - 70), (40, 80, 120), 6, cv2.LINE_AA)
            for a in range(0, 360, 60):
                cv2.ellipse(img, (px + 8, hy - 72), (28, 7), a, 0, 360, (50, 140, 60), -1, cv2.LINE_AA)
    elif loc == "street":
        x = 0
        while x < W:
            bw, bh = rng.randint(50, 90), rng.randint(60, hy - 20)
            col = rng.choice([(120, 140, 170), (150, 150, 160), (100, 120, 150), (170, 160, 140)])
            cv2.rectangle(img, (x, hy - bh), (x + bw - 4, hy), col, -1)
            for wy in range(hy - bh + 10, hy - 12, 18):
                for wx in range(x + 8, x + bw - 14, 16):
                    cv2.rectangle(img, (wx, wy), (wx + 8, wy + 10), (215, 230, 240), -1)
            x += bw
        cv2.rectangle(img, (0, hy), (W, hy + 14), (170, 170, 175), -1)
        for x in range(10, W, 60):
            cv2.rectangle(img, (x, int(H * 0.86)), (x + 30, int(H * 0.86) + 4), (230, 230, 230), -1)
    elif loc == "forest":
        for _ in range(rng.randint(6, 9)):
            tx, tw = rng.randint(0, W), rng.randint(10, 22)
            cv2.rectangle(img, (tx, rng.randint(0, 30)), (tx + tw, hy + 6), (40, 65, 90), -1)
            cv2.circle(img, (tx + tw // 2, rng.randint(10, 60)), rng.randint(35, 55), (45, 120, 60), -1, cv2.LINE_AA)
    else:
        # indoor: a window, a picture, furniture along the back wall, a skirting board and floor pattern
        wx = rng.randint(30, W - 130)
        cv2.rectangle(img, (wx, 30), (wx + 90, 110), (90, 70, 60), -1)
        win = (70, 50, 40) if night else (235, 205, 160)
        cv2.rectangle(img, (wx + 6, 36), (wx + 84, 104), win, -1)
        cv2.line(img, (wx + 45, 36), (wx + 45, 104), (90, 70, 60), 3)
        px = (wx + 160 + rng.randint(0, 120)) % (W - 60)
        cv2.rectangle(img, (px, 45), (px + 50, 85), (60, 60, 70), -1)
        cv2.rectangle(img, (px + 4, 49), (px + 46, 81), rng.choice([(80, 150, 200), (150, 100, 80), (90, 160, 90)]), -1)
        if loc == "kitchen":
            cx = rng.randint(0, W // 2)
            cv2.rectangle(img, (cx, hy - 55), (cx + 200, hy), (120, 160, 190), -1)
            cv2.rectangle(img, (cx - 4, hy - 60), (cx + 204, hy - 52), (200, 200, 205), -1)
            for k in range(4):
                cv2.rectangle(img, (cx + 6 + k * 50, hy - 46), (cx + 46 + k * 50, hy - 6), (100, 140, 170), 2)
        elif loc in ("office", "cafe"):
            sx = rng.randint(0, W - 120)
            cv2.rectangle(img, (sx, hy - 90), (sx + 110, hy), (60, 90, 120), -1)
            for k in range(3):
                cv2.line(img, (sx, hy - 30 - 28 * k), (sx + 110, hy - 30 - 28 * k), (40, 60, 80), 3)
                for bx in range(sx + 6, sx + 104, 9):
                    cv2.rectangle(img, (bx, hy - 52 - 28 * k), (bx + 6, hy - 32 - 28 * k),
                                  rng.choice([(60, 60, 180), (170, 100, 50), (60, 150, 60), (40, 170, 210)]), -1)
        cv2.rectangle(img, (0, hy), (W, hy + 5), shade(top, 0.75), -1)
        step = 40
        for x in range(-W, W * 2, step):
            cv2.line(img, (W // 2 + (x - W // 2) // 3, hy + 5), (x, H), shade(ground, 0.88), 1, cv2.LINE_AA)
        if loc != "room":
            for y in (hy + 25, hy + 60):
                cv2.line(img, (0, y), (W, y), shade(ground, 0.88), 1)
        else:
            rx = rng.randint(80, W - 200)
            cv2.ellipse(img, (rx + 60, int(H * 0.86)), (110, 18), 0, 0, 360, (90, 70, 150), -1, cv2.LINE_AA)

    if bg["time_of_day"] == "sunset" and not indoors:
        tint = np.zeros_like(img)
        tint[:] = (60, 120, 230)
        img = cv2.addWeighted(img, 0.7, tint, 0.3, 0)
    if night:
        img = (img.astype(np.float32) * (0.62 if indoors else 0.45) + np.array([25, 8, 0])).clip(0, 255).astype(np.uint8)
        if not indoors:
            cv2.circle(img, (W - 60, 40), 16, (215, 235, 240), -1, cv2.LINE_AA)
    if rainy:
        img = (img.astype(np.float32) * 0.78 + np.array([22, 18, 12])).clip(0, 255).astype(np.uint8)
    return img


def weather_overlay(img, bg, frame_idx):
    loc = bg["location"]
    if loc in ("kitchen", "room", "office", "cafe") or bg["weather"] not in ("rain", "snow"):
        return img
    rng = random.Random(bg["seed"] * 31 + frame_idx)
    out = img.copy()
    if bg["weather"] == "rain":
        for _ in range(120):
            x, y = rng.randint(0, W), rng.randint(-20, H)
            cv2.line(out, (x, y), (x - 4, y + 14), (215, 205, 195), 1, cv2.LINE_AA)
        return cv2.addWeighted(img, 0.45, out, 0.55, 0)
    for _ in range(90):
        cv2.circle(out, (rng.randint(0, W), rng.randint(0, H)), rng.randint(1, 3), (250, 250, 250), -1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------------------------------------
# characters and objects

def draw_human(p, cx, yb, h, a, pose, phase, motion, facing, idx, seed):
    skin = SKIN[seed % len(SKIN)]
    cloth = color(a.get("clothing_color"), (120, 120, 120))
    hair = color(a.get("hair_color"), (40, 40, 40))
    pants = (70, 55, 50) if a.get("clothing") != "dress" else skin
    long_hair = a.get("_long_hair", False)
    swing = math.sin(phase * 2 * math.pi * 2) * (0.07 * h if motion in ("walk", "run") else 0)
    arm_up = motion == "wave" and math.sin(phase * 2 * math.pi * 3) > -0.3
    if pose == "sleep":
        y = yb - 0.08 * h
        p.use(cloth, idx).ellipse((cx, y), (0.32 * h, 0.07 * h))
        p.use(skin).circle((cx - 0.38 * h, y - 0.02 * h), 0.08 * h)
        p.use(hair).ellipse((cx - 0.42 * h, y - 0.04 * h), (0.07 * h, 0.07 * h))
        return
    if pose == "sit":
        hip_y = yb
        p.use(pants, idx).line((cx - 0.02 * h, hip_y), (cx + facing * 0.16 * h, hip_y), 0.08 * h)
        p.use(pants).line((cx + facing * 0.16 * h, hip_y), (cx + facing * 0.17 * h, hip_y + 0.2 * h), 0.07 * h)
        top = hip_y - 0.38 * h
        base = hip_y
    else:
        hip_y = yb - 0.42 * h
        p.use(pants, idx).line((cx - 0.04 * h, hip_y), (cx - 0.05 * h + swing, yb - 0.02 * h), 0.075 * h)
        p.use(pants).line((cx + 0.04 * h, hip_y), (cx + 0.05 * h - swing, yb - 0.02 * h), 0.075 * h)
        p.use(shade(pants, 0.6)).ellipse((cx - 0.05 * h + swing + facing * 0.02 * h, yb - 0.02 * h), (0.05 * h, 0.022 * h))
        p.use(shade(pants, 0.6)).ellipse((cx + 0.05 * h - swing + facing * 0.02 * h, yb - 0.02 * h), (0.05 * h, 0.022 * h))
        top = yb - 0.8 * h
        base = hip_y + 0.03 * h
    if a.get("clothing") == "dress" and pose != "sit":
        p.use(cloth, idx).poly([(cx - 0.1 * h, top), (cx + 0.1 * h, top), (cx + 0.17 * h, yb - 0.26 * h),
                                (cx - 0.17 * h, yb - 0.26 * h)])
    else:
        p.use(cloth, idx).poly([(cx - 0.12 * h, top + 0.03 * h), (cx + 0.12 * h, top + 0.03 * h),
                                (cx + 0.11 * h, base), (cx - 0.11 * h, base)])
        p.use(cloth).circle((cx - 0.09 * h, top + 0.05 * h), 0.04 * h)
        p.use(cloth).circle((cx + 0.09 * h, top + 0.05 * h), 0.04 * h)
    # arms: the front arm reaches out to where held things go
    sh = top + 0.06 * h
    back_hand = (cx - facing * 0.15 * h - swing * 0.6, sh + 0.3 * h)
    front_hand = (cx + facing * 0.2 * h, sh - 0.22 * h) if arm_up else (cx + facing * 0.19 * h + swing * 0.6, sh + 0.24 * h)
    p.use(shade(cloth, 0.85)).line((cx - facing * 0.11 * h, sh), back_hand, 0.06 * h)
    p.use(shade(cloth, 0.92)).line((cx + facing * 0.11 * h, sh), front_hand, 0.06 * h)
    p.use(skin).circle(back_hand, 0.032 * h)
    p.use(skin).circle(front_hand, 0.032 * h)
    hc = (cx, top - 0.1 * h)
    p.use(skin).circle((cx, top - 0.01 * h), 0.04 * h)
    if long_hair:
        p.use(hair).ellipse((cx, hc[1] + 0.06 * h), (0.11 * h, 0.15 * h))
    p.use(skin).circle(hc, 0.095 * h)
    p.use(hair).ellipse((hc[0], hc[1] - 0.03 * h), (0.1 * h, 0.075 * h), 0, 180, 360)
    p.use(hair).ellipse((hc[0] - facing * 0.05 * h, hc[1] - 0.02 * h), (0.055 * h, 0.06 * h))
    p.use((40, 40, 40)).circle((hc[0] + facing * 0.045 * h, hc[1] + 0.005 * h), 0.012 * h)


def draw_robot(p, cx, yb, h, a, pose, phase, motion, facing, idx):
    body = color(a.get("color"), (150, 150, 150))
    bob = abs(math.sin(phase * 2 * math.pi * 2)) * 0.03 * h if motion in ("walk", "dance") else 0
    yb2 = yb - bob
    if pose == "sit":
        yb2 = yb + 0.25 * h
    p.use(shade(body, 0.7), idx).rect((cx - 0.1 * h, yb2 - 0.3 * h), (cx - 0.03 * h, yb2))
    p.use(shade(body, 0.7)).rect((cx + 0.03 * h, yb2 - 0.3 * h), (cx + 0.1 * h, yb2))
    p.use(body).rect((cx - 0.17 * h, yb2 - 0.68 * h), (cx + 0.17 * h, yb2 - 0.28 * h))
    p.use(shade(body, 0.8)).rect((cx - 0.24 * h, yb2 - 0.64 * h), (cx - 0.18 * h, yb2 - 0.4 * h))
    p.use(shade(body, 0.8)).rect((cx + 0.18 * h, yb2 - 0.64 * h), (cx + 0.24 * h, yb2 - 0.4 * h))
    p.use(body).rect((cx - 0.12 * h, yb2 - 0.92 * h), (cx + 0.12 * h, yb2 - 0.71 * h))
    p.use((60, 220, 250)).circle((cx - 0.05 * h + facing * 0.02 * h, yb2 - 0.82 * h), 0.025 * h)
    p.use((60, 220, 250)).circle((cx + 0.05 * h + facing * 0.02 * h, yb2 - 0.82 * h), 0.025 * h)
    p.use(shade(body, 0.6)).line((cx, yb2 - 0.92 * h), (cx, yb2 - 1.0 * h), 0.02 * h)
    p.use((50, 50, 220)).circle((cx, yb2 - 1.0 * h), 0.03 * h)


def draw_animal(p, cx, yb, h, w, kind, a, pose, phase, motion, facing, idx):
    fur = color(a.get("color"), (60, 100, 150))
    swing = math.sin(phase * 2 * math.pi * 3) * 0.12 * h if motion in ("walk", "run") else 0
    hop = abs(math.sin(phase * 2 * math.pi * 2)) * 0.5 * h if motion in ("jump", "dance") else 0
    yb2 = yb - hop
    if pose in ("sleep", "low"):
        body_c = (cx, yb2 - 0.25 * h)
        p.use(fur, idx).ellipse(body_c, (0.45 * w, 0.24 * h))
        p.use(fur).circle((cx + facing * 0.42 * w, yb2 - 0.22 * h), 0.22 * h)
        p.use(shade(fur, 0.7)).ellipse((cx + facing * 0.38 * w, yb2 - 0.38 * h), (0.07 * h, 0.12 * h))
        return
    body_y = yb2 - 0.55 * h
    for k, dx in enumerate((-0.32, -0.18, 0.18, 0.32)):
        s = swing if k % 2 == 0 else -swing
        p.use(shade(fur, 0.85), idx).line((cx + dx * w, body_y), (cx + dx * w + s, yb2), 0.09 * h)
    p.use(fur, idx).ellipse((cx, body_y), (0.42 * w, 0.24 * h))
    tail_up = kind in ("cat", "fox")
    p.use(fur).line((cx - facing * 0.4 * w, body_y - 0.05 * h),
                    (cx - facing * 0.58 * w, body_y - (0.5 if tail_up else 0.2) * h), 0.07 * h)
    hx, hy = cx + facing * 0.45 * w, body_y - 0.28 * h
    p.use(fur).circle((hx, hy), 0.26 * h)
    p.use(fur).ellipse((hx + facing * 0.2 * h, hy + 0.06 * h), (0.14 * h, 0.1 * h))
    if kind in ("cat", "fox", "kitten"):
        for s in (-1, 1):
            p.use(shade(fur, 0.8)).poly([(hx + s * 0.16 * h, hy - 0.12 * h), (hx + s * 0.06 * h, hy - 0.22 * h),
                                         (hx + s * 0.18 * h, hy - 0.42 * h)])
    elif kind == "rabbit":
        for s in (-1, 1):
            p.use(fur).ellipse((hx + s * 0.07 * h, hy - 0.42 * h), (0.05 * h, 0.2 * h))
    else:
        p.use(shade(fur, 0.65)).ellipse((hx - facing * 0.1 * h, hy + 0.02 * h), (0.07 * h, 0.16 * h), 15 * facing)
    p.use((30, 30, 30)).circle((hx + facing * 0.09 * h, hy - 0.04 * h), 0.04 * h)
    p.use((30, 30, 30)).circle((hx + facing * 0.33 * h, hy + 0.03 * h), 0.04 * h)


def draw_object(p, cx, yb, h, w, kind, a, idx):
    c = color(a.get("color"), (130, 150, 170))
    dk = shade(c, 0.65)
    x0, x1, y0 = cx - w / 2, cx + w / 2, yb - h
    p.use(c, idx)
    if kind == "ball" or kind == "apple":
        p.circle((cx, yb - h / 2), h / 2)
        p.use(shade(c, 1.25)).circle((cx - h * 0.15, yb - h * 0.65), h * 0.12)
    elif kind in ("bench", "table"):
        top_h = h * (0.12 if kind == "table" else 0.14)
        seat_y = y0 if kind == "table" else yb - h * 0.5
        p.use(c).rect((x0, seat_y), (x1, seat_y + top_h))
        p.use(dk).rect((x0 + w * 0.06, seat_y + top_h), (x0 + w * 0.12, yb))
        p.use(dk).rect((x1 - w * 0.12, seat_y + top_h), (x1 - w * 0.06, yb))
        if kind == "bench":
            p.use(c).rect((x0, y0), (x1, y0 + h * 0.12))
            p.use(dk).rect((x0 + w * 0.1, y0 + h * 0.12), (x0 + w * 0.14, seat_y))
            p.use(dk).rect((x1 - w * 0.14, y0 + h * 0.12), (x1 - w * 0.1, seat_y))
    elif kind == "chair":
        p.use(c).rect((x0, yb - h * 0.5), (x1, yb - h * 0.42))
        p.use(c).rect((x0, y0), (x0 + w * 0.15, yb - h * 0.42))
        p.use(dk).rect((x0, yb - h * 0.42), (x0 + w * 0.12, yb))
        p.use(dk).rect((x1 - w * 0.12, yb - h * 0.42), (x1, yb))
    elif kind in ("sofa", "bed"):
        p.use(c).rect((x0, yb - h * 0.5), (x1, yb - h * 0.1))
        p.use(dk).rect((x0, y0), (x1, yb - h * 0.5)) if kind == "sofa" else \
            p.use((240, 240, 240)).ellipse((x0 + w * 0.15, yb - h * 0.55), (w * 0.12, h * 0.15))
        p.use(dk).rect((x0, yb - h * 0.1), (x0 + w * 0.05, yb))
        p.use(dk).rect((x1 - w * 0.05, yb - h * 0.1), (x1, yb))
        if kind == "sofa":
            p.use(c).rect((x0 - w * 0.03, yb - h * 0.65), (x0 + w * 0.08, yb - h * 0.1))
            p.use(c).rect((x1 - w * 0.08, yb - h * 0.65), (x1 + w * 0.03, yb - h * 0.1))
    elif kind == "tree":
        p.use((40, 70, 100)).rect((cx - w * 0.1, yb - h * 0.5), (cx + w * 0.1, yb))
        for dx, dy, r in ((0, 0.62, 0.32), (-0.22, 0.5, 0.24), (0.22, 0.5, 0.24), (0, 0.8, 0.22)):
            p.use(c if (dx, dy) != (0, 0.8) else shade(c, 1.15)).circle((cx + dx * w * 1.2, yb - dy * h), r * h)
    elif kind == "umbrella":
        is_open = a.get("state") == "open"
        if is_open:
            p.use(c).ellipse((cx, y0 + h * 0.35), (w / 2, h * 0.35), 0, 180, 360)
            p.use(dk).line((cx, y0 + h * 0.35), (cx, yb), max(2, w * 0.04))
        else:
            p.use(c).poly([(cx - w * 0.07, y0 + h * 0.25), (cx + w * 0.07, y0 + h * 0.25), (cx, yb - h * 0.2)])
            p.use(dk).line((cx, y0 + h * 0.1), (cx, yb), max(2, w * 0.03))
    elif kind == "car":
        p.use(c).rect((x0, yb - h * 0.6), (x1, yb - h * 0.2))
        p.use(c).poly([(x0 + w * 0.2, yb - h * 0.6), (x0 + w * 0.32, y0), (x1 - w * 0.3, y0), (x1 - w * 0.15, yb - h * 0.6)])
        p.use((230, 210, 180)).poly([(x0 + w * 0.27, yb - h * 0.62), (x0 + w * 0.35, y0 + h * 0.1),
                                     (x1 - w * 0.33, y0 + h * 0.1), (x1 - w * 0.22, yb - h * 0.62)])
        for wx in (x0 + w * 0.22, x1 - w * 0.22):
            p.use((40, 40, 40)).circle((wx, yb - h * 0.2), h * 0.2)
    elif kind == "lamp":
        p.use(dk).rect((cx - w * 0.3, yb - h * 0.05), (cx + w * 0.3, yb))
        p.use(dk).line((cx, yb), (cx, y0 + h * 0.3), max(2, w * 0.1))
        p.use(c).poly([(cx - w * 0.5, y0 + h * 0.32), (cx + w * 0.5, y0 + h * 0.32), (cx + w * 0.3, y0),
                       (cx - w * 0.3, y0)])
    elif kind in ("plant", "flower"):
        p.use((60, 90, 170)).poly([(cx - w * 0.4, yb - h * 0.35), (cx + w * 0.4, yb - h * 0.35), (cx + w * 0.3, yb),
                                   (cx - w * 0.3, yb)])
        p.use((60, 150, 60)).ellipse((cx, yb - h * 0.62), (w * 0.45, h * 0.3))
        if kind == "flower":
            p.use(c).circle((cx, y0 + h * 0.12), w * 0.3)
    elif kind in ("cup", "bowl", "pot", "vase", "bottle", "teapot", "candle"):
        if kind == "teapot":
            p.use(c).ellipse((cx, yb - h * 0.45), (w * 0.35, h * 0.45))
            p.use(c).line((cx + w * 0.3, yb - h * 0.4), (cx + w * 0.5, yb - h * 0.8), max(2, h * 0.15))
            p.use(dk).circle((cx, y0 + h * 0.05), h * 0.1)
        elif kind == "bowl":
            p.use(c).ellipse((cx, y0), (w / 2, h), 0, 0, 180)
        elif kind in ("bottle", "vase", "candle"):
            p.use(c).rect((x0, y0 + h * 0.3), (x1, yb))
            p.use(c).rect((cx - w * 0.25, y0), (cx + w * 0.25, y0 + h * 0.3))
            if kind == "candle":
                p.use((60, 200, 255)).circle((cx, y0 - h * 0.1), w * 0.4)
        else:
            p.use(c).rect((x0, y0), (x1, yb))
            p.use(dk).line((x1, y0 + h * 0.3), (x1 + w * 0.3, y0 + h * 0.5), max(1, h * 0.15))
    elif kind == "book" or kind == "map" or kind == "laptop":
        p.use(c).rect((x0, y0), (x1, yb))
        p.use((235, 235, 235)).rect((x0 + w * 0.05, y0 + h * 0.2), (x1 - w * 0.05, yb - h * 0.2))
    elif kind == "kite" or kind == "balloon":
        if kind == "kite":
            p.use(c).poly([(cx, y0), (x1, yb - h / 2), (cx, yb), (x0, yb - h / 2)])
        else:
            p.use(c).ellipse((cx, y0 + h * 0.35), (w / 2, h * 0.35))
        p.use((60, 60, 60)).line((cx, yb), (cx, yb + h * 0.8), 1)
    elif kind == "bicycle":
        for wx in (x0 + w * 0.22, x1 - w * 0.22):
            p.use((40, 40, 40)).circle((wx, yb - h * 0.3), h * 0.3)
            p.use((200, 200, 200)).circle((wx, yb - h * 0.3), h * 0.22)
        p.use(c).line((x0 + w * 0.22, yb - h * 0.3), (cx, yb - h * 0.75), max(2, h * 0.08))
        p.use(c).line((cx, yb - h * 0.75), (x1 - w * 0.22, yb - h * 0.3), max(2, h * 0.08))
    else:  # box, gift, bag, suitcase, hat, toy, cake, clock, guitar, ... - a recognisable generic prop
        p.use(c).rect((x0, y0 + h * 0.1), (x1, yb))
        p.use(dk).rect((x0, y0), (x1, y0 + h * 0.18))
        if kind in ("box", "gift", "cake"):
            p.use(shade(c, 1.3)).rect((cx - w * 0.08, y0), (cx + w * 0.08, yb))
        if kind in ("bag", "suitcase"):
            p.use(dk).line((cx - w * 0.2, y0), (cx + w * 0.2, y0), max(1, h * 0.08))


def _draw_entity(p, ent, idx, t, frame_seed):
    x = ent["x0"] + (ent["x1"] - ent["x0"]) * _ease(t)
    cx = x * W
    h, w = ent["h"] * H, ent["w"] * W
    yb = (ent["y_base"] if ent["y_base"] is not None else 0.9) * H
    motion, pose = ent["motion"], ent["pose"]
    facing = 1 if ent["x1"] >= ent["x0"] - 1e-6 and (ent["x1"] > ent["x0"] or x < 0.5) else -1
    if motion == "dance":
        facing = 1 if math.sin(t * 2 * math.pi * 2) > 0 else -1
    kind, a = ent["kind"], dict(ent["attributes"])
    if ent["type"] == "character" and kind in HUMAN_KINDS:
        a["_long_hair"] = kind in ("girl", "woman", "lady", "princess", "mother", "mom", "grandma")
        if motion == "jump":
            yb -= abs(math.sin(t * 2 * math.pi * 2)) * 0.12 * H
        draw_human(p, cx, yb, h, a, pose, t, motion, facing, idx, frame_seed)
    elif ent["type"] == "character" and kind in ANIMAL_KINDS:
        draw_animal(p, cx, yb, h, w, kind, a, pose, t, motion, facing, idx)
    elif ent["type"] == "character":
        draw_robot(p, cx, yb, h, a, pose, t, motion, facing, idx)
    else:
        draw_object(p, cx, yb, h, w, kind, a, idx)


def _ease(t):
    return t * t * (3 - 2 * t)


def render_clip(cond, n_frames=24):
    """Renders one scene. Returns (frames, label_maps, ids) where ids[k-1] is the entity drawn with label k."""
    bg = cond["background"]
    base = background(bg, 0)
    ents = cond["entities"]
    ids = [e["id"] for e in ents]
    by_id = {e["id"]: e for e in ents}
    frames, labels = [], []
    for f in range(n_frames):
        t = f / max(1, n_frames - 1)
        img = base.copy()
        lab = np.zeros((H, W), np.uint8)
        p = Painter(img, lab)
        held_draw = []
        for k, e in enumerate(ents):
            if e["held_by"]:
                held_draw.append((k, e))
                continue
            _draw_entity(p, e, k + 1, t, _seed(e["id"]))
        for k, e in held_draw:
            holder = by_id.get(e["held_by"])
            if holder is None:
                continue
            hx = holder["x0"] + (holder["x1"] - holder["x0"]) * _ease(t)
            facing = 1 if holder["x1"] >= holder["x0"] and (holder["x1"] > holder["x0"] or hx < 0.5) else -1
            hh = holder["h"]
            if holder["kind"] in ("dog", "cat", "fox", "rabbit", "bear", "horse"):
                # carried in the mouth
                anchor_x = hx + facing * (holder["w"] * 0.45 + 0.012)
                anchor_y = holder["y_base"] - hh * 0.62
            elif holder["pose"] == "sit":
                anchor_x, anchor_y = hx + facing * hh * 0.11, holder["y_base"] - hh * 0.05
            else:
                anchor_x, anchor_y = hx + facing * hh * 0.2 / (16 / 9), holder["y_base"] - hh * 0.25
            e2 = dict(e, x0=anchor_x, x1=anchor_x, motion=None)
            if e["kind"] == "umbrella":
                e2["y_base"] = anchor_y + e["h"] * 0.1
                e2["x0"] = e2["x1"] = anchor_x - facing * 0.01
            elif e["kind"] in ("kite", "balloon"):
                e2["y_base"] = anchor_y - hh * 0.6
            else:
                e2["y_base"] = anchor_y + e["h"] * 0.5
            _draw_entity(p, e2, k + 1, 0.0, _seed(e["id"]))
        img = weather_overlay(img, bg, f)
        frames.append(img)
        labels.append(lab)
    return frames, labels, ids


def _seed(eid):
    return zlib.crc32(eid.encode())


def annotate(frame, label_map, ids, names, title=None):
    """Adds name tags and a caption for display. Kept off the frames the metrics read."""
    out = frame.copy()
    for k, eid in enumerate(ids, start=1):
        ys, xs = np.nonzero(label_map == k)
        if len(xs) < 12:
            continue
        name = names.get(eid, eid)
        tx, ty = int(xs.mean()), int(ys.min()) - 4
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        tx = min(max(2, tx - tw // 2), W - tw - 2)
        ty = max(th + 2, ty)
        cv2.rectangle(out, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (25, 25, 25), -1)
        cv2.putText(out, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)
    if title:
        cv2.rectangle(out, (0, H - 18), (W, H), (20, 20, 20), -1)
        cv2.putText(out, title[:78], (6, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
    return out
