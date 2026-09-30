"""Reading a clip out of a phone video and writing a video browsers can actually play.

OpenCV's pip wheel can't encode H.264, and mp4v files don't play in Chrome/Firefox, so the writer
pipes frames into the ffmpeg binary that ships with imageio-ffmpeg. If that isn't installed it
falls back to VP8 .webm, which OpenCV can write and every modern browser plays.
"""
import os

import cv2
import numpy as np


def clip_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:  # some containers report 0 / NaN / 1000
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    return {"fps": fps, "frames": n, "duration": n / fps if n > 0 else 0.0, "width": w, "height": h}


def resize_to(frame, max_dim):
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    # even dimensions keep the H.264 encoder happy
    new_w, new_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    if (new_w, new_h) == (w, h):
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def iter_frames(path, start_s=0.0, duration_s=30.0, max_dim=640, step=1):
    """Yields (index_in_clip, time_s, frame) for the requested window of the video."""
    cap = cv2.VideoCapture(path)
    fps = clip_info(path)["fps"]
    first = int(round(start_s * fps))
    last = first + int(round(duration_s * fps))

    # grab() instead of CAP_PROP_POS_FRAMES: seeking in phone videos lands on the nearest keyframe
    idx = 0
    while idx < first and cap.grab():
        idx += 1

    k = 0
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - first) % step == 0:
            yield k, idx / fps, resize_to(frame, max_dim)
            k += 1
        idx += 1
    cap.release()


def read_frame_pair(path, time_s, max_dim=960):
    """Returns frames n and n+1 closest to time_s (consecutive frames of the original video)."""
    info = clip_info(path)
    n = int(round(time_s * info["fps"]))
    if info["frames"] > 1:
        n = max(0, min(n, info["frames"] - 2))
    cap = cv2.VideoCapture(path)
    idx = 0
    while idx < n and cap.grab():
        idx += 1
    ok1, f1 = cap.read()
    ok2, f2 = cap.read()
    cap.release()
    if not (ok1 and ok2):
        return None
    return resize_to(f1, max_dim), resize_to(f2, max_dim), n, info["fps"]


class VideoWriter:
    """Browser-friendly writer. Use .path afterwards - the extension depends on the backend used."""

    def __init__(self, path_no_ext, fps, size):
        self.size = size
        self._gen = None
        self._cv = None
        try:
            import imageio_ffmpeg
            self.path = path_no_ext + ".mp4"
            self._gen = imageio_ffmpeg.write_frames(
                self.path, size, fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                macro_block_size=2, ffmpeg_log_level="error",
                output_params=["-crf", "26", "-preset", "veryfast", "-movflags", "+faststart"],
            )
            self._gen.send(None)
        except Exception:
            self._gen = None
            self.path = path_no_ext + ".webm"
            self._cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"VP80"), fps, size)

    def write(self, frame):
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(frame))
        else:
            self._cv.write(frame)

    def close(self):
        if self._gen is not None:
            self._gen.close()
        elif self._cv is not None:
            self._cv.release()
        return os.path.exists(self.path) and os.path.getsize(self.path) > 0
