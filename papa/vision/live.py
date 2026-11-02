"""Live view encoding. Downscale, stamp, JPEG.

The stamp (REAL, MOCK or REMOTE, a clock, the source size) is burned into every
frame so a frozen stream or a mocked camera is obvious at a glance.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


def encode_jpeg(rgb: np.ndarray, width: int, label: str, quality: int = 75) -> bytes:
    h, w = rgb.shape[:2]
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if width < w:
        out = cv2.resize(out, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    t = time.time()
    text = f"{label}  {time.strftime('%H:%M:%S', time.localtime(t))}.{int(t * 1000) % 1000:03d}  {w}x{h}"
    scale = max(0.45, out.shape[1] / 1600)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = out.shape[0] - 8
    cv2.rectangle(out, (4, y - th - 6), (12 + tw, y + base + 2), (0, 0, 0), -1)
    cv2.putText(out, text, (8, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (180, 220, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode the camera frame")
    return buf.tobytes()
