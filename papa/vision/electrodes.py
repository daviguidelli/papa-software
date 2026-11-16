"""Electrode array detection: copper lines on a white holder.

Runs on the corrected (straight-down) view, where an array is a true
rectangle. Steps:

1. keep copper-coloured pixels (HSV window; copper sits near hue 13-19 with
   low-to-medium saturation, well apart from the white plate and blue parts),
2. close the gaps between electrode lines so each array is one solid blob,
3. keep blobs of the right size and shape that are mostly copper and sit on a
   bright holder, and fit a rotated rectangle to each,
4. refine each side to sub-pixel accuracy (see refine): the outermost copper
   edge is found along short stretches of each side, a line is fitted through
   them, and the four lines are intersected. The blob outline alone is only good to about a pixel, which on
   a 10 mm array is about a degree; the refined sides are good to a few
   hundredths of a pixel.

Sizes are fractions of the image, so the same settings work at any frame size.
Every setting can be overridden from calibration.yaml under
detection.electrodes (same names as ElectrodeParams).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class ElectrodeParams:
    h_min: int = 5            # OpenCV hue, 0-179
    h_max: int = 28
    s_min: int = 45
    v_min: int = 70
    min_area_frac: float = 0.0002   # blob area / image area
    max_area_frac: float = 0.05
    close_frac: float = 0.012       # gap-closing size / image width; wider than the gap
                                    # between lines, narrower than the gap between arrays
    min_fill: float = 0.55          # blob area / rotated-rectangle area
    max_aspect: float = 2.5         # long side / short side
    min_copper: float = 0.35        # copper pixels / rectangle area
    min_surround_v: int = 140       # brightness around the array (white holder)

    @classmethod
    def from_cfg(cls, d: dict | None) -> "ElectrodeParams":
        names = {f.name: f.type for f in fields(cls)}
        out = cls()
        for k, v in (d or {}).items():
            if k in names and v is not None:
                out = replace(out, **{k: type(getattr(out, k))(v)})
        return out


@dataclass
class Array:
    corners: np.ndarray       # 4 x 2, pixels of the image searched, minAreaRect order
    center: tuple[float, float]
    copper: float             # fraction of the rectangle that is copper


def copper_mask(rgb: np.ndarray, p: ElectrodeParams) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (p.h_min, p.s_min, p.v_min), (p.h_max, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def copper_score(rgb: np.ndarray, p: ElectrodeParams) -> np.ndarray:
    """0-1 copper score per pixel. Soft (saturation of copper-hued pixels):
    blended edge pixels keep copper's hue at lower saturation and so count
    partly, which is where the sub-pixel detail comes from."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_ok = ((hsv[:, :, 0] >= p.h_min) & (hsv[:, :, 0] <= p.h_max)
              & (hsv[:, :, 2] >= p.v_min)).astype(np.float32)
    return hue_ok * np.clip(hsv[:, :, 1] / 255.0, 0, 1)


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    m = pts.astype(np.float32).reshape(-1, 1, 2)
    return cv2.remap(img, m[:, :, 0], m[:, :, 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).ravel()


def refine(score: np.ndarray, corners: np.ndarray, reach: float, segments: int = 10) -> np.ndarray | None:
    """Sub-pixel corners from the four sides of a rough rectangle (None if it fails).

    Each side is cut into short segments. A segment's samples are averaged
    along the side, which turns electrode lines and gaps into one density
    profile across the edge; the edge is the outermost point where that profile
    reaches half the copper level just inside it. A line through the segment
    edges is each side; neighbouring sides meet at the corners.
    """
    c = np.asarray(corners, dtype=np.float64)
    centre = c.mean(0)
    offsets = np.arange(-reach, reach + 1e-9, 0.25)
    lines = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        length = np.linalg.norm(b - a)
        d = (b - a) / length
        n = np.array([d[1], -d[0]])
        if n @ ((a + b) / 2 - centre) < 0:
            n = -n                                   # outward
        pts = []
        bounds = np.linspace(0.12, 0.88, segments + 1)
        for t0, t1 in zip(bounds[:-1], bounds[1:]):
            ts = np.arange(t0 * length, t1 * length, 0.5)
            grid = (a + np.outer(ts, d))[:, None, :] + offsets[None, :, None] * n
            prof = _sample(score, grid.reshape(-1, 2)).reshape(len(ts), len(offsets)).mean(0)
            outside = float(np.median(prof[offsets > 0.6 * reach]))
            peak = float(prof[offsets <= 0].max())
            if peak - outside < 0.05:
                continue
            half = (peak + outside) / 2
            idx = np.nonzero(prof >= half)[0]
            j = idx[-1]                              # outermost point at or above half
            if j + 1 >= len(offsets):
                continue
            f = (prof[j] - half) / (prof[j] - prof[j + 1])
            s_edge = offsets[j] + f * (offsets[j + 1] - offsets[j])
            pts.append(a + d * (t0 + t1) / 2 * length + n * s_edge)
        if len(pts) < max(3, segments // 2):
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):                # corner i is where side i-1 meets side i
        (p1, v1), (p2, v2) = lines[i - 1], lines[i]
        A = np.array([v1, -v2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            return None
        s_, _ = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s_ * v1)
    out = np.array(out)
    if np.max(np.linalg.norm(out - c, axis=1)) > reach:   # refinement wandered off
        return None
    return out


def detect(rgb: np.ndarray, p: ElectrodeParams | None = None) -> list[Array]:
    """Arrays in an RGB image (normally the corrected view), largest first."""
    p = p or ElectrodeParams()
    h, w = rgb.shape[:2]
    mask = copper_mask(rgb, p)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    k = max(3, int(round(p.close_frac * w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blobs = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)

    score = copper_score(rgb, p)
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[Array] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not p.min_area_frac * w * h <= area <= p.max_area_frac * w * h:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) == 0:
            continue
        if area / (rw * rh) < p.min_fill or max(rw, rh) / min(rw, rh) > p.max_aspect:
            continue
        corners = cv2.boxPoints(rect)
        box = np.zeros_like(mask)
        cv2.fillPoly(box, [corners.astype(np.int32)], 255)
        copper = cv2.countNonZero(cv2.bitwise_and(mask, box)) / max(1, cv2.countNonZero(box))
        if copper < p.min_copper:
            continue
        ring = cv2.subtract(cv2.dilate(box, kernel), box)
        if cv2.mean(gray, mask=ring)[0] < p.min_surround_v:
            continue
        refined = refine(score, corners.astype(np.float64), reach=k / 2 + 4)
        if refined is not None:
            corners = refined
        corners = np.asarray(corners, dtype=np.float64)
        found.append(Array(corners, tuple(map(float, corners.mean(0))), float(copper)))
    found.sort(key=lambda a: -cv2.contourArea(a.corners.astype(np.float32)))
    return found
