"""Detection by blue filter.

The camera answers one question: is blue present where it should be? Films are
blue, and so is the swab stem. Nothing else is asked of it, which is why a
fixed camera and a hue filter are enough.

Filtering happens in HSV rather than raw RGB: a hue range survives the lab
lights brightening or dimming, a fixed RGB cutoff does not.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class HsvBand:
    """Hue bounds in OpenCV units (0-179, i.e. degrees halved), as stored in
    calibration.yaml. Blue film lands around 105-120 there."""
    h: tuple[int, int]
    s_min: int
    v_min: int

    @classmethod
    def from_cfg(cls, d: dict) -> "HsvBand":
        h = d.get("h", [100, 130])
        return cls((int(h[0]), int(h[1])), int(d.get("s_min", 80)), int(d.get("v_min", 60)))


@dataclass(frozen=True)
class Detection:
    present: bool
    fraction: float
    misaligned: bool = False


class CameraMap:
    """Pixel to machine-mm mapping from four jogged points.

    One homography covers the whole bed, so every detection region is derived
    from a calibrated origin. Adding a slot or a swab needs no extra setup.
    """

    def __init__(self, points_px: list[Point], points_mm: list[Point]) -> None:
        if len(points_px) < 4 or len(points_mm) < 4:
            raise ValueError("camera mapping needs four point pairs")
        src = np.array(points_mm[:4], dtype=np.float32)
        dst = np.array(points_px[:4], dtype=np.float32)
        self._mm_to_px = cv2.getPerspectiveTransform(src, dst)

    @classmethod
    def from_cfg(cls, detection: dict) -> "CameraMap":
        m = detection.get("camera_to_machine", {})
        return cls(m.get("points_px") or [], m.get("points_mm") or [])

    def to_px(self, pts_mm: list[Point]) -> np.ndarray:
        arr = np.array([pts_mm], dtype=np.float32)
        return cv2.perspectiveTransform(arr, self._mm_to_px)[0]


def blue_mask(frame_rgb: np.ndarray, band: HsvBand) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    lo = np.array([band.h[0], band.s_min, band.v_min], dtype=np.uint8)
    hi = np.array([band.h[1], 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def _fraction_in_polygon(mask: np.ndarray, poly_px: np.ndarray) -> float:
    region = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(region, poly_px.astype(np.int32), 255)
    area = int(np.count_nonzero(region))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(mask, mask, mask=region))) / area


def slide_present(
    frame_rgb: np.ndarray,
    cmap: CameraMap,
    origin_mm: Point,
    area_w_mm: float,
    area_h_mm: float,
    band: HsvBand,
    min_fraction: float = 0.35,
    misaligned_fraction: float = 0.12,
) -> Detection:
    """Project the pattern-area square at a slot origin and count blue inside.

    A slide sitting off its slot fills only part of the square, so a fraction
    between the two thresholds is reported as misaligned rather than absent.
    """
    ox, oy = origin_mm
    corners = [
        (ox, oy),
        (ox + area_w_mm, oy),
        (ox + area_w_mm, oy + area_h_mm),
        (ox, oy + area_h_mm),
    ]
    poly = cmap.to_px(corners)
    frac = _fraction_in_polygon(blue_mask(frame_rgb, band), poly)
    return Detection(
        present=frac >= min_fraction,
        fraction=frac,
        misaligned=misaligned_fraction <= frac < min_fraction,
    )


def swab_present(
    frame_rgb: np.ndarray,
    cmap: CameraMap,
    position_mm: Point,
    roi_mm: float,
    band: HsvBand,
    min_fraction: float = 0.30,
) -> Detection:
    """Check a small disc at a rack position. The stem is seen end-on.

    Sanity check once the camera is mounted: the 2.54 mm stem should span about
    ten pixels or more, or this gets unreliable.
    """
    x, y = position_mm
    r = roi_mm / 2
    corners = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    poly = cmap.to_px(corners)
    frac = _fraction_in_polygon(blue_mask(frame_rgb, band), poly)
    return Detection(present=frac >= min_fraction, fraction=frac)


def sample_hsv(frame_rgb: np.ndarray, center_px: tuple[int, int], radius_px: int = 6) -> HsvBand:
    """Build a band from real pixels, so nobody hand-types hue numbers."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    cx, cy = center_px
    patch = hsv[max(0, cy - radius_px):cy + radius_px, max(0, cx - radius_px):cx + radius_px]
    if patch.size == 0:
        raise ValueError("sample point is outside the frame")
    h = patch[:, :, 0].astype(int)
    s = patch[:, :, 1].astype(int)
    v = patch[:, :, 2].astype(int)
    return HsvBand(
        h=(max(0, int(np.percentile(h, 5)) - 5), min(179, int(np.percentile(h, 95)) + 5)),
        s_min=max(20, int(np.percentile(s, 10)) - 20),
        v_min=max(20, int(np.percentile(v, 10)) - 20),
    )
