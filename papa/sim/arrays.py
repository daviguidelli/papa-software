"""Synthetic camera frames with electrode arrays at known printer positions.

Arrays are drawn in printer mm, mapped to the plate (straight-down) view with a
PrinterAxes fit, then warped into a camera photo with the real lens and tilt
model. Detection and alignment can then be checked against the exact pose that
was drawn. Used by the tests and by the simulated camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from papa.ops.geometry import to_machine
from papa.vision import correction as corr

PLATE_RGB = (222, 224, 226)
HOLDER_RGB = (240, 241, 242)
COPPER_RGB = (190, 120, 60)


@dataclass(frozen=True)
class SimArray:
    origin_mm: tuple[float, float]  # machine X, Y of the array's local top-left corner
    w: float                         # local x size, mm
    h: float                         # local y size, mm
    rot_deg: float = 0.0


_maps: dict = {}


def _photo_maps(size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """For every photo pixel, where it lands in the plate view."""
    if size not in _maps:
        w, h = size
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        plate = corr.get_correction(size).to_plate(np.c_[xs.ravel(), ys.ravel()])
        _maps[size] = (plate[:, 0].reshape(h, w).astype(np.float32),
                       plate[:, 1].reshape(h, w).astype(np.float32))
    return _maps[size]


def render(arrays: list[SimArray], machine_axes: dict[str, str], printer_axes: corr.PrinterAxes,
           size: tuple[int, int] = corr.GUI_PHOTO_SIZE, line_mm: float = 0.4,
           pitch_mm: float = 1.0, holder_mm: float = 3.0, base: np.ndarray | None = None) -> np.ndarray:
    """RGB photo of the plate with the given arrays. With `base` (an RGB photo of
    the same size) the arrays are drawn over it instead of over a bare plate."""
    plate_w, plate_h = 3200, 2200      # covers the 1080p reference view with margin
    plate = np.zeros((plate_h, plate_w, 3), np.uint8)
    plate[:] = PLATE_RGB
    cover = np.zeros((plate_h, plate_w), np.uint8)

    def poly(a: SimArray, pts_local: list[tuple[float, float]]) -> np.ndarray:
        mm = [to_machine(machine_axes, a.origin_mm[0], a.origin_mm[1], x, y, a.rot_deg)
              for x, y in pts_local]
        return np.round(printer_axes.mm_to_plate(mm) * 8).astype(np.int32)   # 3 bits subpixel

    for a in arrays:
        m = holder_mm
        holder = [poly(a, [(-m, -m), (a.w + m, -m), (a.w + m, a.h + m), (-m, a.h + m)])]
        cv2.fillPoly(plate, holder, HOLDER_RGB, cv2.LINE_AA, shift=3)
        cv2.fillPoly(cover, holder, 255, cv2.LINE_AA, shift=3)
        y = 0.0
        while y + line_mm <= a.h + 1e-9:     # copper lines along local x, edge to edge
            cv2.fillPoly(plate, [poly(a, [(0, y), (a.w, y), (a.w, y + line_mm), (0, y + line_mm)])],
                         COPPER_RGB, cv2.LINE_AA, shift=3)
            y += pitch_mm
        if a.h - (y - pitch_mm + line_mm) > 1e-6:   # make the last line meet the bottom edge
            cv2.fillPoly(plate, [poly(a, [(0, a.h - line_mm), (a.w, a.h - line_mm), (a.w, a.h), (0, a.h)])],
                         COPPER_RGB, cv2.LINE_AA, shift=3)
        cv2.fillPoly(plate, [poly(a, [(0, 0), (line_mm, 0), (line_mm, a.h), (0, a.h)])],
                     COPPER_RGB, cv2.LINE_AA, shift=3)   # a bus bar down the left edge
    mx, my = _photo_maps(tuple(size))
    photo = cv2.remap(plate, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=PLATE_RGB)
    if base is None:
        return photo
    alpha = cv2.remap(cover, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=0).astype(np.float32)[:, :, None] / 255.0
    return (photo * alpha + base * (1 - alpha)).astype(np.uint8)
