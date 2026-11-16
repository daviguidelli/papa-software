"""Per-substrate alignment: place each scrub pattern on the array the camera found.

The flow:
  1. scan: one camera frame, corrected to a straight-down view
  2. find the copper electrode arrays (papa.vision.electrodes)
  3. each array's corners go to printer X, Y in mm (papa.vision.correction)
  4. each substrate's pattern (the bridge sketch: array width x height and its
     bridges, origin at the array's top-left) is fitted onto those corners as a
     rotation plus a shift, and its size is compared with the measured one
  5. the scrub toolpath for that substrate is the normal plan, placed with that
     rotation and shift (scrub settings start.x/start.y and rotation_deg)

"Top-left" and the rotation are relative to the printer axes through the
machine.yaml axis mapping (local +x, local +y), the same mapping a scrub uses
when nothing is corrected. Arrays are expected within 45 deg of that nominal
orientation; past that, which corner is "top-left" is ambiguous.

Substrates are numbered S1, S2, ... in reading order of the local frame: rows
from local top to bottom, then left to right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from papa.ops.geometry import axis_matrix as _axis
from papa.ops.geometry import to_local, to_machine
from papa.vision import correction as corr
from papa.vision import electrodes as elec

DEFAULTS: dict[str, Any] = {
    "size_tolerance_mm": 0.3,   # measured vs sketched array size before a warning
    "max_rotation_deg": 10.0,   # larger turns are refused (likely a misdetection)
    "patterns": {},             # "S1": {"name", "area": {"w","h"}, "bridges": [...]}
}


def merged(raw: dict | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    out["patterns"] = dict(out.get("patterns") or {})
    return out


@dataclass
class Placement:
    """Where one substrate's pattern goes, and how well the array matched it."""
    id: str
    corners_mm: list[list[float]]   # measured, in local order TL, TR, BR, BL
    center_mm: list[float]
    origin_mm: list[float]          # machine X, Y of the pattern's local (0, 0)
    rotation_deg: float             # + = turned from +X toward +Y
    measured_mm: list[float]        # measured width (local x), height (local y)
    pattern_mm: list[float]         # sketched width, height
    size_error_mm: list[float]      # measured - sketched
    fit_rms_mm: float               # corner misfit after rotation + shift
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        r = lambda v: round(float(v), 4)                   # noqa: E731
        return {
            "id": self.id,
            "corners_mm": [[r(x), r(y)] for x, y in self.corners_mm],
            "center_mm": [r(v) for v in self.center_mm],
            "origin_mm": [r(v) for v in self.origin_mm],
            "rotation_deg": r(self.rotation_deg),
            "measured_mm": [r(v) for v in self.measured_mm],
            "pattern_mm": [r(v) for v in self.pattern_mm],
            "size_error_mm": [r(v) for v in self.size_error_mm],
            "fit_rms_mm": r(self.fit_rms_mm),
            "warnings": list(self.warnings),
        }


def local_order(axes: dict[str, str], corners_mm: np.ndarray) -> np.ndarray:
    """Reorder 4 machine-frame corners as local TL, TR, BR, BL (local y points down)."""
    c = np.asarray(corners_mm, dtype=np.float64).reshape(4, 2)
    cx, cy = c.mean(0)
    loc = np.array([to_local(axes, cx, cy, x, y) for x, y in c])
    return c[np.argsort(np.arctan2(loc[:, 1], loc[:, 0]))]   # -135, -45, 45, 135 deg


def number(axes: dict[str, str], centers_mm: list[np.ndarray], row_mm: float) -> list[int]:
    """Indices in reading order of the local frame (rows top to bottom, then left to right)."""
    loc = [to_local(axes, 0.0, 0.0, float(x), float(y)) for x, y in centers_mm]
    idx = sorted(range(len(loc)), key=lambda i: loc[i][1])
    rows: list[list[int]] = []
    for i in idx:
        if rows and abs(loc[i][1] - loc[rows[-1][0]][1]) < row_mm:
            rows[-1].append(i)
        else:
            rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda j: loc[j][0])]


def fit(axes: dict[str, str], ordered_mm: np.ndarray, w: float, h: float) -> tuple[list[float], float, float]:
    """Rotation + shift that best lays the w x h pattern on the measured corners.

    Returns (origin, rotation_deg, rms). The pattern keeps its sketched size,
    so a size difference is shared evenly around the array, not stretched.
    """
    local = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    nominal = np.array([to_machine(axes, 0.0, 0.0, x, y) for x, y in local])
    m = np.asarray(ordered_mm, dtype=np.float64)
    pc, mc = nominal - nominal.mean(0), m - m.mean(0)
    th = math.atan2(float(np.sum(pc[:, 0] * mc[:, 1] - pc[:, 1] * mc[:, 0])),
                    float(np.sum(pc[:, 0] * mc[:, 0] + pc[:, 1] * mc[:, 1])))
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = m.mean(0) - R @ nominal.mean(0)
    rms = float(np.sqrt(np.mean(np.sum((nominal @ R.T + t - m) ** 2, axis=1))))
    return [float(t[0]), float(t[1])], math.degrees(th), rms


def place(sid: str, axes: dict[str, str], corners_mm: np.ndarray, pattern: dict,
          settings: dict[str, Any]) -> Placement:
    st = merged(settings)
    w, h = float(pattern["area"]["w"]), float(pattern["area"]["h"])
    c = local_order(axes, corners_mm)
    mw = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    mh = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    origin, rot, rms = fit(axes, c, w, h)
    tol = float(st["size_tolerance_mm"])
    warns = []
    if abs(mw - w) > tol or abs(mh - h) > tol:
        warns.append(f"Measured {mw:.2f} x {mh:.2f} mm, pattern is {w:.2f} x {h:.2f} mm "
                     f"(tolerance {tol:g} mm).")
        if abs(mw - h) <= tol and abs(mh - w) <= tol and abs(w - h) > tol:
            warns.append("The array looks turned 90 deg from the pattern.")
    if abs(rot) > float(st["max_rotation_deg"]):
        warns.append(f"Turned {rot:+.1f} deg, more than the {st['max_rotation_deg']:g} deg allowed. "
                     "Straighten the substrate or check the detection.")
    return Placement(sid, c.tolist(), c.mean(0).tolist(), origin, rot, [mw, mh], [w, h],
                     [mw - w, mh - h], rms, warns)


def scrub_settings(base: dict[str, Any], pattern: dict, p: Placement) -> dict[str, Any]:
    """The Scrub tab's settings with this substrate's pattern and placement.
    Contact Z, speeds and swab size come from the Scrub tab unchanged."""
    st = dict(base)
    st["area"] = {"w": float(pattern["area"]["w"]), "h": float(pattern["area"]["h"])}
    st["bridges"] = list(pattern.get("bridges") or [])
    start = dict(base.get("start") or {})
    if start:
        start["x"], start["y"] = round(p.origin_mm[0], 4), round(p.origin_mm[1], 4)
    st["start"] = start or None
    st["rotation_deg"] = round(p.rotation_deg, 4)
    return st


def pattern_from_sketch(payload: dict, name: str = "") -> dict:
    """A bridge-sketcher export (or the Scrub tab's sketch) as a stored pattern."""
    sub = payload.get("substrate") or payload.get("area") or {}
    return {"name": name or payload.get("name") or "",
            "area": {"w": float(sub.get("w", 25.0)), "h": float(sub.get("h", 25.0))},
            "bridges": list(payload.get("bridges") or [])}


@dataclass
class ScanResult:
    photo_size: tuple[int, int]
    arrays_px: list[np.ndarray]           # corners in photo pixels, local order
    placements: list[Placement]
    pattern_source: dict[str, str]        # "S1" -> "own" or "scrub tab"
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "photo_size": list(self.photo_size),
            "substrates": [{**p.as_dict(), "pattern_source": self.pattern_source[p.id],
                            "corners_px": [[round(float(x), 2), round(float(y), 2)] for x, y in px]}
                           for p, px in zip(self.placements, self.arrays_px)],
            "warnings": list(self.warnings),
        }


def scan(rgb: np.ndarray, axes: dict[str, str], printer_axes: corr.PrinterAxes,
         settings: dict[str, Any], default_pattern: dict,
         params: elec.ElectrodeParams | None = None) -> ScanResult:
    """Find every array in one RGB camera frame and place its pattern."""
    st = merged(settings)
    size = (rgb.shape[1], rgb.shape[0])
    if not corr.supported(size):
        raise ValueError(f"no lens calibration for {size[0]} x {size[1]} frames")
    c = corr.get_correction(size)
    found = elec.detect(c.correct_image(rgb), params)
    to_mm = lambda pts: printer_axes.plate_to_mm(c.corrected_to_plate(pts))   # noqa: E731
    corners_mm = [to_mm(a.corners) for a in found]
    warnings: list[str] = []
    if not found:
        warnings.append("No electrode arrays found. Check lighting and focus, or the copper "
                        "colour settings (detection.electrodes).")
        return ScanResult(size, [], [], {}, warnings)
    sizes = [float(np.linalg.norm(cm[1] - cm[0]) + np.linalg.norm(cm[2] - cm[1])) / 2 for cm in corners_mm]
    order = number(axes, [cm.mean(0) for cm in corners_mm], row_mm=0.5 * float(np.median(sizes)))
    placements, arrays_px, source = [], [], {}
    for n, i in enumerate(order, 1):
        sid = f"S{n}"
        own = st["patterns"].get(sid)
        pattern = own or default_pattern
        source[sid] = "own" if own else "scrub tab"
        p = place(sid, axes, corners_mm[i], pattern, st)
        # photo pixels of the same corners, in the same (local) order
        ordered_mm = np.asarray(p.corners_mm)
        arrays_px.append(c.from_plate(printer_axes.mm_to_plate(ordered_mm)))
        placements.append(p)
    return ScanResult(size, arrays_px, placements, source, warnings)


def detect_only(rgb: np.ndarray, params: elec.ElectrodeParams | None = None) -> list[np.ndarray]:
    """Arrays in one frame as photo-pixel corners, with no printer mapping.
    For checking the camera and the copper settings before camera points exist."""
    size = (rgb.shape[1], rgb.shape[0])
    if not corr.supported(size):
        raise ValueError(f"no lens calibration for {size[0]} x {size[1]} frames")
    c = corr.get_correction(size)
    found = elec.detect(c.correct_image(rgb), params)
    out = [c.from_plate(c.corrected_to_plate(a.corners)) for a in found]
    out.sort(key=lambda px: (round(float(px[:, 1].mean()) / 100), float(px[:, 0].mean())))
    return out


def rectify(rgb: np.ndarray, machine_axes: dict[str, str], printer_axes: corr.PrinterAxes,
            origin_mm: list[float], rot_deg: float, w: float, h: float,
            px_per_mm: float) -> np.ndarray:
    """A straight-on picture of one substrate: exactly its w x h mm pattern area
    (local frame, top-left origin, x right, y down) at px_per_mm, taken from a
    camera photo through the lens model and the camera points."""
    import cv2

    W, H = max(1, int(round(w * px_per_mm))), max(1, int(round(h * px_per_mm)))
    us = (np.arange(W) + 0.5) / px_per_mm
    vs = (np.arange(H) + 0.5) / px_per_mm
    U, V = np.meshgrid(us, vs)
    (ax, ay), (bx, by) = _axis(machine_axes)
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    dx, dy = U * ax + V * bx, U * ay + V * by
    mx = origin_mm[0] + c * dx - s * dy
    my = origin_mm[1] + s * dx + c * dy
    size = (rgb.shape[1], rgb.shape[0])
    photo = corr.get_correction(size).from_plate(printer_axes.mm_to_plate(np.c_[mx.ravel(), my.ravel()]))
    mapx = photo[:, 0].reshape(H, W).astype(np.float32)
    mapy = photo[:, 1].reshape(H, W).astype(np.float32)
    return cv2.remap(rgb, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def field_mm(photo_size: tuple[int, int], printer_axes: corr.PrinterAxes) -> tuple[float, float, float, float]:
    """Printer X/Y box (x0, x1, y0, y1) that the camera frame covers."""
    w, h = photo_size
    t = np.linspace(0, 1, 60)
    border = np.vstack([np.c_[t * (w - 1), 0 * t], np.c_[t * (w - 1), 0 * t + h - 1],
                        np.c_[0 * t, t * (h - 1)], np.c_[0 * t + w - 1, t * (h - 1)]])
    mm = printer_axes.plate_to_mm(corr.get_correction(photo_size).to_plate(border))
    return float(mm[:, 0].min()), float(mm[:, 0].max()), float(mm[:, 1].min()), float(mm[:, 1].max())


def printer_view(rgb: np.ndarray, printer_axes: corr.PrinterAxes, box: tuple[float, float, float, float],
                 px_per_mm: float) -> np.ndarray:
    """The plate as seen square to the printer: image right is exactly +X and
    image up is exactly +Y. `box` is (x0, x1, y0, y1) in printer mm; pixel
    (u, v) is at X = x0 + (u + 0.5) / px_per_mm, Y = y1 - (v + 0.5) / px_per_mm."""
    import cv2

    x0, x1, y0, y1 = box
    W, H = max(1, int(round((x1 - x0) * px_per_mm))), max(1, int(round((y1 - y0) * px_per_mm)))
    X, Y = np.meshgrid(x0 + (np.arange(W) + 0.5) / px_per_mm, y1 - (np.arange(H) + 0.5) / px_per_mm)
    size = (rgb.shape[1], rgb.shape[0])
    photo = corr.get_correction(size).from_plate(printer_axes.mm_to_plate(np.c_[X.ravel(), Y.ravel()]))
    return cv2.remap(rgb, photo[:, 0].reshape(H, W).astype(np.float32),
                     photo[:, 1].reshape(H, W).astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT)
