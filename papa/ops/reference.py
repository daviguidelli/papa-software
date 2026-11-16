"""Reference substrate: one known glass square with a known G-code toolpath,
and how to move that toolpath onto each new substrate.

The reference is a photo of one substrate (any camera, e.g. a phone), the four
corners of its glass square clicked on that photo, the corner that carries the
red dot, and where that glass sat on the printer when the G-code was made
(make_toolpath.py assumes a square, square to the printer, from x_min, y_min,
`size` mm on a side, photographed with image right = printer +X and image up =
printer +Y).

For a new substrate, its glass corners in printer mm (from a scan through the
camera points) are matched corner for corner with the reference, the red dot
fixing which is which, and a rotation plus shift is fitted. The G-code's X/Y
moves are rotated and shifted by exactly that; Z, feeds and everything else
are left as they are.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class ReferenceError(ValueError):
    pass


# ---- corners and the red dot ---------------------------------------------------

def image_order(corners: Any) -> np.ndarray:
    """4 corners as top-left, top-right, bottom-right, bottom-left in the image."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    m = c.mean(0)
    return c[np.argsort(np.arctan2(c[:, 1] - m[1], c[:, 0] - m[0]))]


def red_corner(rgb: np.ndarray, corners: Any, reach: float = 0.2) -> int | None:
    """Index (into corners) of the corner with the red dot next to it, or None.

    Looks for strongly red pixels within `reach` of a side length of each
    corner, inside or just outside the square."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 80))
    ys, xs = np.nonzero(red)
    if not len(xs):
        return None
    pts = np.c_[xs, ys].astype(np.float64)
    counts = []
    for corner in c:
        counts.append(int(np.sum(np.linalg.norm(pts - corner, axis=1) < reach * side)))
    best = int(np.argmax(counts))
    if counts[best] < 4 or sorted(counts)[-2] > 0.5 * counts[best]:
        return None            # no dot, or not clearly at one corner
    return best


# ---- the reference in printer mm -------------------------------------------------

def reference_mm(placement: dict) -> np.ndarray:
    """The reference glass corners in printer mm, in image order (TL, TR, BR,
    BL of the reference photo), assuming photo right = +X and up = +Y."""
    x0, y0, s = float(placement["x_min"]), float(placement["y_min"]), float(placement["size"])
    return np.array([[x0, y0 + s], [x0 + s, y0 + s], [x0 + s, y0], [x0, y0]])


def photo_to_mm(photo_corners: Any, placement: dict) -> np.ndarray:
    """Homography from reference-photo pixels to printer mm (for drawing the
    G-code over the photo)."""
    H, _ = cv2.findHomography(np.float32(image_order(photo_corners)), np.float32(reference_mm(placement)))
    return H


# ---- matching a new substrate --------------------------------------------------------

@dataclass
class Match:
    rotation_deg: float          # new = R(rotation) * reference + shift, + from +X toward +Y
    shift_mm: list[float]        # where the reference's centre moved to, minus where it was
    rms_mm: float                # corner misfit after the fit
    size_mm: float               # measured side of the new glass (mean)
    red_found: bool
    R: np.ndarray
    t: np.ndarray
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"rotation_deg": round(self.rotation_deg, 4), "shift_mm": [round(v, 4) for v in self.shift_mm],
                "rms_mm": round(self.rms_mm, 4), "size_mm": round(self.size_mm, 4),
                "red_found": self.red_found, "warnings": list(self.warnings)}


def _ccw(c: np.ndarray) -> np.ndarray:
    """Order points counter-clockwise (in a +X right, +Y up frame), starting anywhere."""
    m = c.mean(0)
    return c[np.argsort(np.arctan2(c[:, 1] - m[1], c[:, 0] - m[0]))]


def match(ref_mm: np.ndarray, ref_red: int, new_mm: np.ndarray, new_red_mm: np.ndarray | None,
          size: float, tolerance_mm: float = 0.5) -> Match:
    """Fit new = R ref + t. ref_mm: reference corners (mm), ref_red: index of
    the red-dot corner in ref_mm; new_mm: the new glass corners (mm, any order);
    new_red_mm: the new red-dot corner position (mm) if the dot was found."""
    ref = np.asarray(ref_mm, dtype=np.float64)
    red_pt = ref[ref_red]
    ref_ccw = _ccw(ref)
    k_ref = int(np.argmin(np.linalg.norm(ref_ccw - red_pt, axis=1)))
    ref_ccw = np.roll(ref_ccw, -k_ref, axis=0)            # red-dot corner first
    new_ccw = _ccw(np.asarray(new_mm, dtype=np.float64))
    warns: list[str] = []
    best = None
    if new_red_mm is not None:
        k = int(np.argmin(np.linalg.norm(new_ccw - np.asarray(new_red_mm), axis=1)))
        shifts = [k]
    else:
        shifts = range(4)                                  # no dot: the smallest turn wins
        warns.append("Red dot not found on this substrate: assumed it sits the same way round as the "
                      "reference (smallest turn).")
    for k in shifts:
        new = np.roll(new_ccw, -k, axis=0)
        a, b = ref_ccw - ref_ccw.mean(0), new - new.mean(0)
        th = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])), float(np.sum(a * b)))
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        t = new.mean(0) - R @ ref_ccw.mean(0)
        rms = float(np.sqrt(np.mean(np.sum((ref_ccw @ R.T + t - new) ** 2, axis=1))))
        cand = (abs(th) if new_red_mm is None else 0.0, th, R, t, rms, new)
        if best is None or cand[0] < best[0]:
            best = cand
    _, th, R, t, rms, new = best
    sides = [float(np.linalg.norm(new[i] - new[(i + 1) % 4])) for i in range(4)]
    side = float(np.mean(sides))
    if abs(side - size) > tolerance_mm:
        warns.append(f"Glass measures {side:.2f} mm a side, the reference is {size:g} mm. Check its corners.")
    if rms > tolerance_mm:
        warns.append(f"The corners do not fit a square well ({rms:.2f} mm off). Check them.")
    centre_shift = (R @ ref_ccw.mean(0) + t) - ref_ccw.mean(0)
    return Match(math.degrees(th), centre_shift.tolist(), rms, side, new_red_mm is not None, R, t, warns)


# ---- G-code ---------------------------------------------------------------------------

_WORD = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")


@dataclass
class GcodeStats:
    moves: int
    x: tuple[float, float]
    y: tuple[float, float]
    z_cut: float | None


def _split(line: str) -> tuple[str, str]:
    """Code and comment of one line (';' to the end, or '(...)')."""
    code, sep, comment = line.partition(";")
    code = re.sub(r"\([^)]*\)", "", code)
    return code, (sep + comment) if sep else ""


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def transform_gcode(text: str, R: np.ndarray | None = None, t: np.ndarray | None = None,
                    header: str = "") -> tuple[str, list[dict]]:
    """Rotate and shift the X/Y of every move. Returns the new G-code and the
    path as segments ({x0,y0,x1,y1,cut}) in the new coordinates.

    Handles G0/G1/G2/G3 (arcs: I/J rotated with the move), G90/G91, and moves
    that give only X or only Y (both are written out, since a turned move
    changes both). Refuses inches (G20) and G92 position changes in X/Y,
    which cannot be moved safely."""
    R = np.eye(2) if R is None else np.asarray(R, dtype=np.float64)
    t = np.zeros(2) if t is None else np.asarray(t, dtype=np.float64)
    out = [header.rstrip("\n")] if header else []
    absolute, motion = True, None
    x = y = z = None
    segs: list[dict] = []
    for n, line in enumerate(text.splitlines(), 1):
        code, comment = _split(line)
        words = [(m.group(1).upper(), float(m.group(2))) for m in _WORD.finditer(code)]
        gs = [int(v) for k, v in words if k == "G"]
        if 20 in gs:
            raise ReferenceError(f"line {n}: inches (G20) are not supported, use G21")
        if 92 in gs and any(k in "XY" for k, _ in words):
            raise ReferenceError(f"line {n}: G92 changes the X/Y position; this file cannot be moved safely")
        if 90 in gs:
            absolute = True
        if 91 in gs:
            absolute = False
        for g in gs:
            if g in (0, 1, 2, 3):
                motion = g
        vals = dict(words)
        if "Z" in vals and motion is not None:
            z = vals["Z"] if absolute or z is None else z + vals["Z"]
        if motion is None or not ("X" in vals or "Y" in vals) or any(g in (28, 92, 53) for g in gs):
            out.append(line)
            continue
        if absolute:
            nx = vals.get("X", x if x is not None else 0.0)
            ny = vals.get("Y", y if y is not None else 0.0)
            px, py = R @ np.array([nx, ny]) + t
        else:
            d = R @ np.array([vals.get("X", 0.0), vals.get("Y", 0.0)])
            nx = (x or 0.0) + vals.get("X", 0.0)
            ny = (y or 0.0) + vals.get("Y", 0.0)
            px, py = d
        new_words = {"X": _fmt(px), "Y": _fmt(py)}
        if motion in (2, 3) and ("I" in vals or "J" in vals):
            ij = R @ np.array([vals.get("I", 0.0), vals.get("J", 0.0)])
            new_words.update({"I": _fmt(ij[0]), "J": _fmt(ij[1])})
        # rebuild the code part: keep every other word in place, X/Y (and I/J) rewritten
        parts, done = [], set()
        for m in _WORD.finditer(code):
            k = m.group(1).upper()
            if k in new_words:
                if k not in done:
                    parts.append(k + new_words[k])
                    done.add(k)
            else:
                parts.append(m.group(0).replace(" ", ""))
        after = {"X": "YZIJFE", "Y": "ZIJFE", "I": "JFE", "J": "FE"}     # usual word order
        for k in ("X", "Y", "I", "J"):
            if k in new_words and k not in done:
                i = next((j for j, p in enumerate(parts) if p[0].upper() in after[k]), len(parts))
                parts.insert(i, k + new_words[k])
                done.add(k)
        out.append(" ".join(parts) + (" " + comment if comment else ""))
        if x is not None and y is not None:
            a, b = R @ np.array([x, y]) + t, R @ np.array([nx, ny]) + t
            segs.append({"x0": float(a[0]), "y0": float(a[1]), "x1": float(b[0]), "y1": float(b[1]),
                         "z": z, "g": motion})
        x, y = nx, ny
    zs = [s["z"] for s in segs if s["z"] is not None]
    zcut = min(zs) if zs else None
    for s in segs:
        s["cut"] = s["g"] != 0 and (zcut is None or (s["z"] is not None and s["z"] <= zcut + 1e-6))
    return "\n".join(out) + "\n", segs


def gcode_stats(text: str) -> GcodeStats:
    _, segs = transform_gcode(text)
    if not segs:
        raise ReferenceError("no X/Y moves found in this file")
    xs = [v for s in segs for v in (s["x0"], s["x1"])]
    ys = [v for s in segs for v in (s["y0"], s["y1"])]
    zs = [s["z"] for s in segs if s["cut"] and s["z"] is not None]
    return GcodeStats(len(segs), (min(xs), max(xs)), (min(ys), max(ys)), min(zs) if zs else None)
