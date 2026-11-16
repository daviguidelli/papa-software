"""Slide-local to machine coordinates, through the configured axis mapping.

A rotated mounting base is a config change, so this is the only place that
knows which way local +y runs. The orchestrator and the validation page both
call it, so a traced square and a scribed square can never disagree.

A substrate that sits a little turned is handled here too: `rot_deg` turns the
local pattern about its origin in the machine frame (positive from +X toward
+Y), after the axis mapping. With rot_deg = 0 the result is exactly the plain
axis mapping.
"""
from __future__ import annotations

import math

_UNIT = {"+X": (1.0, 0.0), "-X": (-1.0, 0.0), "+Y": (0.0, 1.0), "-Y": (0.0, -1.0)}


def axis_matrix(axes: dict[str, str]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Machine direction of local +x and local +y: ((xx, xy), (yx, yy))."""
    return _UNIT[axes["local_x"]], _UNIT[axes["local_y"]]


def to_machine(axes: dict[str, str], slot_x: float, slot_y: float,
               lx: float, ly: float, rot_deg: float = 0.0) -> tuple[float, float]:
    (ax, ay), (bx, by) = axis_matrix(axes)
    dx, dy = lx * ax + ly * bx, lx * ay + ly * by
    if rot_deg:
        c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        dx, dy = c * dx - s * dy, s * dx + c * dy
    return slot_x + dx, slot_y + dy


def to_local(axes: dict[str, str], slot_x: float, slot_y: float,
             mx: float, my: float, rot_deg: float = 0.0) -> tuple[float, float]:
    """Inverse of to_machine."""
    dx, dy = mx - slot_x, my - slot_y
    if rot_deg:
        c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        dx, dy = c * dx + s * dy, -s * dx + c * dy
    (ax, ay), (bx, by) = axis_matrix(axes)
    # the axis matrix is a signed permutation, so its inverse is its transpose
    return dx * ax + dy * ay, dx * bx + dy * by


def square(w: float, h: float) -> list[tuple[float, float]]:
    """Corners in slide-local mm, clockwise from the origin (top-left)."""
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
