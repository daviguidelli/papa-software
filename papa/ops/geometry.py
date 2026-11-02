"""Slide-local to machine coordinates, through the configured axis mapping.

A rotated mounting base is a config change, so this is the only place that
knows which way local +y runs. The orchestrator and the validation page both
call it, so a traced square and a scribed square can never disagree.
"""
from __future__ import annotations


def to_machine(axes: dict[str, str], slot_x: float, slot_y: float,
               lx: float, ly: float) -> tuple[float, float]:
    dx = lx if axes["local_x"] == "+X" else -lx
    dy = ly if axes["local_y"] == "+Y" else -ly
    if axes["local_x"] in ("+Y", "-Y") or axes["local_y"] in ("+X", "-X"):
        dx, dy = dy, dx
    return slot_x + dx, slot_y + dy


def square(w: float, h: float) -> list[tuple[float, float]]:
    """Corners in slide-local mm, clockwise from the origin (top-left)."""
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
