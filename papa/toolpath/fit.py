"""Fit strokes into the pattern area, in slide-local coordinates.

Slide-local: origin at the slide's top-left corner, +x right, +y down.
DXF is y-up, so y is flipped here once, and never again.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dxf_loader import Stroke


@dataclass(frozen=True)
class FitResult:
    strokes: list[Stroke]
    scale: float
    source_size: tuple[float, float]
    fitted_size: tuple[float, float]


def bounds(strokes: list[Stroke]) -> tuple[float, float, float, float]:
    xs = [p[0] for s in strokes for p in s.points]
    ys = [p[1] for s in strokes for p in s.points]
    if not xs:
        raise ValueError("no geometry to fit")
    return min(xs), min(ys), max(xs), max(ys)


def fit(
    strokes: list[Stroke],
    area_w_mm: float,
    area_h_mm: float,
    inset_mm: float = 0.0,
    center: bool = True,
) -> FitResult:
    """Scale uniformly to fit the area, flip to y-down, translate into place.

    Uniform scale only: scaling x and y independently would distort the
    pattern, and a distorted pattern is not the pattern that was drawn.
    """
    if area_w_mm <= 0 or area_h_mm <= 0:
        raise ValueError("pattern area must be positive")
    x0, y0, x1, y1 = bounds(strokes)
    w, h = x1 - x0, y1 - y0
    if w == 0 and h == 0:
        raise ValueError("geometry has no extent")

    sx = area_w_mm / w if w > 0 else float("inf")
    sy = area_h_mm / h if h > 0 else float("inf")
    scale = min(sx, sy)

    fw, fh = w * scale, h * scale
    ox = inset_mm + ((area_w_mm - fw) / 2 if center else 0.0)
    oy = inset_mm + ((area_h_mm - fh) / 2 if center else 0.0)

    out: list[Stroke] = []
    for s in strokes:
        pts = [
            (ox + (p[0] - x0) * scale, oy + (y1 - p[1]) * scale)  # y flip here
            for p in s.points
        ]
        out.append(Stroke(points=pts))
    return FitResult(
        strokes=out, scale=scale, source_size=(w, h), fitted_size=(fw, fh)
    )


def to_svg(strokes: list[Stroke], area_w_mm: float, area_h_mm: float) -> str:
    """Preview exactly what will be scribed, in slide-local mm."""
    paths = []
    for s in strokes:
        d = "M " + " L ".join(f"{p[0]:.3f} {p[1]:.3f}" for p in s.points)
        paths.append(f'<path d="{d}" fill="none" stroke="#4C9BE8" stroke-width="0.25"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {area_w_mm} {area_h_mm}" '
        f'width="{area_w_mm}mm" height="{area_h_mm}mm">'
        f'<rect x="0" y="0" width="{area_w_mm}" height="{area_h_mm}" fill="none" '
        f'stroke="#D9B45A" stroke-width="0.15" stroke-dasharray="1 1"/>'
        + "".join(paths)
        + "</svg>"
    )
