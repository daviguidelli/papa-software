"""How close the scribed lines come to each other.

A swab does not scribe a mathematical line. At a given contact force it removes
a band of film of some width - the kerf - centred on the toolpath. Two paths
closer together than one kerf come out as a single wide clearing rather than
two lines, and no amount of positional accuracy fixes that.

The kerf is a property of the swab, the force, the speed and the film, so it
has to be measured rather than derived. See docs/EXPERIMENTS.md. Until it is
measured, this module still answers the half of the question that is pure
geometry: how close do the paths actually come?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dxf_loader import Stroke


@dataclass(frozen=True)
class Spacing:
    """Closest approach between parts of the pattern that are not neighbours."""
    min_distance_mm: float
    at_point_mm: tuple[float, float] | None
    between_strokes: tuple[int, int] | None
    crossings: int = 0

    def merges_at(self, kerf_mm: float) -> bool:
        """Whether a kerf this wide would fuse the two closest paths."""
        return self.min_distance_mm < kerf_mm

    def margin_mm(self, kerf_mm: float) -> float:
        """Clear film left between the two closest cuts. Negative means fused."""
        return self.min_distance_mm - kerf_mm


def _segments(strokes: list[Stroke]) -> list[tuple[tuple[float, float], tuple[float, float], int]]:
    out = []
    for i, stroke in enumerate(strokes):
        for a, b in zip(stroke.points, stroke.points[1:]):
            if a != b:
                out.append((a, b, i))
    return out


def crossing_pairs(strokes: list[Stroke]) -> tuple[set[tuple[int, int]], int]:
    """Which strokes genuinely cross, and how many crossings there are."""
    pairs: set[tuple[int, int]] = set()
    count = 0
    for (x, y, si, sj) in _intersections(strokes):
        pairs.add((min(si, sj), max(si, sj)))
        count += 1
    return pairs, count


def intersections(strokes: list[Stroke]) -> list[tuple[float, float]]:
    """Where paths genuinely cross, solved exactly rather than sampled.

    Sampling cannot find a crossing: two paths that meet at a point have no
    sampled points in common, so they look like a near-miss of about half the
    sample step. Solving the segment pairs gives the real answer, and lets the
    near-miss search ignore the places where lines are supposed to meet.
    """
    return [(x, y) for (x, y, _, _) in _intersections(strokes)]


def _intersections(strokes: list[Stroke]) -> list[tuple[float, float, int, int]]:
    segs = _segments(strokes)
    hits: list[tuple[float, float, int, int]] = []
    for i in range(len(segs)):
        (p1, p2, si) = segs[i]
        for j in range(i + 1, len(segs)):
            (p3, p4, sj) = segs[j]
            if si == sj and j - i <= 1:
                continue            # consecutive segments share an endpoint
            d = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
            if abs(d) < 1e-12:
                continue            # parallel
            t = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
            u = ((p3[0] - p1[0]) * (p2[1] - p1[1]) - (p3[1] - p1[1]) * (p2[0] - p1[0])) / d
            if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
                hits.append((p1[0] + t * (p2[0] - p1[0]),
                             p1[1] + t * (p2[1] - p1[1]), si, sj))
    return hits


def _sample(strokes: list[Stroke], step_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample every stroke to a fixed spacing: points, stroke index, arclength."""
    pts: list[tuple[float, float]] = []
    sid: list[int] = []
    arc: list[float] = []
    for i, stroke in enumerate(strokes):
        travelled = 0.0
        pts.append(stroke.points[0])
        sid.append(i)
        arc.append(0.0)
        for a, b in zip(stroke.points, stroke.points[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d == 0:
                continue
            n = max(1, int(d // step_mm))
            for k in range(1, n + 1):
                t = k / n
                pts.append((a[0] + dx * t, a[1] + dy * t))
                sid.append(i)
                arc.append(travelled + d * t)
            travelled += d
    return np.array(pts, dtype=float), np.array(sid), np.array(arc, dtype=float)


def closest_approach(
    strokes: list[Stroke],
    step_mm: float = 0.2,
    ignore_along_stroke_mm: float = 1.0,
    fold_ratio: float = 0.9,
) -> Spacing:
    """Smallest distance between two parts of the pattern that are not neighbours.

    Two points on the same stroke only count when the stroke has folded back
    toward itself: the straight-line distance has to be clearly shorter than
    the distance along the path. Without that test a straight line looks like
    it approaches itself everywhere, which says nothing about merging. A
    hairpin, which genuinely can fuse, still counts.
    """
    if len(strokes) == 0:
        return Spacing(float("inf"), None, None)

    pts, sid, arc = _sample(strokes, step_mm)
    n = len(pts)
    if n < 2:
        return Spacing(float("inf"), None, None)

    best = float("inf")
    best_at: tuple[float, float] | None = None
    best_pair: tuple[int, int] | None = None
    # Strokes that cross are excluded from the near-miss search entirely.
    # Close to a crossing two paths are genuinely close, and with any kerf at
    # all they merge there - but that is the drawing asking them to meet, not a
    # spacing problem. The limitation: a pair that crosses once and also runs
    # parallel elsewhere is not checked along that parallel run.
    crossed, n_cross = crossing_pairs(strokes)

    # Brute force in chunks. Patterns here are a few thousand points at most,
    # and being obviously correct beats being clever.
    chunk = 512
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = pts[start:stop]
        d = np.linalg.norm(block[:, None, :] - pts[None, :, :], axis=2)

        same = sid[start:stop][:, None] == sid[None, :]
        gap = np.abs(arc[start:stop][:, None] - arc[None, :])
        # on one stroke: ignore near neighbours, and ignore anything that has
        # not folded back (straight run, where distance along == distance across)
        not_folded = d >= gap * fold_ratio
        d[same & ((gap <= ignore_along_stroke_mm) | not_folded)] = np.inf
        idx = np.arange(start, stop)
        d[idx - start, idx] = np.inf

        if crossed:
            si = sid[start:stop][:, None]
            sj = sid[None, :]
            lo, hi = np.minimum(si, sj), np.maximum(si, sj)
            mask = np.zeros(d.shape, dtype=bool)
            for (a, b) in crossed:
                mask |= (lo == a) & (hi == b)
            d[mask] = np.inf

        local = float(d.min())
        if local < best:
            i, j = np.unravel_index(int(d.argmin()), d.shape)
            best = local
            best_at = (float(block[i][0]), float(block[i][1]))
            best_pair = (int(sid[start + i]), int(sid[j]))

    return Spacing(best, best_at, best_pair, crossings=n_cross)


def report(strokes: list[Stroke], kerf_mm: float | None, step_mm: float = 0.2) -> dict:
    """What the GUI and the CLI show about spacing before anything is scribed."""
    s = closest_approach(strokes, step_mm=step_mm)
    finite = s.min_distance_mm != float("inf")
    out: dict = {
        "min_distance_mm": round(s.min_distance_mm, 4) if finite else None,
        "at_mm": s.at_point_mm,
        "between_strokes": s.between_strokes,
        "crossings": s.crossings,
        "kerf_mm": kerf_mm,
    }
    if kerf_mm and finite:
        out["margin_mm"] = round(s.margin_mm(kerf_mm), 4)
        out["merges"] = s.merges_at(kerf_mm)
        out["warning"] = (
            f"two paths come within {s.min_distance_mm:.3f} mm, closer than the "
            f"{kerf_mm:.3f} mm kerf: they will scribe as one line"
            if s.merges_at(kerf_mm) else None
        )
    elif not kerf_mm:
        out["warning"] = (
            "kerf_mm is not measured yet, so nothing can be said about whether "
            "these lines stay separate. See docs/EXPERIMENTS.md"
        )
    return out
