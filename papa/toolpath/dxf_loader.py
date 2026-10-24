"""DXF to strokes.

Scope is straight segments only: LINE, LWPOLYLINE and POLYLINE. Anything else
is rejected by name rather than silently approximated, because a swab dragged
along a wrong path ruins the slide.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ezdxf

Point = tuple[float, float]
SUPPORTED = {"LINE", "LWPOLYLINE", "POLYLINE"}


class UnsupportedEntity(ValueError):
    pass


@dataclass
class Stroke:
    """One continuous pen-down path in DXF units."""
    points: list[Point]

    @property
    def length(self) -> float:
        return sum(
            ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            for a, b in zip(self.points, self.points[1:])
        )


def _segments(path: Path) -> list[tuple[Point, Point]]:
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    segs: list[tuple[Point, Point]] = []
    rejected: set[str] = set()

    for e in msp:
        kind = e.dxftype()
        if kind == "LINE":
            s, t = e.dxf.start, e.dxf.end
            segs.append(((float(s.x), float(s.y)), (float(t.x), float(t.y))))
        elif kind == "LWPOLYLINE":
            pts = [(float(p[0]), float(p[1])) for p in e.get_points("xy")]
            if e.closed and len(pts) > 2:
                pts.append(pts[0])
            segs.extend(zip(pts, pts[1:]))
        elif kind == "POLYLINE":
            if not e.is_2d_polyline:
                rejected.add("POLYLINE (3D)")
                continue
            pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
            if e.is_closed and len(pts) > 2:
                pts.append(pts[0])
            segs.extend(zip(pts, pts[1:]))
        else:
            rejected.add(kind)

    if rejected:
        raise UnsupportedEntity(
            "this DXF contains entities PAPA cannot scribe: "
            + ", ".join(sorted(rejected))
            + ". Only straight lines and polylines are supported."
        )
    return [s for s in segs if s[0] != s[1]]


def _key(p: Point, tol: float) -> tuple[int, int]:
    return (round(p[0] / tol), round(p[1] / tol))


def chain(segments: list[tuple[Point, Point]], tol: float = 1e-6) -> list[Stroke]:
    """Join segments that share endpoints into continuous strokes.

    Walking shared endpoints means a square drawn as four separate LINEs is
    scribed in one pen-down pass instead of four, which matters because every
    lift and re-approach is another Z move and another chance to miss.
    """
    adj: dict[tuple[int, int], list[int]] = {}
    for i, (a, b) in enumerate(segments):
        adj.setdefault(_key(a, tol), []).append(i)
        adj.setdefault(_key(b, tol), []).append(i)

    used = [False] * len(segments)
    strokes: list[Stroke] = []

    def walk(point: Point) -> list[Point]:
        out: list[Point] = []
        cur = point
        while True:
            nxt = None
            for i in adj.get(_key(cur, tol), []):
                if used[i]:
                    continue
                a, b = segments[i]
                if _key(a, tol) == _key(cur, tol):
                    nxt, used[i] = b, True
                    break
                if _key(b, tol) == _key(cur, tol):
                    nxt, used[i] = a, True
                    break
            if nxt is None:
                return out
            out.append(nxt)
            cur = nxt

    for i, (a, b) in enumerate(segments):
        if used[i]:
            continue
        used[i] = True
        fwd = [a, b] + walk(b)
        back = walk(a)
        pts = list(reversed(back)) + fwd
        strokes.append(Stroke(points=pts))
    return strokes


def order_strokes(strokes: list[Stroke]) -> list[Stroke]:
    """Greedy nearest-start ordering to shorten pen-up travel."""
    if not strokes:
        return []
    remaining = list(strokes)
    out = [remaining.pop(0)]
    while remaining:
        cur = out[-1].points[-1]

        def d(s: Stroke) -> float:
            p = s.points[0]
            return (p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2

        remaining.sort(key=d)
        out.append(remaining.pop(0))
    return out


def load(path: str | Path) -> list[Stroke]:
    """Read a DXF and return ordered strokes in DXF units."""
    return order_strokes(chain(_segments(Path(path))))
