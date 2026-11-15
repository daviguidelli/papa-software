"""Scrub motion: turns a bridge sketch into a serpentine raster for the swab.

Geometry lives in the substrate frame (mm, origin top-left of the pattern area,
x right, y down). The path starts at the top-left and, with an odd number of
rows, ends at the bottom-right. Each bridge is a keep-out rectangle grown by the
swab radius plus a margin, so its corners come out rounded. Rows are cut where
they cross a grown keep-out; the swab lifts and hops across each gap.

Placement onto the machine is one function (`to_machine`) so a skew correction
can later be a rotation plus offset applied there and nowhere else.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from papa.ops.geometry import to_machine

DEFAULTS: dict[str, Any] = {
    "swab_mm": 1.0,           # contact width of the swab
    "margin_mm": 0.1,         # extra clearance around every bridge
    "pitch_frac": 0.7,        # row spacing as a fraction of the swab width
    "scrub_mm_s": 2.0,        # speed while in contact
    "travel_mm_s": 20.0,      # XY speed while lifted
    "z_mm_s": 2.0,            # Z speed, down and up
    "lift_mm": 3.0,           # travel Z = contact Z + this
    "start": None,            # {"x","y","z","ref"} captured by jogging
    "area": {"w": 25.0, "h": 25.0},
    "bridges": [],            # [{"id","cx","cy","l","w"}] l along x, w along y
}

RANGES = {
    "swab_mm": (0.1, 10.0), "margin_mm": (0.0, 5.0), "pitch_frac": (0.2, 1.0),
    "scrub_mm_s": (0.1, 30.0), "travel_mm_s": (1.0, 150.0), "z_mm_s": (0.1, 10.0),
    "lift_mm": (0.5, 50.0),
}


class ScrubError(RuntimeError):
    pass


class Aborted(ScrubError):
    pass


def merged(raw: dict | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    return out


def clean_bridges(items: Any) -> list[dict]:
    out = []
    for i, b in enumerate(items or []):
        try:
            cx, cy, l, w = (float(b[k]) for k in ("cx", "cy", "l", "w"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ScrubError(f"bridge {i + 1} is missing cx, cy, l or w") from exc
        if l <= 0 or w <= 0:
            raise ScrubError(f"bridge {i + 1} needs a positive length and width")
        out.append({"id": b.get("id", i + 1), "cx": cx, "cy": cy, "l": l, "w": w})
    return out


def _dist_to_rect(px: float, py: float, b: dict) -> float:
    dx = max(abs(px - b["cx"]) - b["l"] / 2, 0.0)
    dy = max(abs(py - b["cy"]) - b["w"] / 2, 0.0)
    return math.hypot(dx, dy)


def _blocked(y: float, bridges: list[dict], r: float) -> list[tuple[float, float]]:
    spans = []
    for b in bridges:
        y0, y1 = b["cy"] - b["w"] / 2, b["cy"] + b["w"] / 2
        dy = max(y0 - y, y - y1, 0.0)
        if dy >= r:
            continue
        half = math.sqrt(r * r - dy * dy)
        spans.append((b["cx"] - b["l"] / 2 - half, b["cx"] + b["l"] / 2 + half))
    spans.sort()
    merged_spans: list[list[float]] = []
    for a, c in spans:
        if merged_spans and a <= merged_spans[-1][1]:
            merged_spans[-1][1] = max(merged_spans[-1][1], c)
        else:
            merged_spans.append([a, c])
    return [(a, c) for a, c in merged_spans]


def _free(y: float, w: float, bridges: list[dict], r: float) -> list[tuple[float, float]]:
    out, cur = [], 0.0
    for a, c in _blocked(y, bridges, r):
        if a > cur:
            out.append((cur, min(a, w)))
        cur = max(cur, c)
        if cur >= w:
            break
    if cur < w:
        out.append((cur, w))
    return [(a, c) for a, c in out if c - a > 1e-6 and a < w and c > 0]


def _line_clear(x0: float, y0: float, x1: float, y1: float, bridges: list[dict], r: float) -> bool:
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.05))
    for i in range(n + 1):
        t = i / n
        px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        if any(_dist_to_rect(px, py, b) < r - 1e-6 for b in bridges):
            return False
    return True


def plan(settings: dict[str, Any]) -> dict[str, Any]:
    st = merged(settings)
    w, h = float(st["area"]["w"]), float(st["area"]["h"])
    sw, margin = float(st["swab_mm"]), float(st["margin_mm"])
    bridges = clean_bridges(st["bridges"])
    r = sw / 2 + margin
    pitch = sw * float(st["pitch_frac"])
    rows = int(math.ceil(h / pitch)) + 1
    if rows % 2 == 0:
        rows += 1
    ys = [h * i / (rows - 1) for i in range(rows)] if rows > 1 else [0.0]

    segs: list[dict] = []
    cur: tuple[float, float] | None = None
    for i, y in enumerate(ys):
        free = _free(y, w, bridges, r)
        if i % 2:
            free = [(c, a) for a, c in reversed(free)]
        for a, c in free:
            if cur is not None:
                contact = (abs(cur[0] - a) < 1e-6 and _line_clear(cur[0], cur[1], a, y, bridges, r)
                           and abs(cur[1] - y) <= pitch * 1.5 + 1e-6)
                if not (abs(cur[0] - a) < 1e-6 and abs(cur[1] - y) < 1e-9):
                    segs.append({"kind": "drag" if contact else "hop",
                                 "x0": cur[0], "y0": cur[1], "x1": a, "y1": y})
            segs.append({"kind": "scrub", "x0": a, "y0": y, "x1": c, "y1": y})
            cur = (c, y)

    for s in segs:
        for k in ("x0", "y0", "x1", "y1"):
            s[k] = round(s[k], 4)
    scrub_len = sum(math.hypot(s["x1"] - s["x0"], s["y1"] - s["y0"])
                    for s in segs if s["kind"] != "hop")
    hops = sum(1 for s in segs if s["kind"] == "hop")
    time_s = scrub_len / float(st["scrub_mm_s"])

    gx = np.arange(0.125, w, 0.25)
    gy = np.arange(0.125, h, 0.25)
    X, Y = np.meshgrid(gx, gy)
    covered = np.zeros_like(X, dtype=bool)
    for s in segs:
        if s["kind"] == "hop":
            continue
        ax, ay, bx, by = s["x0"], s["y0"], s["x1"], s["y1"]
        dx, dy = bx - ax, by - ay
        ln = dx * dx + dy * dy
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / ln, 0.0, 1.0) if ln > 0 else 0.0
        covered |= np.hypot(X - (ax + t * dx), Y - (ay + t * dy)) <= sw / 2
    inside = np.zeros_like(covered)
    for b in bridges:
        inside |= (np.abs(X - b["cx"]) <= b["l"] / 2) & (np.abs(Y - b["cy"]) <= b["w"] / 2)
    want = ~inside
    missed = want & ~covered
    mask = [
        "".join("x" if missed[j, i] else "." for i in range(0, missed.shape[1], 2))
        for j in range(0, missed.shape[0], 2)
    ]
    return {
        "area": {"w": w, "h": h}, "bridges": bridges, "radius_mm": r, "pitch_mm": pitch, "rows": rows,
        "segments": segs, "hops": hops, "scrub_mm": round(scrub_len, 2),
        "est_s": round(time_s, 1),
        "uncovered_pct": round(100.0 * float(missed.sum()) / max(1, int(want.sum())), 2),
        "uncovered_mask": mask,
        "start": [0.0, 0.0], "end": [segs[-1]["x1"], segs[-1]["y1"]] if segs else [0.0, 0.0],
    }


def problems(settings: dict[str, Any], machine: Any, ref: str | None, floor: float | None) -> list[str]:
    st = merged(settings)
    out: list[str] = []
    s = st["start"]
    if not s:
        return ["Jog to the top-left of the pattern area, then press Set start here."]
    if ref is not None and s.get("ref") != ref:
        out.append("The Z reference changed after you set the start. Set it again.")
    z = float(s["z"])
    if floor is None:
        out.append("Set the Z floor first (at or below the contact Z).")
    elif z < floor - 1e-6:
        out.append("The contact Z is below the Z floor.")
    if z + float(st["lift_mm"]) > machine.limits["z"][1]:
        out.append("Travel Z would go above the Z limit; lower the lift.")
    w, h = float(st["area"]["w"]), float(st["area"]["h"])
    for lx, ly in ((0, 0), (w, 0), (w, h), (0, h)):
        mx, my = to_machine(machine.axes, float(s["x"]), float(s["y"]), lx, ly)
        if not machine.in_limits(mx, my, z):
            out.append("The pattern area runs outside the machine limits from this start.")
            break
    try:
        clean_bridges(st["bridges"])
    except ScrubError as exc:
        out.append(str(exc))
    return out


def describe(settings: dict[str, Any], machine: Any, ref: str | None, floor: float | None) -> dict[str, Any]:
    st = merged(settings)
    bad = problems(st, machine, ref, floor)
    try:
        p = plan(st)
        summary = {k: p[k] for k in ("rows", "hops", "scrub_mm", "est_s", "uncovered_pct", "end")}
    except ScrubError as exc:
        summary = None
        bad.append(str(exc))
    return {"settings": {k: v for k, v in st.items() if k != "bridges"},
            "bridges": len(clean_bridges(st["bridges"])) if summary else 0,
            "plan": summary, "problems": bad, "ready": not bad}


@dataclass
class ScrubStatus:
    running: bool = False
    finished: bool = False
    error: str = ""
    row: int = 0
    rows: int = 0
    done: int = 0
    total: int = 0

    def as_dict(self) -> dict:
        return {"running": self.running, "finished": self.finished, "error": self.error,
                "row": self.row, "rows": self.rows, "done": self.done, "total": self.total}


class ScrubRunner:
    """Runs the plan on a thread. Stop takes effect between segments; E-STOP is
    still the only thing that halts a move already under way."""

    def __init__(self, printer: Callable[[], Any], machine: Any) -> None:
        self._printer = printer
        self.machine = machine
        self.status = ScrubStatus()
        self._abort = threading.Event()
        self.thread: threading.Thread | None = None

    def _check(self) -> None:
        if self._abort.is_set():
            raise Aborted("stopped between moves")

    def start(self, settings: dict[str, Any], ref: str | None, floor: float | None) -> threading.Thread:
        st = merged(settings)
        bad = problems(st, self.machine, ref, floor)
        if bad:
            raise ScrubError(" ".join(bad))
        p = plan(st)
        self._abort.clear()
        self.status = ScrubStatus(running=True, rows=p["rows"], total=len(p["segments"]))
        self.thread = threading.Thread(target=self._run, args=(st, p), daemon=True)
        self.thread.start()
        return self.thread

    def stop(self) -> None:
        self._abort.set()

    def _run(self, st: dict[str, Any], p: dict[str, Any]) -> None:
        s = self.status
        pr = self._printer
        sx, sy, cz = float(st["start"]["x"]), float(st["start"]["y"]), float(st["start"]["z"])
        tz = cz + float(st["lift_mm"])
        zs, ts, cs = float(st["z_mm_s"]), float(st["travel_mm_s"]), float(st["scrub_mm_s"])
        axes = self.machine.axes
        lifted = False

        def xy(lx: float, ly: float, speed: float) -> None:
            mx, my = to_machine(axes, sx, sy, lx, ly)
            pr().move(x=mx, y=my, speed_mm_s=speed)

        def up() -> None:
            nonlocal lifted
            pr().move(z=tz, speed_mm_s=zs)
            lifted = True

        def down() -> None:
            nonlocal lifted
            pr().move(z=cz, speed_mm_s=zs)
            lifted = False

        try:
            pos = pr().position()
            if not all(a in pos.homed for a in "xyz"):
                raise ScrubError("Home X and Y and set Z first.")
            self._check()
            up()
            self._check()
            xy(0.0, 0.0, ts)
            self._check()
            down()
            cur_row_y = 0.0
            row = 1
            for i, seg in enumerate(p["segments"]):
                self._check()
                if seg["kind"] == "hop":
                    up()
                    xy(seg["x1"], seg["y1"], ts)
                    self._check()
                    down()
                else:
                    xy(seg["x1"], seg["y1"], cs)
                if seg["kind"] == "scrub" and seg["y1"] > cur_row_y + 1e-6:
                    row += 1
                    cur_row_y = seg["y1"]
                s.row, s.done = row, i + 1
            up()
            s.finished = True
        except Aborted as exc:
            s.error = str(exc)
            self._safe_lift(tz, zs, lifted)
        except Exception as exc:                          # noqa: BLE001
            s.error = str(exc) or type(exc).__name__
        finally:
            s.running = False

    def _safe_lift(self, tz: float, zs: float, lifted: bool) -> None:
        if lifted:
            return
        try:
            self._printer().move(z=tz, speed_mm_s=zs)
        except Exception:                                  # noqa: BLE001
            pass
