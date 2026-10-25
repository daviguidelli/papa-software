"""Gantry control. Needs: printer on the network, nothing else.

Klipper reports where it COMMANDED the toolhead, not where it is. Everything
here checks software agreement; real distance needs a ruler (--ruler).

  PAPA_SIM=1 PAPA_REAL=printer python -m scripts.bench.b1_gantry [--ruler]
"""
from __future__ import annotations

import argparse
import statistics

from papa.config import load_machine
from papa.hw import make_printer

from ._common import Report, main


def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler", action="store_true", help="also check a 50 mm move by hand")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    m = load_machine()
    r = Report("b1_gantry")
    p = make_printer(m)

    # 1. link
    ping = getattr(p, "ping", lambda: True)()
    if not r.check("printer answers on the network", ping, m.moonraker_url):
        r.note("fix: IP in config/machine.yaml, then Moonraker trusted_clients")
        return r.finish()
    info = p.info() if hasattr(p, "info") else {}
    r.check("Klipper state is ready", info.get("state", "ready") == "ready", str(info.get("state")))
    r.note("stand next to the power switch before homing")

    # 2. home
    p.home()
    pos = p.position()
    r.check("all axes report homed", pos.is_homed, pos.homed)

    # 3. small relative moves, each way, return to start. Homing leaves the
    # toolhead at an endstop, where a negative jog would leave the bed.
    sp = m.speeds["travel"] / 3
    p.move(z=30, speed_mm_s=10)
    p.move(x=100, y=100, speed_mm_s=sp)
    start = p.position()
    for axis in "xyz":
        for d in (1.0, -1.0):
            if axis == "z" and d < 0:
                continue          # never test downward Z blind
            kw = {"d" + axis: d}
            p.move_relative(speed_mm_s=sp, **kw)
            got = getattr(p.position(), axis) - getattr(start, axis)
            r.check(f"{axis.upper()} {d:+.0f} mm reads back", abs(got - d) < 0.05, f"{got:+.3f}")
            p.move_relative(speed_mm_s=sp, **{"d" + axis: -d})

    # 4. out-of-range is refused rather than attempted
    try:
        p.move(x=m.limits["x"][1] + 50, speed_mm_s=20)
        refused = False
    except Exception:
        refused = True
    r.check("a move past the X limit is refused", refused)
    p.script("M400") if hasattr(p, "script") else None

    # 5. repeatability of a commanded target
    errs = []
    for _ in range(args.repeats):
        p.move(x=100, y=100, z=30, speed_mm_s=sp)
        p.move(x=200, y=180, z=30, speed_mm_s=sp)
        p.move(x=100, y=100, z=30, speed_mm_s=sp)
        q = p.position()
        errs.append(max(abs(q.x - 100), abs(q.y - 100)))
    r.check(f"{args.repeats} round trips return to the same commanded point",
            max(errs) < 0.02, f"worst {max(errs):.4f} mm")

    # 6. timing: an M400 move should not return early
    import time
    t = time.time()
    p.move(x=250, y=100, z=30, speed_mm_s=20)
    dt = time.time() - t
    r.check("a 150 mm move at 20 mm/s blocks until finished", dt > 5 or "Mock" in type(p).__name__,
            f"{dt:.1f} s")

    if args.ruler:
        p.move(x=100, y=100, z=30, speed_mm_s=sp)
        input("  mark the toolhead position on a ruler taped to the bed, then Enter ")
        p.move(x=150, y=100, z=30, speed_mm_s=sp)
        meas = float(input("  measured travel in mm (should be 50): "))
        r.check("ruler agrees with a 50 mm commanded move", abs(meas - 50) < 0.5, f"{meas}")
        r.data["ruler_error_mm"] = meas - 50

    p.move(z=30, speed_mm_s=10)
    return r.finish()


if __name__ == "__main__":
    main(run)
