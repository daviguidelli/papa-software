"""Servo gripper. Needs: Pi, servo on GPIO13, its own 4xAA pack, common ground.

Do the first run with NO swab and the jaws clear of anything. Find the limits
by hand before cycling:

  python -m scripts.bench.b3_servo --find          # type pulse widths, see where it binds
  python -m scripts.bench.b3_servo --cycles 20
"""
from __future__ import annotations

import argparse
import time

from papa.config import load_calibration, load_machine
from papa.hw import make_gripper

from ._common import Report, main


def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--find", action="store_true")
    ap.add_argument("--cycles", type=int, default=10)
    args = ap.parse_args()

    m, cal = load_machine(), load_calibration()
    r = Report("b3_servo")
    g = make_gripper(m, cal)

    if args.find:
        print("  type a pulse width in microseconds (500-2500), or q. Start near 1500;")
        print("  change by 50 us. Stop the moment it hums or binds.")
        while True:
            s = input("  us> ").strip()
            if s.lower().startswith("q"):
                break
            if hasattr(g, "_pulse"):
                g._pulse(int(s))
        r.note("put the open/closed values you found in calibration.yaml under gripper:")
        return r.finish()

    r.note(f"open_us={g.open_us if hasattr(g, 'open_us') else '?'} "
           f"closed_us={g.closed_us if hasattr(g, 'closed_us') else '?'}")
    g.open()
    r.ask("jaws are fully open and not humming")
    g.close()
    r.ask("jaws are fully closed on nothing, not stalled or buzzing")
    g.open()

    t = time.time()
    for _ in range(args.cycles):
        g.open()
        g.close()
    r.check(f"{args.cycles} open/close cycles completed", True, f"{time.time() - t:.1f} s")
    g.open()
    r.ask("motion was smooth, no jitter at rest")
    r.ask("a swab placed between the jaws is held firmly when closed (try pulling)")
    r.ask("servo pack voltage still above 4.8 V after cycling")
    if hasattr(g, "release"):
        g.release()
    return r.finish()


if __name__ == "__main__":
    main(run)
