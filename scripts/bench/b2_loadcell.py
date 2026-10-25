"""Load cell. Needs: Pi, HX711 wired, nothing on the cell.

  python -m scripts.bench.b2_loadcell                 # noise, drift, rate
  python -m scripts.bench.b2_loadcell --known-g 50    # also calibrate the scale

Add the weight gently; the cell is easily overloaded. Never exceed its rating.
Rating is on the part label. A 100 g cell saturates at 0.98 N, so it cannot
see a 1.1 N working force.
"""
from __future__ import annotations

import argparse
import statistics
import time

from papa.config import load_calibration, load_machine
from papa.hw import make_loadcell, make_printer

from ._common import Report, main


def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-g", type=float, default=0.0)
    ap.add_argument("--rating-g", type=float, default=0.0, help="cell rating from its label")
    ap.add_argument("--working-n", type=float, default=1.1)
    ap.add_argument("--drift-s", type=float, default=30.0)
    args = ap.parse_args()

    m, cal = load_machine(), load_calibration()
    r = Report("b2_loadcell")
    lc = make_loadcell(m, cal, make_printer(m))

    working_g = args.working_n * 1000 / 9.80665
    r.note(f"working force {args.working_n} N = {working_g:.0f} gf")
    if args.rating_g:
        r.check("cell rating comfortably above the working force (>= 2x)",
                args.rating_g >= 2 * working_g, f"rating {args.rating_g} g")
        abort = m.zcal["abort_force_g"]
        r.check("abort_force_g sits between working force and rating",
                working_g < abort < args.rating_g, f"abort {abort} g")
    else:
        r.note("pass --rating-g from the cell label to check it against the working force")

    # rate
    t = time.time()
    raw = [lc.read_counts() for _ in range(80)]
    rate = len(raw) / (time.time() - t)
    r.check("reads arrive (HX711 responding)", len(raw) == 80, f"{rate:.0f} samples/s")

    # noise at rest, after tare
    lc.tare()
    rest = [lc.grams(1) for _ in range(100)]
    sd = statistics.pstdev(rest)
    r.check("rest noise under 0.3 g (1 sigma)", sd < 0.3, f"{sd:.3f} g")
    r.check("tare brings the mean to ~0", abs(statistics.mean(rest)) < 0.3,
            f"{statistics.mean(rest):+.3f} g")

    # drift
    a = statistics.mean(lc.grams(5) for _ in range(10))
    time.sleep(args.drift_s)
    b = statistics.mean(lc.grams(5) for _ in range(10))
    r.check(f"drift over {args.drift_s:.0f} s under 0.3 g", abs(b - a) < 0.3, f"{b - a:+.3f} g")

    # scale
    if args.known_g:
        input(f"  place the {args.known_g} g weight gently on the cell, then Enter ")
        if hasattr(lc, "calibrate_with_known_weight"):
            scale = lc.calibrate_with_known_weight(args.known_g)
            r.data["scale_counts_per_g"] = scale
            r.data["tare_counts"] = lc.offset
            r.note(f"put in calibration.yaml: scale_counts_per_g={scale:.4f}, "
                   f"tare_counts={lc.offset}")
        reading = lc.grams(20)
        r.check("known weight reads within 2%", abs(reading - args.known_g) < 0.02 * args.known_g,
                f"{reading:.2f} g")
        input("  remove the weight, Enter ")
        lc.tare()
        r.check("returns to ~0 after removal", abs(lc.grams(20)) < 0.5, f"{lc.grams(20):+.2f} g")

    return r.finish()


if __name__ == "__main__":
    main(run)
