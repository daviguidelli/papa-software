"""Combined: gantry + load cell + gripper. Only after b1-b3 pass.

Z-probe repeatability is the number that decides everything downstream: if the
contact height wanders, scribe depth wanders with it.

  python -m scripts.bench.b5_combined --cell-x 40 --cell-y 40 --runs 10
Load a swab in the gripper first.
"""
from __future__ import annotations

import argparse
import statistics

from papa.config import load_calibration, load_machine
from papa.hw import make_gripper, make_loadcell, make_printer
from papa.ops.zcal import GripFailed, OverForce, calibrate_z

from ._common import Report, main


def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-x", type=float, required=True)
    ap.add_argument("--cell-y", type=float, required=True)
    ap.add_argument("--z-safe", type=float, required=True,
                    help="travel height just above the cell; probing descends max_travel_mm from here")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    m, cal = load_machine(), load_calibration()
    r = Report("b5_combined")
    p = make_printer(m)
    lc = make_loadcell(m, cal, p)
    g = make_gripper(m, cal)

    p.home()
    g.open()
    input("  put a swab between the jaws, Enter to close ")
    g.close()

    z = m.zcal
    zs, forces = [], []
    for i in range(args.runs):
        try:
            res = calibrate_z(
                p, lc, cell_x=args.cell_x, cell_y=args.cell_y, z_safe=args.z_safe,
                target_force_g=z["target_force_g"], abort_force_g=z["abort_force_g"],
                step_mm=z["step_mm"], max_travel_mm=z["max_travel_mm"],
                approach_speed_mm_s=m.speeds["z_approach"],
                travel_speed_mm_s=m.speeds["travel"],
                settle_readings=int(z["settle_readings"]), backoff_mm=z["backoff_mm"],
            )
        except (GripFailed, OverForce) as exc:
            r.check(f"probe {i + 1}", False, str(exc))
            break
        zs.append(res.contact_z)
        forces.append(res.force_g)
        print(f"    probe {i + 1}: z={res.contact_z:.3f}  force={res.force_g:.2f} g")
        p.move(z=args.z_safe, speed_mm_s=m.speeds["travel"])

    if len(zs) >= 3:
        sd = statistics.pstdev(zs)
        spread = max(zs) - min(zs)
        r.check("contact height repeats within one step (spread <= step_mm)",
                spread <= z["step_mm"] + 1e-9, f"spread {spread:.3f} mm, sd {sd:.3f}")
        r.data.update(mean_contact_z=statistics.mean(zs), spread_mm=spread,
                      mean_force_g=statistics.mean(forces))
        r.note(f"stiffness hint: probing resolution is {z['step_mm']} mm per step; "
               "see how many grams one step adds in the force column")
    g.open()
    return r.finish()


if __name__ == "__main__":
    main(run)
