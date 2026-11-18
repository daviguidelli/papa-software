"""Z home against the load cell, and the heights that hang off it.

The probe steps the swab down onto the cell until the force reaches the target
(1.1 N), interpolates the Z where it crossed, and calls that Z `ref_z`. Every
other height is typed relative to it: slide, pickup, bin and safe Z. The slide
height is the top of the glass relative to the top of the cell, so pressing the
swab down to the slide gives the same force as pressing it onto the cell.

A check between swabs probes again but never changes Z. It reports how far the
new swab's contact moved from `ref_z`, so a different swab length is visible
without touching the reference.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from papa.ops.zcal import calibrate_z

DEFAULTS: dict[str, Any] = {
    "ref_z": 50.0,            # what Z reads at the cell contact right after Z home
    "probe_start_mm": 3.0,    # the check starts this far above the contact
    "floor_margin_mm": 0.5,   # floor sits this far below the lowest height (0 = exact)
    "up_mm": 150.0,           # ceiling above ref_z
    "slide_mm": None,         # heights, mm above the cell contact (negative = below)
    "pickup_mm": None,
    "bin_mm": None,
    "safe_mm": None,
    "ref": None,              # guard reference the last Z home created
    "homed_at": None,
    "last_check": None,       # {"drift_mm","contact_z","force_n","at"}
}
HEIGHTS = ("slide_mm", "pickup_mm", "bin_mm", "safe_mm")


class ZHomeError(RuntimeError):
    pass


def merged(raw: dict | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    return out


def absolute(st: dict, name: str) -> float | None:
    v = st.get(name)
    return None if v is None else round(float(st["ref_z"]) + float(v), 4)


def floor_z(st: dict) -> float:
    rel = [0.0] + [float(st[k]) for k in ("slide_mm", "pickup_mm", "bin_mm") if st.get(k) is not None]
    return round(float(st["ref_z"]) + min(rel) - float(st["floor_margin_mm"]), 4)


def problems(st: dict, guard_ref: str | None) -> list[str]:
    out = []
    if st.get("ref") is None:
        out.append("Home Z on the load cell first.")
    elif guard_ref is not None and st["ref"] != guard_ref:
        out.append("The Z reference changed after the last Z home. Home Z again.")
    for k, label in (("slide_mm", "slide"), ("pickup_mm", "pickup"), ("bin_mm", "bin"), ("safe_mm", "safe Z")):
        if st.get(k) is None:
            out.append(f"Set the {label} height.")
    if st.get("safe_mm") is not None:
        top = max(float(st[k]) for k in ("slide_mm", "pickup_mm", "bin_mm") if st.get(k) is not None) \
            if any(st.get(k) is not None for k in ("slide_mm", "pickup_mm", "bin_mm")) else None
        if top is not None and float(st["safe_mm"]) < top:
            out.append("Safe Z must be at or above the slide, pickup and bin heights.")
    return out


def patch_for(st: dict, auto_raw: dict, scrub_raw: dict, ref: str) -> dict:
    """Calibration changes that turn the relative heights into taught Zs under
    reference `ref`. XY of every taught position is left alone."""
    auto = dict(auto_raw or {})
    refs = dict(auto.get("refs") or {})
    upd_auto: dict[str, Any] = {}
    for key, name in (("pickup", "pickup_mm"), ("bin", "bin_mm")):
        z = absolute(st, name)
        if z is not None and auto.get(key):
            upd_auto[key] = {**auto[key], "z": z}
            refs[key] = ref
    sz = absolute(st, "safe_mm")
    if sz is not None:
        upd_auto["safe_z"] = sz
        refs["safe_z"] = ref
    upd_auto["refs"] = refs
    out: dict[str, Any] = {"autonomous": upd_auto}
    z = absolute(st, "slide_mm")
    start = (scrub_raw or {}).get("start")
    if z is not None and start:
        out["scrub"] = {"start": {**start, "z": z, "ref": ref}}
    return out


def probe(printer: Any, loadcell: Any, lc: dict, zc: dict, sp: dict, *, start_z: float,
          speed: float | None = None, step: float | None = None):
    return calibrate_z(
        printer, loadcell,
        cell_x=lc["x"], cell_y=lc["y"], z_safe=start_z,
        target_force_n=zc["target_force_n"], abort_force_n=zc["abort_force_n"],
        step_mm=step or zc["step_mm"], max_travel_mm=zc["max_travel_mm"],
        approach_speed_mm_s=speed or sp["z_approach"], travel_speed_mm_s=sp["travel"],
        settle_readings=int(zc.get("settle_readings", 3)), backoff_mm=zc.get("backoff_mm", 0.5),
        z_speed_mm_s=min(float(sp.get("z_lift", 10.0)), 10.0),
    )


def declare(printer: Any, ref_z: float, probe_z: float, contact_z: float) -> float:
    """Tell Klipper the head is at the Z it should read if the contact were
    exactly ref_z. Returns the new Z of the head."""
    p = printer.position()
    new_z = ref_z + (p.z - contact_z)
    printer.script(f"SET_KINEMATIC_POSITION X={p.x:.4f} Y={p.y:.4f} Z={new_z:.4f}", wait=False)
    return new_z


def check(printer: Any, loadcell: Any, lc: dict, zc: dict, sp: dict, st: dict, safe_z: float,
          stop: Callable[[], None] = lambda: None) -> dict:
    """Probe the cell with the swab now in the jaws. Z is not changed."""
    ref_z = float(st["ref_z"])
    start_z = ref_z + float(st["probe_start_mm"])
    if printer.position().z < safe_z - 1e-6:
        printer.move(z=safe_z, speed_mm_s=min(float(sp.get("z_lift", 10.0)), 10.0))
    stop()
    r = probe(printer, loadcell, lc, zc, sp, start_z=start_z)
    out = {"contact_z": round(r.contact_z, 4), "drift_mm": round(r.contact_z - ref_z, 4),
           "force_n": round(r.force_n, 3), "steps": r.steps, "at": time.time()}
    printer.move(z=safe_z, speed_mm_s=min(float(sp.get("z_lift", 10.0)), 10.0))
    return out


def park(printer: Any, safe_z: float | None, speed: float = 10.0) -> bool:
    """Rise to the safe Z if below it. Never moves down."""
    if safe_z is None:
        return False
    p = printer.position()
    if "z" not in p.homed:
        return False
    if p.z < safe_z - 1e-6:
        printer.move(z=safe_z, speed_mm_s=speed)
    return True
