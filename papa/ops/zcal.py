"""Z calibration against the load cell.

This runs after every swab pickup, because each swab sits slightly differently
in the gripper. Stepping bounds the overshoot to one step, which matters: the
HX711 gives only 10 to 80 samples per second, so watching force during a
continuous descent would find contact too late.

If force never rises within max_travel, the swab is missing or the grip
failed. That gives grip detection with no extra sensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GripFailed(RuntimeError):
    """No contact within max travel: no swab, or the gripper did not hold it."""


class OverForce(RuntimeError):
    """Abort force exceeded. Protects the cell, which is easily overloaded."""


@dataclass(frozen=True)
class ZCalResult:
    contact_z: float        # Z where the force crosses the target, interpolated between two steps
    force_n: float          # the reading that was accepted
    steps: int
    stepped_z: float = 0.0  # the Z the head actually stopped at, before the back-off

    def scribe_z(self, surface_offset_mm: float) -> float:
        """Scribe height on glass, derived from the probed contact height.

        The offset is what links the two Z references; without it a load cell
        reading cannot be transferred to the slide.
        """
        return self.contact_z + surface_offset_mm


def calibrate_z(
    printer: Any,
    loadcell: Any,
    *,
    cell_x: float,
    cell_y: float,
    z_safe: float,
    target_force_n: float,
    abort_force_n: float,
    step_mm: float,
    max_travel_mm: float,
    approach_speed_mm_s: float,
    travel_speed_mm_s: float = 100.0,
    settle_readings: int = 3,
    backoff_mm: float = 0.5,
    z_speed_mm_s: float = 10.0,
) -> ZCalResult:
    if abort_force_n <= target_force_n:
        raise ValueError("abort force must exceed target force")

    # Go to the start Z first: rise if below it, XY at that height, then settle
    # onto it. Z never moves at travel speed, so a start Z below the current
    # height is approached at z_speed_mm_s, not 150 mm/s.
    if printer.position().z < z_safe:
        printer.move(z=z_safe, speed_mm_s=z_speed_mm_s)
    printer.move(x=cell_x, y=cell_y, speed_mm_s=travel_speed_mm_s)
    if printer.position().z > z_safe + 1e-6:
        printer.move(z=z_safe, speed_mm_s=z_speed_mm_s)
    getattr(loadcell, "settle", lambda: True)()   # skip the start-up spikes before zeroing
    loadcell.tare()

    travelled = 0.0
    hits = 0
    steps = 0
    prev_z = printer.position().z
    prev_f = 0.0
    cross: tuple[float, float, float, float] | None = None
    while travelled < max_travel_mm:
        printer.move_relative(dz=-step_mm, speed_mm_s=approach_speed_mm_s)
        travelled += step_mm
        steps += 1
        z_now = prev_z - step_mm
        force = loadcell.newtons()

        if force >= abort_force_n:
            printer.move(z=z_safe, speed_mm_s=z_speed_mm_s)
            raise OverForce(
                f"load cell read {force:.3f} N, at or above the abort force "
                f"{abort_force_n:.3f} N"
            )

        if force <= -target_force_n:
            printer.move(z=z_safe, speed_mm_s=z_speed_mm_s)
            raise GripFailed(
                f"the load cell reads {force:.3f} N while pressed: the sign is backwards. "
                "Re-run Tare and Calibrate scale with the known mass pushing DOWN on the cell "
                "(the same way the swab will), then save. The scale comes out negative if the "
                "cell reads negative under load, which flips the reading."
            )

        if force >= target_force_n:
            if hits == 0:
                cross = (prev_z, prev_f, z_now, force)
            hits += 1
            if hits >= settle_readings:
                stepped = printer.position().z
                z0, f0, z1, f1 = cross
                contact_z = z1 if f1 <= f0 else z0 + (target_force_n - f0) * (z1 - z0) / (f1 - f0)
                if backoff_mm:
                    printer.move_relative(dz=backoff_mm, speed_mm_s=approach_speed_mm_s)
                return ZCalResult(contact_z=contact_z, force_n=force, steps=steps, stepped_z=stepped)
        else:
            hits = 0
        prev_z, prev_f = z_now, force

    printer.move(z=z_safe, speed_mm_s=z_speed_mm_s)
    raise GripFailed(
        f"no contact within {max_travel_mm:.2f} mm: the swab is missing or the "
        "grip failed"
    )
