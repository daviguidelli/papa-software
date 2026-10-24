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
    contact_z: float
    force_g: float
    steps: int

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
    target_force_g: float,
    abort_force_g: float,
    step_mm: float,
    max_travel_mm: float,
    approach_speed_mm_s: float,
    travel_speed_mm_s: float = 100.0,
    settle_readings: int = 3,
    backoff_mm: float = 0.5,
) -> ZCalResult:
    if abort_force_g <= target_force_g:
        raise ValueError("abort force must exceed target force")

    printer.move(z=z_safe, speed_mm_s=travel_speed_mm_s)
    printer.move(x=cell_x, y=cell_y, speed_mm_s=travel_speed_mm_s)
    loadcell.tare()

    travelled = 0.0
    hits = 0
    steps = 0
    while travelled < max_travel_mm:
        printer.move_relative(dz=-step_mm, speed_mm_s=approach_speed_mm_s)
        travelled += step_mm
        steps += 1
        force = loadcell.grams()

        if force >= abort_force_g:
            printer.move(z=z_safe, speed_mm_s=travel_speed_mm_s)
            raise OverForce(
                f"load cell read {force:.2f} g, at or above the abort force "
                f"{abort_force_g:.2f} g"
            )

        if force >= target_force_g:
            hits += 1
            if hits >= settle_readings:
                contact_z = printer.position().z
                if backoff_mm:
                    printer.move_relative(dz=backoff_mm, speed_mm_s=approach_speed_mm_s)
                return ZCalResult(contact_z=contact_z, force_g=force, steps=steps)
        else:
            hits = 0

    printer.move(z=z_safe, speed_mm_s=travel_speed_mm_s)
    raise GripFailed(
        f"no contact within {max_travel_mm:.2f} mm: the swab is missing or the "
        "grip failed"
    )
