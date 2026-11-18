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



AIR_SPEED_MM_S = 5.0  # through air nothing can be hit hard
AIR_SAMPLES = 1       # one reading per air step (0.1 s); the glitch check still re-reads spikes
FINE_SAMPLES = 3
SLOW_FROM_N = 0.3     # first touch: switch to fine steps
FINE_STEP_MM = 0.1
MAX_JUMP_N = 3.0      # a real 0.25 mm step stays under this; glitches are hundreds of N
GLITCH_RETRIES = 10


def _steady_force(loadcell, prev_f: float, samples: int = 5) -> float:
    """A reading the probe can trust. The HX711 sometimes spits out wild values
    (hundreds of N). Real force changes a little per step, so a reading far from
    the last one is read again; only a value that stays put is believed."""
    import time
    last = None
    for _ in range(GLITCH_RETRIES):
        try:
            f = loadcell.newtons(samples)
        except TimeoutError:
            # The chip dropped out for a moment. The head is stopped, so wait
            # for it to come back (wake gives it up to 12 s) and read again.
            wake = getattr(loadcell, "wake", None)
            if wake is None:
                raise
            wake()
            getattr(loadcell, "settle", lambda: True)()
            f = loadcell.newtons()
        if abs(f - prev_f) <= MAX_JUMP_N:
            return f
        if last is not None and abs(f - last) <= 0.2:      # two agreeing reads: it is real
            return f
        last = f
        time.sleep(0.05)
    return f

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
    loadcell.tare(10)

    travelled = 0.0
    hits = 0
    steps = 0
    prev_z = printer.position().z
    prev_f = 0.0
    cross: tuple[float, float, float, float] | None = None
    while travelled < max_travel_mm:
        # Big steps through air, small ones once the swab starts to touch, so
        # the fast approach cannot overshoot into the abort force.
        air = prev_f < SLOW_FROM_N
        step = step_mm if air else min(step_mm, FINE_STEP_MM)
        printer.move_relative(dz=-step, speed_mm_s=max(approach_speed_mm_s, AIR_SPEED_MM_S) if air else approach_speed_mm_s)
        travelled += step
        steps += 1
        z_now = prev_z - step
        try:
            force = _steady_force(loadcell, prev_f, AIR_SAMPLES if air else FINE_SAMPLES)
        except Exception:
            printer.move(z=z_safe, speed_mm_s=z_speed_mm_s)       # never leave the head pressing
            raise

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
            # Confirm by reading again at this same Z, not by stepping further
            # down: with 0.25 mm steps two more steps would press far too hard.
            cross = (prev_z, prev_f, z_now, force)
            hits = 1
            while hits < settle_readings:
                again = _steady_force(loadcell, force, FINE_SAMPLES)
                if again < target_force_n:
                    break
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
