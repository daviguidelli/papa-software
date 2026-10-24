"""The orchestrator: one loop, owned by the Pi.

The whiteboard's two columns collapse into this single timeline. A purple step
is "send a move and wait for it to finish" (every script ends with M400); a
black step is a local Pi action. The mainboard never runs any of our code.

States: preflight -> pick swab -> grip -> z cal -> scribe block -> worn?
-> (swap: dispose, pick, grip, z cal) -> slide done? -> dispose -> next slide.
Pause, soft stop, hard stop and error can happen from any state, and state is
saved after every block so any of them resumes from the right place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from papa.config import Calibration, Machine, Swab
from papa.ops.jobs import Job
from papa.ops.swabs import Rack, RackEmpty, Wear
from papa.ops.zcal import GripFailed, OverForce, calibrate_z
from papa.toolpath.blocks import Block, resume_points


class Stopped(Exception):
    """Raised inside the loop when a stop was requested."""


@dataclass
class Status:
    state: str = "idle"
    stage: str = ""
    slot_id: int | None = None
    swab: str | None = None
    swab_index: int = 0
    wear_fraction: float = 0.0
    scribe_fraction: float = 0.0
    overall_fraction: float = 0.0
    message: str = ""


@dataclass
class Orchestrator:
    machine: Machine
    cal: Calibration
    printer: Any
    loadcell: Any
    gripper: Any
    rack: Rack
    blocks: list[Block]
    job: Job
    on_status: Callable[[Status], None] | None = None

    _pause: bool = False
    _soft_stop: bool = False
    _hard_stop: bool = False
    _wear: Wear = field(default_factory=Wear)
    _scribe_z: float = 0.0
    status: Status = field(default_factory=Status)

    # --- external controls ---

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False

    def soft_stop(self) -> None:
        """Finish the current block, lift, then stop. Protects the slide."""
        self._soft_stop = True

    def hard_stop(self) -> None:
        """Emergency stop. Klipper shuts down; a restart and re-home follow."""
        self._hard_stop = True
        self.printer.emergency_stop()

    # --- helpers ---

    def _emit(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self.status, k, v)
        self.status.overall_fraction = self.job.fraction
        if self.on_status:
            self.on_status(self.status)

    def _checkpoint(self) -> None:
        if self._hard_stop:
            raise Stopped("emergency stop")
        while self._pause and not self._hard_stop and not self._soft_stop:
            self.job.state = "paused"
            self.job.save()
            self._emit(state="paused")
            time.sleep(0.2)
        if self._hard_stop:
            raise Stopped("emergency stop")
        if self.job.state == "paused":
            self.job.state = "running"
            self._emit(state="running")

    def _to_machine(self, slot_x: float, slot_y: float, lx: float, ly: float) -> tuple[float, float]:
        """Slide-local to machine, through the configured axis mapping.

        A rotated mounting base is a config change, so this is the only place
        that knows which way local +y runs.
        """
        axes = self.machine.axes
        dx = lx if axes["local_x"] == "+X" else -lx
        dy = ly if axes["local_y"] == "+Y" else -ly
        if axes["local_x"] in ("+Y", "-Y") or axes["local_y"] in ("+X", "-X"):
            # swapped axes: local x runs along machine Y
            dx, dy = dy, dx
        return slot_x + dx, slot_y + dy

    # --- the steps ---

    def preflight(self) -> list[str]:
        """Everything that must be true before the first move. Names what is
        missing rather than failing with a generic error mid-job."""
        problems: list[str] = []
        if not self.printer.ping():
            problems.append("Moonraker is not reachable")
        else:
            try:
                if not self.printer.position().is_homed:
                    problems.append("the gantry is not homed")
            except Exception as exc:
                problems.append(f"could not read position: {exc}")
        need = len(self.job.slot_ids) * self.job.swabs_per_slide
        if self.rack.available() < need:
            problems.append(
                f"{need} swabs needed, {self.rack.available()} loaded in the rack"
            )
        plan = self.rack.reserve(self.job.slot_ids, self.job.swabs_per_slide)
        swab_ids = [
            s.id for lab in sum(plan.values(), []) for s in self.cal.swabs if s.label == lab
        ]
        problems.extend(f"{m} is not calibrated" for m in self.cal.uncalibrated(self.job.slot_ids, swab_ids))
        if not self.cal.pattern_fits():
            problems.append("the pattern square is larger than the substrate")
        return problems

    def pick_and_prepare(self) -> Swab:
        """Pick swab, grip, calibrate Z. Runs at the start of every slide and
        again on every mid-slide swap, because each swab sits differently."""
        failures = 0
        max_fail = int(self.machine.raw.get("retry", {}).get("max_grip_failures", 3))
        speeds = self.machine.speeds
        while True:
            self._checkpoint()
            label = self.rack.next_label()
            swab = self.rack.take(label)
            self._emit(stage="pick swab", swab=label)

            self.printer.move(z=self.cal.load_cell["z_safe"], speed_mm_s=speeds["z_lift"])
            self.printer.move(x=swab.x, y=swab.y, speed_mm_s=speeds["travel"])
            self.gripper.open()
            self.printer.move(z=swab.z_grip, speed_mm_s=speeds["z_approach"])
            self._emit(stage="grip")
            self.gripper.close()
            self.printer.move(z=self.cal.load_cell["z_safe"], speed_mm_s=speeds["z_lift"])

            self._emit(stage="z calibration")
            z = self.machine.zcal
            lc = self.cal.load_cell
            try:
                result = calibrate_z(
                    self.printer, self.loadcell,
                    cell_x=lc["x"], cell_y=lc["y"], z_safe=lc["z_safe"],
                    target_force_g=z["target_force_g"], abort_force_g=z["abort_force_g"],
                    step_mm=z["step_mm"], max_travel_mm=z["max_travel_mm"],
                    approach_speed_mm_s=speeds["z_approach"],
                    travel_speed_mm_s=speeds["travel"],
                    settle_readings=int(z.get("settle_readings", 3)),
                    backoff_mm=z.get("backoff_mm", 0.5),
                )
            except GripFailed as exc:
                failures += 1
                self._emit(message=f"{label}: {exc}")
                if failures >= max_fail:
                    raise
                continue
            except OverForce:
                raise

            self._scribe_z = result.scribe_z(self.cal.surface_offset_mm)
            self._wear = Wear()
            self.job.current_swab = label
            self.job.slide(self.job.current_slot).swabs_used.append(label)
            self.job.save()
            return swab

    def dispose(self) -> None:
        d = self.cal.disposal
        speeds = self.machine.speeds
        self._emit(stage="dispose")
        self.printer.move(z=self.cal.load_cell["z_safe"], speed_mm_s=speeds["z_lift"])
        self.printer.move(x=d["x"], y=d["y"], speed_mm_s=speeds["travel"])
        self.printer.move(z=d["z_release"], speed_mm_s=speeds["z_approach"])
        self.gripper.open()
        self.printer.move(z=self.cal.load_cell["z_safe"], speed_mm_s=speeds["z_lift"])

    def scribe_block(self, block: Block, slot_x: float, slot_y: float, resume: bool) -> None:
        """One block: lift, travel to the start, lower, scribe, lift."""
        speeds = self.machine.speeds
        overlap = self.machine.toolpath.get("resume_overlap_mm", 0.0)
        pts = resume_points(block, overlap) if resume else list(block.points)
        mx, my = self._to_machine(slot_x, slot_y, pts[0][0], pts[0][1])

        if block.starts_stroke or resume:
            self.printer.move(z=self._scribe_z + 2.0, speed_mm_s=speeds["z_lift"])
            self.printer.move(x=mx, y=my, speed_mm_s=speeds["travel"])
            self.printer.move(z=self._scribe_z, speed_mm_s=speeds["z_approach"])

        lines = ["G90"]
        for lx, ly in pts[1:]:
            px, py = self._to_machine(slot_x, slot_y, lx, ly)
            lines.append(f"G1 X{px:.4f} Y{py:.4f} F{speeds['scribe'] * 60:.0f}")
        t0 = time.time()
        self.printer.script("\n".join(lines))
        self._wear.add(path_mm=block.length, contact_s=time.time() - t0)

        if block.ends_stroke:
            self.printer.move(z=self._scribe_z + 2.0, speed_mm_s=speeds["z_lift"])

    def run(self) -> Job:
        """The loop. Every block boundary is a checkpoint and a save."""
        problems = self.preflight()
        if problems:
            self.job.state = "error"
            self.job.message = "; ".join(problems)
            self.job.save()
            self._emit(state="error", message=self.job.message)
            return self.job

        wear_cfg = self.machine.wear
        self.job.state = "running"
        self.job.save()
        self._emit(state="running")

        try:
            while True:
                slide = self.job.next_slide()
                if slide is None:
                    break
                self.job.current_slot = slide.slot_id
                slot = self.cal.slot(slide.slot_id)
                self._emit(slot_id=slide.slot_id, swab_index=len(slide.swabs_used))

                # A fresh swab starts every slide, even if the last one barely wore.
                self.pick_and_prepare()
                resume = slide.block_index > 0

                while slide.block_index < len(self.blocks):
                    self._checkpoint()
                    if self._soft_stop:
                        raise Stopped("soft stop")

                    block = self.blocks[slide.block_index]
                    self._emit(
                        stage="scribe",
                        scribe_fraction=slide.block_index / max(1, len(self.blocks)),
                        wear_fraction=self._wear.fraction(
                            wear_cfg["max_path_mm"], wear_cfg["max_contact_s"]
                        ),
                    )
                    self.scribe_block(block, slot.x, slot.y, resume)
                    resume = False
                    slide.block_index += 1
                    self.job.save()      # after every block, not every slide

                    if (
                        slide.block_index < len(self.blocks)
                        and self._wear.worn(wear_cfg["max_path_mm"], wear_cfg["max_contact_s"])
                    ):
                        self._emit(message="swab hit its wear limit, swapping")
                        self.dispose()
                        self.pick_and_prepare()
                        resume = True    # back up slightly so no gap is left
                        self._emit(swab_index=len(slide.swabs_used))

                self.dispose()
                slide.done = True
                self.job.save()

            self.job.state = "done"
            self.job.current_slot = None
            self.job.message = "all slides scribed"
            self.job.save()
            self._emit(state="done", stage="", message=self.job.message)

        except Stopped as exc:
            self.job.state = "paused" if "soft" in str(exc) else "error"
            self.job.message = str(exc)
            self.job.save()
            self._emit(state=self.job.state, message=self.job.message)
        except (RackEmpty, OverForce, GripFailed) as exc:
            self.job.state = "paused"
            self.job.message = str(exc)
            self.job.save()
            self._emit(state="paused", message=self.job.message)

        return self.job
