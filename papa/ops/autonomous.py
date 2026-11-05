"""One-button moves: grip a swab, dispose of it.

Not the integrated run. Each button does one thing from wherever the gantry
happens to be, with the same safety rule every time: Z goes up to the saved
safe height before any XY move, and only then does XY travel, and only then
does Z come down onto the target.

The positions are taught by jogging there and pressing Save here, and live in
calibration.yaml under `autonomous`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULTS: dict[str, Any] = {
    "pickup": None,          # {"x": , "y": , "z": } swab position, jaws open and ready
    "bin": None,             # {"x": , "y": , "z": } centre of the bin, where the jaws open
    "safe_z": None,          # XY only ever moves at or above this Z
    "lift_mm": 30.0,         # after the grip, rise this far above the pickup Z
    "grip_wait_s": 1.0,      # time allowed for the jaws to close before lifting
    "release_wait_s": 0.6,   # time allowed for the jaws to open over the bin
    "xy_speed_mm_s": 60.0,
    "z_speed_mm_s": 8.0,
}

TARGETS = ("pickup", "bin", "safe_z")


class AutoError(RuntimeError):
    pass


class Aborted(AutoError):
    pass


def merged(raw: dict | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    return out


def centre(machine: Any) -> dict[str, float]:
    """The middle of the XY gantry, from the machine limits."""
    lo_x, hi_x = machine.limits["x"]
    lo_y, hi_y = machine.limits["y"]
    return {"x": round((lo_x + hi_x) / 2, 3), "y": round((lo_y + hi_y) / 2, 3)}


def describe(settings: dict[str, Any], machine: Any) -> dict[str, Any]:
    return {
        "settings": settings,
        "centre": centre(machine),
        "ready": {
            "grip": not problems(settings, machine, "grip"),
            "dispose": not problems(settings, machine, "dispose"),
        },
        "problems": {
            "grip": problems(settings, machine, "grip"),
            "dispose": problems(settings, machine, "dispose"),
        },
    }


def problems(settings: dict[str, Any], machine: Any, action: str) -> list[str]:
    out: list[str] = []
    safe = settings.get("safe_z")
    if safe is None:
        out.append("Save the safe Z first.")
    target_name = "pickup" if action == "grip" else "bin"
    t = settings.get(target_name)
    if t is None:
        out.append(f"Save the {target_name} position first.")
    if t is not None:
        if not machine.in_limits(float(t["x"]), float(t["y"]), float(t["z"])):
            out.append(f"The {target_name} position is outside the machine limits.")
        if safe is not None and float(safe) < float(t["z"]):
            out.append(f"Safe Z must be at or above the {target_name} Z.")
    if safe is not None and not machine.limits["z"][0] <= float(safe) <= machine.limits["z"][1]:
        out.append("Safe Z is outside the machine limits.")
    if action == "grip" and t is not None:
        top = float(t["z"]) + float(settings["lift_mm"])
        if top > machine.limits["z"][1]:
            out.append("The carry lift would go above the Z limit; lower the lift.")
    return out


@dataclass
class Step:
    name: str
    state: str = "pending"     # pending | running | done | failed | skipped


@dataclass
class AutoStatus:
    running: bool = False
    action: str = ""
    steps: list[Step] = field(default_factory=list)
    error: str = ""
    finished: bool = False

    def as_dict(self) -> dict:
        return {
            "running": self.running, "action": self.action, "error": self.error,
            "finished": self.finished,
            "steps": [{"name": s.name, "state": s.state} for s in self.steps],
        }


class AutoRunner:
    """Runs one action on a thread and reports its steps.

    The thread is what the hub treats as busy, so while it runs every other
    motion endpoint refuses. Stop takes effect between steps; E-STOP is still
    the only thing that halts a move already under way.
    """

    def __init__(self, printer: Callable[[], Any], gripper: Callable[[], Any], machine: Any) -> None:
        self._printer = printer
        self._gripper = gripper
        self.machine = machine
        self.status = AutoStatus()
        self._abort = threading.Event()
        self.thread: threading.Thread | None = None

    # ---- helpers used by the sequences ----
    def _check(self) -> None:
        if self._abort.is_set():
            raise Aborted("stopped between steps")

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            self._check()
            time.sleep(min(0.05, end - time.monotonic()))

    def _z_to(self, z: float, st: dict) -> None:
        self._printer().move(z=z, speed_mm_s=float(st["z_speed_mm_s"]))

    def _xy_to(self, x: float, y: float, st: dict) -> None:
        self._printer().move(x=x, y=y, speed_mm_s=float(st["xy_speed_mm_s"]))

    def _lift_to_safe(self, st: dict) -> None:
        if self._printer().position().z < float(st["safe_z"]) - 1e-6:
            self._z_to(float(st["safe_z"]), st)

    def _homed_xy(self) -> None:
        p = self._printer().position()
        if not all(a in p.homed for a in "xy"):
            raise AutoError("Home X and Y first.")

    # ---- the two actions ----
    def grip_steps(self, st: dict) -> list[tuple[str, Callable[[], None]]]:
        t = st["pickup"]
        c = centre(self.machine)
        return [
            ("Check X and Y are homed", self._homed_xy),
            ("Open the jaws", lambda: self._gripper().open()),
            ("Rise to safe Z", lambda: self._lift_to_safe(st)),
            ("Move to the pickup in XY", lambda: self._xy_to(float(t["x"]), float(t["y"]), st)),
            ("Lower onto the swab", lambda: self._z_to(float(t["z"]), st)),
            ("Close the jaws", lambda: self._gripper().close()),
            ("Wait for the grip", lambda: self._sleep(float(st["grip_wait_s"]))),
            ("Lift the swab", lambda: self._z_to(float(t["z"]) + float(st["lift_mm"]), st)),
            ("Move to the centre of the gantry", lambda: self._xy_to(c["x"], c["y"], st)),
        ]

    def dispose_steps(self, st: dict) -> list[tuple[str, Callable[[], None]]]:
        t = st["bin"]
        return [
            ("Check X and Y are homed", self._homed_xy),
            ("Rise to safe Z", lambda: self._lift_to_safe(st)),
            ("Move over the bin in XY", lambda: self._xy_to(float(t["x"]), float(t["y"]), st)),
            ("Lower into the bin", lambda: self._z_to(float(t["z"]), st)),
            ("Open the jaws", lambda: self._gripper().open()),
            ("Wait for the release", lambda: self._sleep(float(st["release_wait_s"]))),
            ("Rise to safe Z", lambda: self._z_to(float(st["safe_z"]), st)),
        ]

    # ---- run control ----
    def start(self, action: str, settings: dict[str, Any]) -> threading.Thread:
        if action not in ("grip", "dispose"):
            raise AutoError(f"unknown action {action}")
        bad = problems(settings, self.machine, action)
        if bad:
            raise AutoError(" ".join(bad))
        steps = (self.grip_steps if action == "grip" else self.dispose_steps)(settings)
        self._abort.clear()
        self.status = AutoStatus(running=True, action=action, steps=[Step(n) for n, _ in steps])
        self.thread = threading.Thread(target=self._run, args=(steps,), daemon=True)
        self.thread.start()
        return self.thread

    def stop(self) -> None:
        self._abort.set()

    def _run(self, steps: list[tuple[str, Callable[[], None]]]) -> None:
        s = self.status
        try:
            for i, (_, fn) in enumerate(steps):
                self._check()
                s.steps[i].state = "running"
                try:
                    fn()
                except Exception:
                    s.steps[i].state = "failed"
                    raise
                s.steps[i].state = "done"
            s.finished = True
        except Aborted as exc:
            s.error = str(exc)
            for st in s.steps:
                if st.state == "pending":
                    st.state = "skipped"
        except Exception as exc:                          # noqa: BLE001
            s.error = f"{type(exc).__name__}: {exc}" if not str(exc) else str(exc)
            for st in s.steps:
                if st.state == "pending":
                    st.state = "skipped"
        finally:
            s.running = False
