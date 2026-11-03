"""Hold-to-jog: the toolhead keeps moving while the button is held.

The browser sends a heartbeat every ~100 ms while a button is down. If the
heartbeats stop (button released, tab closed, Wi-Fi drop) motion stops within
DEADMAN_S. Moves are queued to Klipper in short pieces with a small look-ahead,
so at most LOOKAHEAD_S of motion is still queued when the heartbeat dies. Every
piece is checked against the axis limits and the Z floor first.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

DEADMAN_S = 0.35
LOOKAHEAD_S = 0.25
TICK_S = 0.05
MAX_SPEED = {"xy": 100.0, "z": 10.0}
MIN_SPEED = 0.02


class HoldJog:
    def __init__(self, printer_fn: Callable[[], Any], machine: Any, guard_fn: Callable[[], Any],
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._printer_fn = printer_fn
        self._machine = machine
        self._guard_fn = guard_fn
        self._clock = clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._dir = (0, 0, 0)
        self._speed = 0.0
        self._beat = 0.0
        self.stopped_because = ""

    @property
    def active(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def beat(self, dx: int, dy: int, dz: int, speed_mm_s: float) -> None:
        d = (_sign(dx), _sign(dy), _sign(dz))
        if d == (0, 0, 0):
            raise ValueError("no direction given")
        if sum(1 for a in d if a) > 1:
            raise ValueError("hold-jog moves one axis at a time")
        cap = MAX_SPEED["z"] if d[2] else MAX_SPEED["xy"]
        speed = float(speed_mm_s)
        if not MIN_SPEED <= speed <= cap:
            raise ValueError(f"speed must be {MIN_SPEED} to {cap} mm/s for this axis")
        with self._lock:
            if not self.active and self.stopped_because not in ("", "released"):
                raise ValueError(self.stopped_because)
            self._dir, self._speed, self._beat = d, speed, self._clock()
            if self.active:
                return
            self.stopped_because = ""
            self._thread = threading.Thread(target=self._run, daemon=True, name="hold-jog")
            self._thread.start()

    def reset(self) -> None:
        """Forget why the last hold stopped; called when the button is released."""
        self.stopped_because = ""

    def stop(self, reason: str = "released") -> None:
        with self._lock:
            self._dir = (0, 0, 0)
            self.stopped_because = self.stopped_because or reason
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    def _run(self) -> None:
        pr = self._printer_fn()
        try:
            p = pr.position()
            pos = [p.x, p.y, p.z]
            sent_mm = 0.0
            t0 = self._clock()
            while True:
                with self._lock:
                    d, speed, beat = self._dir, self._speed, self._beat
                if d == (0, 0, 0):
                    self.stopped_because = self.stopped_because or "released"
                    return
                if self._clock() - beat > DEADMAN_S:
                    self.stopped_because = "no heartbeat from the page"
                    return
                elapsed = self._clock() - t0
                if sent_mm - speed * elapsed < speed * LOOKAHEAD_S:
                    step = speed * TICK_S * 2
                    axis = next(i for i, a in enumerate(d) if a)
                    name = "xyz"[axis]
                    target = pos[axis] + d[axis] * step
                    lo, hi = self._machine.limits[name]
                    if not lo <= target <= hi:
                        self.stopped_because = f"{name.upper()} limit reached"
                        return
                    if axis == 2 and d[2] < 0:
                        fl = self._guard_fn().floor
                        if fl is None:
                            self.stopped_because = "Z floor is not set"
                            return
                        if target < fl:
                            self.stopped_because = f"Z floor {fl:.3f} reached"
                            return
                    pos[axis] = target
                    sent_mm += step
                    pr.script(f"G90\nG1 {name.upper()}{target:.4f} F{speed * 60:.0f}", wait=False)
                time.sleep(TICK_S)
        except Exception as exc:                                   # noqa: BLE001
            self.stopped_because = f"error: {exc}"
        finally:
            with self._lock:
                self._dir = (0, 0, 0)


def _sign(v: Any) -> int:
    v = float(v or 0)
    return 1 if v > 0 else -1 if v < 0 else 0
