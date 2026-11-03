"""goBILDA Proton servo gripper on hardware PWM (GPIO13, physical pin 33).

Hardware PWM rather than software: a software-timed pulse jitters, and a
jittering gripper drops swabs. The servo has its own supply; the Pi provides the
signal, and the grounds must be common.

Calibration safety: manual pulses ramp at a chosen slew rate (us per second)
instead of jumping, and open()/close() refuse to run until both positions have
been locked in the validation GUI. The very first pulse after power-up cannot
be ramped, because the horn's resting position is unknown; start with the jaws
empty and clear of everything.
"""
from __future__ import annotations

import time
from typing import Protocol

PERIOD_US = 20_000  # 50 Hz
MIN_US, MAX_US = 600, 2400  # goBILDA Proton servo spec
RAMP_TICK_S = 0.02
MAX_SLEW_US_S = 2000.0


class GripperNotCalibrated(RuntimeError):
    pass


class Gripper(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...


def ramp_points(start: int, end: int, slew_us_s: float) -> list[tuple[int, float]]:
    """Pulse widths to send on the way from start to end, each with the pause
    after it. Whole microseconds only; the last point is always end."""
    slew = max(1.0, min(float(slew_us_s), MAX_SLEW_US_S))
    n = abs(end - start)
    if n == 0:
        return []
    per_tick = max(1.0, slew * RAMP_TICK_S)
    pts: list[tuple[int, float]] = []
    pos = float(start)
    sign = 1 if end > start else -1
    while abs(end - pos) > per_tick:
        pos += sign * per_tick
        pts.append((int(round(pos)), per_tick / slew))
    pts.append((end, 0.0))
    return pts


class ServoGripper:
    def __init__(self, pwm_pin: int, open_us: int, closed_us: int,
                 settle_s: float = 0.45, calibrated: bool = False) -> None:
        import lgpio

        self._lg = lgpio
        self._h = lgpio.gpiochip_open(0)
        self.pin = pwm_pin
        self.open_us = int(open_us)
        self.closed_us = int(closed_us)
        self.settle_s = settle_s
        self.calibrated = bool(calibrated)
        lgpio.gpio_claim_output(self._h, pwm_pin, 0)
        self._state = "unknown"
        self.pulse_us: int | None = None

    def _write(self, width_us: int) -> None:
        duty = 100.0 * width_us / PERIOD_US
        self._lg.tx_pwm(self._h, self.pin, 50, duty)
        self.pulse_us = width_us

    def _pulse(self, width_us: int, slew_us_s: float | None = None,
               settle_s: float | None = None) -> None:
        width_us = max(MIN_US, min(MAX_US, int(width_us)))
        if slew_us_s and self.pulse_us is not None:
            for w, pause in ramp_points(self.pulse_us, width_us, slew_us_s):
                self._write(w)
                if pause:
                    time.sleep(pause)
        else:
            self._write(width_us)
        time.sleep(self.settle_s if settle_s is None else settle_s)

    def _require_calibrated(self) -> None:
        if not self.calibrated:
            raise GripperNotCalibrated(
                "Lock both OPEN and CLOSED on the Servo tab before using the gripper"
            )

    def open(self, slew_us_s: float | None = None) -> None:
        self._require_calibrated()
        self._pulse(self.open_us, slew_us_s)
        self._state = "open"

    def close(self, slew_us_s: float | None = None) -> None:
        self._require_calibrated()
        self._pulse(self.closed_us, slew_us_s)
        self._state = "closed"

    def pulse(self, width_us: int, slew_us_s: float | None = None) -> None:
        """Drive to an arbitrary pulse width. Used only to find open and closed."""
        self._pulse(width_us, slew_us_s, settle_s=0.05)
        self._state = "manual"

    @property
    def state(self) -> str:
        return self._state

    def release(self) -> None:
        """Stop driving the servo so it is not holding torque between moves."""
        self._lg.tx_pwm(self._h, self.pin, 50, 0)

    def shutdown(self) -> None:
        self.release()
        self._lg.gpiochip_close(self._h)
