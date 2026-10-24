"""goBILDA Proton servo gripper on hardware PWM (GPIO13).

Hardware PWM rather than software: a software-timed pulse jitters, and a
jittering gripper drops swabs. The servo runs from its own 4xAA pack, so the
Pi only provides the signal.
"""
from __future__ import annotations

import time
from typing import Protocol

PERIOD_US = 20_000  # 50 Hz


class Gripper(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...


class ServoGripper:
    def __init__(self, pwm_pin: int, open_us: int, closed_us: int,
                 settle_s: float = 0.45) -> None:
        import lgpio

        self._lg = lgpio
        self._h = lgpio.gpiochip_open(0)
        self.pin = pwm_pin
        self.open_us = int(open_us)
        self.closed_us = int(closed_us)
        self.settle_s = settle_s
        lgpio.gpio_claim_output(self._h, pwm_pin, 0)
        self._state = "unknown"

    def _pulse(self, width_us: int) -> None:
        width_us = max(400, min(2600, int(width_us)))
        duty = 100.0 * width_us / PERIOD_US
        self._lg.tx_pwm(self._h, self.pin, 50, duty)
        time.sleep(self.settle_s)

    def open(self) -> None:
        self._pulse(self.open_us)
        self._state = "open"

    def close(self) -> None:
        self._pulse(self.closed_us)
        self._state = "closed"

    @property
    def state(self) -> str:
        return self._state

    def release(self) -> None:
        """Stop driving the servo so it is not holding torque between moves."""
        self._lg.tx_pwm(self._h, self.pin, 50, 0)

    def shutdown(self) -> None:
        self.release()
        self._lg.gpiochip_close(self._h)
