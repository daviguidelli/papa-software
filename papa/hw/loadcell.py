"""TAL221 load cell through an HX711.

The HX711 delivers roughly 10 to 80 samples per second, which is why Z
calibration descends in steps and reads between them instead of watching a
continuous signal while moving.
"""
from __future__ import annotations

import statistics
import time
from typing import Protocol


class LoadCell(Protocol):
    def read_counts(self) -> int: ...
    def grams(self, samples: int = 5) -> float: ...
    def tare(self, samples: int = 20) -> int: ...


class HX711LoadCell:
    """Bit-banged HX711 reader. Uses lgpio, present on current Raspberry Pi OS."""

    def __init__(self, dat_pin: int, clk_pin: int,
                 scale_counts_per_g: float = 1.0, tare_counts: int = 0) -> None:
        import lgpio  # imported here so the module loads fine off-Pi

        self._lg = lgpio
        self._h = lgpio.gpiochip_open(0)
        self.dat, self.clk = dat_pin, clk_pin
        lgpio.gpio_claim_input(self._h, dat_pin)
        lgpio.gpio_claim_output(self._h, clk_pin, 0)
        self.scale = scale_counts_per_g or 1.0
        self.offset = tare_counts

    def _ready(self) -> bool:
        return self._lg.gpio_read(self._h, self.dat) == 0

    def read_counts(self) -> int:
        deadline = time.time() + 1.0
        while not self._ready():
            if time.time() > deadline:
                raise TimeoutError("HX711 not ready: check DAT/CLK wiring and power")
            time.sleep(0.001)
        value = 0
        for _ in range(24):
            self._lg.gpio_write(self._h, self.clk, 1)
            self._lg.gpio_write(self._h, self.clk, 0)
            value = (value << 1) | self._lg.gpio_read(self._h, self.dat)
        # 25th pulse selects channel A, gain 128
        self._lg.gpio_write(self._h, self.clk, 1)
        self._lg.gpio_write(self._h, self.clk, 0)
        if value & 0x800000:  # sign extend 24-bit two's complement
            value -= 1 << 24
        return value

    def grams(self, samples: int = 5) -> float:
        """Median of a few samples: one outlier should not stop a descent."""
        reads = [self.read_counts() for _ in range(max(1, samples))]
        return (statistics.median(reads) - self.offset) / self.scale

    def tare(self, samples: int = 20) -> int:
        self.offset = int(statistics.median(self.read_counts() for _ in range(samples)))
        return self.offset

    def calibrate_with_known_weight(self, grams_known: float, samples: int = 20) -> float:
        if grams_known <= 0:
            raise ValueError("known weight must be positive")
        raw = statistics.median(self.read_counts() for _ in range(samples))
        self.scale = (raw - self.offset) / grams_known
        return self.scale

    def close(self) -> None:
        self._lg.gpiochip_close(self._h)
