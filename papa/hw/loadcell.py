"""TAL221 load cell through an HX711.

The HX711 delivers roughly 10 to 80 samples per second, which is why Z
calibration descends in steps and reads between them instead of watching a
continuous signal while moving.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from typing import Any, Protocol


G_TO_N = 0.00980665       # one gram of mass, as a force in newtons


class LoadCell(Protocol):
    def read_counts(self) -> int: ...
    def grams(self, samples: int = 5) -> float: ...
    def newtons(self, samples: int = 5) -> float: ...
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
        # Until a known weight has set the scale, "grams" would really be raw
        # counts. The GUI must say counts, not grams.
        self.scaled = bool(scale_counts_per_g) and float(scale_counts_per_g) != 1.0
        self._bus = threading.Lock()

    def _ready(self) -> bool:
        return self._lg.gpio_read(self._h, self.dat) == 0

    def read_counts(self) -> int:
        with self._bus:                      # one 24-bit read at a time, even with a sampler running
            for attempt in range(3):         # a dropped read (chip reset) is retried, not fatal
                try:
                    return self._read_counts()
                except TimeoutError:
                    if attempt == 2:
                        raise
            raise TimeoutError("HX711 not ready")

    def wake(self, timeout_s: float = 12.0) -> None:
        """CLK low and keep asking until the chip answers. After it has been idle
        or reset it can take several seconds before DAT goes low."""
        t0 = time.monotonic()
        while True:
            self._lg.gpio_write(self._h, self.clk, 0)
            try:
                self.read_counts()
                return
            except TimeoutError:
                if time.monotonic() - t0 > timeout_s:
                    raise
                time.sleep(0.2)

    def _read_counts(self) -> int:
        self._lg.gpio_write(self._h, self.clk, 0)    # CLK high > 60 us = power-down; make sure it is low
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

    def settle(self, min_s: float = 0.8, timeout_s: float = 6.0, window: int = 8) -> bool:
        """Wait out the start-up swing. After a power dip or a long idle the chip
        outputs wild values (hundreds of N) for a second or two. Read and discard
        until the last `window` readings agree, then report whether they did."""
        self.wake()
        tol = max(300.0, abs(self.scale) * 5.0)          # about 0.05 N
        t0, reads = time.monotonic(), []
        while time.monotonic() - t0 < timeout_s:
            reads.append(self.read_counts())
            if time.monotonic() - t0 >= min_s and len(reads) >= window:
                w = reads[-window:]
                if max(w) - min(w) <= tol:
                    return True
        return False

    def grams(self, samples: int = 5) -> float:
        """Median of a few samples: one outlier should not stop a descent."""
        reads = [self.read_counts() for _ in range(max(1, samples))]
        return (statistics.median(reads) - self.offset) / self.scale

    def newtons(self, samples: int = 5) -> float:
        """Force in newtons. The scale is set once with a known mass in grams."""
        return self.grams(samples) * G_TO_N

    def tare(self, samples: int = 20) -> int:
        self.offset = int(statistics.median(self.read_counts() for _ in range(samples)))
        return self.offset

    def calibrate_with_known_weight(self, grams_known: float, samples: int = 20) -> float:
        if grams_known <= 0:
            raise ValueError("known weight must be positive")
        raw = statistics.median(self.read_counts() for _ in range(samples))
        self.scale = (raw - self.offset) / grams_known
        self.scaled = True
        return self.scale

    def close(self) -> None:
        self._lg.gpiochip_close(self._h)


class LoadCellSampler:
    """Reads the cell continuously in the background and keeps the recent raw
    counts, so the GUI can draw a live trace and judge the noise.

    It runs only while something is asking for data: every call to touch() keeps
    it alive for idle_s seconds, then it stops by itself. stop() ends it at once,
    which the probe, tare and scale calls use so nothing else is reading the
    chip at the same time.
    """

    def __init__(self, cell: Any, maxlen: int = 6000, idle_s: float = 1.5,
                 min_interval_s: float = 0.05, warmup_s: float = 2.0) -> None:
        self.cell = cell
        self.warmup_s = warmup_s if hasattr(cell, "settle") else 0.0   # only the real chip swings at start-up
        self.idle_s = idle_s
        self.min_interval_s = min_interval_s
        self._buf: deque[tuple[int, float, int]] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._touched = 0.0
        self._t0 = time.monotonic()
        self.error = ""

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def touch(self) -> None:
        self._touched = time.monotonic()
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="loadcell-sampler")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _run(self) -> None:
        t_run = time.monotonic()
        while not self._stop.is_set() and time.monotonic() - self._touched < self.idle_s:
            t_start = time.monotonic()
            try:
                counts = int(self.cell.read_counts())
            except Exception as exc:                                  # noqa: BLE001
                self.error = str(exc)
                time.sleep(0.25)
                continue
            self.error = ""
            if time.monotonic() - t_run < self.warmup_s:      # start-up swing: do not show or use it
                continue
            with self._lock:
                self._seq += 1
                self._buf.append((self._seq, t_start - self._t0, counts))
            rest = self.min_interval_s - (time.monotonic() - t_start)
            if rest > 0:
                time.sleep(rest)

    def since(self, seq: int, limit: int = 2000) -> tuple[int, list[tuple[float, int]]]:
        """(latest seq, samples newer than seq). A client that asks with seq 0
        gets the most recent `limit` samples."""
        with self._lock:
            items = [(t, c) for (q, t, c) in self._buf if q > seq]
            last = self._seq
        return last, items[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
