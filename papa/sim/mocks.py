"""Mocks for every hardware interface, so the whole system runs with no printer.

PAPA_SIM=1 swaps these in. The state machine is unchanged: it issues the same
calls, the Z probe really descends and really stops, and every G-code line is
logged so a simulated job doubles as a regression fixture.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from papa.hw.moonraker import Position


@dataclass
class MockMoonraker:
    x: float = 0.0
    y: float = 0.0
    z: float = 100.0
    homed: str = ""
    speed_factor: float = 200.0   # simulated mm/s, keeps dry runs quick
    log: list[str] = field(default_factory=list)
    shut_down: bool = False
    base_url: str = "mock://moonraker"

    def close(self) -> None:
        pass

    def script(self, gcode: str, wait: bool = True) -> None:
        if self.shut_down:
            raise RuntimeError("Klipper is shut down; firmware restart required")
        body = gcode.rstrip()
        if wait and not body.upper().endswith("M400"):
            body += "\nM400"
        relative = False
        for raw in body.splitlines():
            line = raw.strip()
            if not line:
                continue
            self.log.append(line)
            up = line.upper()
            if up.startswith("G91"):
                relative = True
            elif up.startswith("G90"):
                relative = False
            elif up.startswith("G28"):
                axes = [a for a in "XYZ" if a in up[3:]] or ["X", "Y", "Z"]
                for a in axes:
                    setattr(self, a.lower(), 0.0)
                    if a.lower() not in self.homed:
                        self.homed += a.lower()
            elif up.startswith(("G0 ", "G1 ", "G0\t", "G1\t")):
                tx, ty, tz = self.x, self.y, self.z
                for tok in up.split()[1:]:
                    try:
                        v = float(tok[1:])
                    except ValueError:
                        continue
                    if tok[0] == "X":
                        tx = self.x + v if relative else v
                    elif tok[0] == "Y":
                        ty = self.y + v if relative else v
                    elif tok[0] == "Z":
                        tz = self.z + v if relative else v
                if not (-0.001 <= tx <= 350 and -0.001 <= ty <= 350 and -0.001 <= tz <= 340):
                    raise RuntimeError(f"Move out of range: {tx:.3f} {ty:.3f} {tz:.3f}")
                dist = ((tx - self.x) ** 2 + (ty - self.y) ** 2 + (tz - self.z) ** 2) ** 0.5
                self.x, self.y, self.z = tx, ty, tz
                if self.speed_factor > 0:
                    time.sleep(min(dist / self.speed_factor, 0.05))

    def position(self) -> Position:
        return Position(self.x, self.y, self.z, self.homed)

    def info(self) -> dict:
        return {"state": "shutdown" if self.shut_down else "ready", "hostname": "mock"}

    def emergency_stop(self) -> None:
        self.shut_down = True
        self.log.append("!! emergency_stop")

    def firmware_restart(self) -> None:
        self.shut_down = False
        self.homed = ""
        self.log.append("!! firmware_restart")

    def home(self, axes: str = "") -> None:
        self.script(f"G28 {axes}".strip())

    def move(self, x=None, y=None, z=None, speed_mm_s: float = 50.0) -> None:
        parts = ["G90", "G1"]
        for name, v in (("X", x), ("Y", y), ("Z", z)):
            if v is not None:
                parts[-1] += f" {name}{v:.4f}"
        parts[-1] += f" F{speed_mm_s * 60:.0f}"
        self.script("\n".join(parts))

    def move_relative(self, dx=0.0, dy=0.0, dz=0.0, speed_mm_s: float = 20.0) -> None:
        g = "G91\nG1"
        for name, v in (("X", dx), ("Y", dy), ("Z", dz)):
            if v:
                g += f" {name}{v:.4f}"
        g += f" F{speed_mm_s * 60:.0f}\nG90"
        self.script(g)

    def ping(self) -> bool:
        return not self.shut_down


@dataclass
class MockLoadCell:
    """Force as a function of simulated Z against a virtual surface.

    Below contact_z the swab compresses, so force rises with depth. With
    has_swab False nothing ever touches, which is exactly how a failed grip
    presents itself on the real machine.
    """
    printer: MockMoonraker
    contact_z: float = 12.0
    stiffness_g_per_mm: float = 60.0
    has_swab: bool = True
    noise_g: float = 0.05
    offset: int = 0
    scale: float = 1.0

    def grams(self, samples: int = 5) -> float:
        if not self.has_swab:
            return float(np.random.normal(0, self.noise_g))
        depth = self.contact_z - self.printer.z
        f = max(0.0, depth) * self.stiffness_g_per_mm
        return float(f + np.random.normal(0, self.noise_g))

    def read_counts(self) -> int:
        return int(self.grams() * self.scale + self.offset)

    def tare(self, samples: int = 20) -> int:
        self.offset = 0
        return 0

    def calibrate_with_known_weight(self, grams_known: float, samples: int = 20) -> float:
        self.scale = 1.0
        return self.scale


@dataclass
class MockGripper:
    log: list[str] = field(default_factory=list)
    fail_next: bool = False
    state: str = "unknown"

    def open(self) -> None:
        self.log.append("open")
        self.state = "open"

    def close(self) -> None:
        self.log.append("close")
        if self.fail_next:
            self.fail_next = False
            self.state = "failed"
            return
        self.state = "closed"

    def release(self) -> None:
        self.log.append("release")

    def shutdown(self) -> None:
        self.log.append("shutdown")


@dataclass
class MockCamera:
    """A synthetic frame with blue films in chosen slots and blue stems in
    chosen rack positions, so the detection code has something real to filter."""
    width: int = 1920
    height: int = 1080
    slide_slots: tuple[int, ...] = (1, 2, 3, 5, 6, 8)
    loaded_swabs: tuple[str, ...] = ()
    px_per_mm: float = 5.0

    def frame(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        img[:, :, 2] = 34  # faintly cool background, like the lab bench
        for i, sid in enumerate(self.slide_slots):
            r, c = divmod(sid - 1, 3)
            x = int(520 + c * 300)
            y = int(180 + r * 280)
            s = int(25 * self.px_per_mm)
            img[y:y + s, x:x + s] = (28, 92, 190)  # film blue, RGB
        for label in self.loaded_swabs:
            row = int(label[:-1])
            side = label[-1]
            cx = int(120 + (0 if side == "R" else 70))
            cy = int(120 + (row - 1) * 60)
            rr = int(self.px_per_mm * 2.54 / 2)
            yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
            mask = xx * xx + yy * yy <= rr * rr
            sub = img[cy - rr:cy + rr + 1, cx - rr:cx + rr + 1]
            if sub.shape[:2] == mask.shape:
                sub[mask] = (36, 108, 214)
        return img

    def close(self) -> None:
        pass
