"""Moonraker client. The mainboard is a motion server; this is the only way
the Pi talks to it.

Four calls are enough: send a G-code script, read the live position, home,
stop. Every move script ends with M400 so the call returns when motion has
actually finished rather than when it was queued.

Two setup gotchas, both outside this file:
  * Moonraker must list the Pi's address under trusted_clients, or it refuses.
  * A direct Pi-to-mainboard cable has no router, so both ends need fixed IPs
    on the same subnet.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


class MoonrakerError(RuntimeError):
    pass


@dataclass
class Position:
    x: float
    y: float
    z: float
    homed: str = ""

    @property
    def is_homed(self) -> bool:
        return all(a in self.homed for a in "xyz")


class Moonraker:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._c = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def close(self) -> None:
        self._c.close()

    def __enter__(self) -> "Moonraker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- the four calls ---

    def script(self, gcode: str, wait: bool = True) -> None:
        """Send G-code. With wait, M400 is appended so motion completes first."""
        body = gcode.rstrip()
        if wait and not body.upper().endswith("M400"):
            body += "\nM400"
        r = self._c.post("/printer/gcode/script", json={"script": body})
        if r.status_code != 200:
            raise MoonrakerError(f"gcode rejected ({r.status_code}): {r.text[:300]}")

    def position(self) -> Position:
        r = self._c.get(
            "/printer/objects/query",
            params={"toolhead": "position,homed_axes"},
        )
        if r.status_code != 200:
            raise MoonrakerError(f"query failed ({r.status_code}): {r.text[:300]}")
        th = r.json()["result"]["status"]["toolhead"]
        p = th["position"]
        return Position(float(p[0]), float(p[1]), float(p[2]), str(th.get("homed_axes", "")))

    def info(self) -> dict:
        r = self._c.get("/printer/info")
        if r.status_code != 200:
            raise MoonrakerError(f"info failed ({r.status_code})")
        return r.json()["result"]

    def emergency_stop(self) -> None:
        """Hard stop. Klipper shuts down; a firmware restart and re-home follow."""
        self._c.post("/printer/emergency_stop")

    # --- conveniences built on script() ---

    def home(self, axes: str = "") -> None:
        self.script(f"G28 {axes}".strip())

    def firmware_restart(self) -> None:
        self._c.post("/printer/firmware_restart")

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
        try:
            self.info()
            return True
        except Exception:
            return False
