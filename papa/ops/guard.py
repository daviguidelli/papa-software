"""The Z floor: the lowest Z the machine may be sent to, chosen by the operator.

Why it exists: the toolhead carries a gripper and swab that reach lower than
the stock nozzle, so Klipper's own limits do not protect the glass or the
bed. Nothing is assumed. Until an operator sets a floor, Z cannot move down at
all.

GuardedPrinter wraps the real or mock printer, so the validation page and the
orchestrator are held to the same floor. Raw G-code through script() is not
inspected; nothing in this codebase sends Z moves that way.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from papa.ops.jobs import STATE_DIR

EPS = 1e-6


class ZFloorError(RuntimeError):
    pass


class Guard:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else STATE_DIR / "z_floor.json"
        self._floor: float | None = None
        self._ceiling: float | None = None
        self._ref: str | None = None
        self._last_z: float | None = None
        try:
            d = json.loads(self.path.read_text())
            v, c = d.get("z"), d.get("ceiling")
            self._floor = None if v is None else float(v)
            self._ceiling = None if c is None else float(c)
            self._ref = d.get("ref")
            lz = d.get("last_z")
            self._last_z = None if lz is None else float(lz)
        except Exception:                                  # noqa: BLE001
            self._floor, self._ceiling, self._ref, self._last_z = None, None, None, None

    @property
    def floor(self) -> float | None:
        return self._floor

    @property
    def ceiling(self) -> float | None:
        return self._ceiling

    @property
    def ref(self) -> str | None:
        """Identifies the current hand-declared Z reference; changes every time one is declared."""
        return self._ref

    @property
    def last_z(self) -> float | None:
        """The last Z the machine reported while Z was set; survives restarts."""
        return self._last_z

    def remember_z(self, z: float) -> None:
        if self._last_z is None or abs(self._last_z - z) > 0.005:
            self._last_z = float(z)
            self._save()

    @property
    def allow_z_home(self) -> bool:
        return os.environ.get("PAPA_ALLOW_Z_HOME", "") not in ("", "0", "false", "False")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"z": self._floor, "ceiling": self._ceiling, "ref": self._ref, "last_z": self._last_z}))
        os.replace(tmp, self.path)

    def set_floor(self, z: float) -> None:
        self._floor = float(z)
        self._save()

    def clear(self) -> None:
        self._floor = None
        self._save()

    def set_ceiling(self, z: float | None) -> None:
        self._ceiling = None if z is None else float(z)
        self._save()

    def reset_reference(self, ceiling: float | None) -> None:
        """A new Z reference was declared: the old floor belonged to the old one."""
        self._floor = None
        self._ceiling = None if ceiling is None else float(ceiling)
        self._ref = str(time.time_ns())
        self._save()

    def check_target(self, target_z: float, current_z: Callable[[], float]) -> None:
        """Raise unless this Z target is allowed. Moving up is allowed up to the
        ceiling, if one is set."""
        if (self._ceiling is not None and target_z > self._ceiling + EPS
                and target_z > current_z() + EPS):
            raise ZFloorError(f"Z {target_z:.3f} is above the ceiling {self._ceiling:.3f}")
        if self._floor is not None and target_z >= self._floor - EPS:
            return
        if target_z >= current_z() - EPS:
            return
        if self._floor is None:
            raise ZFloorError("Set the Z floor before moving Z down")
        raise ZFloorError(f"Z {target_z:.3f} is below the floor {self._floor:.3f}")


class GuardedPrinter:
    def __init__(self, inner: Any, guard: Guard) -> None:
        self._inner = inner
        self.guard = guard

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def move(self, x=None, y=None, z=None, speed_mm_s: float = 50.0) -> None:
        if z is not None:
            self.guard.check_target(float(z), lambda: self._inner.position().z)
        self._inner.move(x=x, y=y, z=z, speed_mm_s=speed_mm_s)

    def move_relative(self, dx=0.0, dy=0.0, dz=0.0, speed_mm_s: float = 20.0) -> None:
        if dz:
            cur = self._inner.position().z
            self.guard.check_target(cur + dz, lambda: cur)
        self._inner.move_relative(dx=dx, dy=dy, dz=dz, speed_mm_s=speed_mm_s)
