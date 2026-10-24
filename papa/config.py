"""Config loading. Two files, deliberately separate.

machine.yaml      set once by whoever builds the machine
calibration.yaml  written by the Calibration tab, the only file an operator
                  ever changes

Both are plain dicts wrapped in small accessors so a missing key fails loudly
at load time instead of halfway through a job.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("PAPA_CONFIG_DIR", REPO_ROOT / "config"))


def sim_mode() -> bool:
    """True when every hardware driver should be swapped for a mock."""
    return os.environ.get("PAPA_SIM", "") not in ("", "0", "false", "False")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


@dataclass(frozen=True)
class Machine:
    raw: dict[str, Any]

    @property
    def moonraker_url(self) -> str:
        return str(self.raw["moonraker_url"]).rstrip("/")

    @property
    def timeout_s(self) -> float:
        return float(self.raw.get("moonraker_timeout_s", 30))

    @property
    def axes(self) -> dict[str, str]:
        return dict(self.raw["axes"])

    @property
    def limits(self) -> dict[str, list[float]]:
        return {k: [float(v[0]), float(v[1])] for k, v in self.raw["limits_mm"].items()}

    @property
    def speeds(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["speeds_mm_s"].items()}

    @property
    def zcal(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["zcal"].items()}

    @property
    def wear(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["wear"].items()}

    @property
    def toolpath(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw.get("toolpath", {}).items()}

    @property
    def pins(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.raw["pins"].items()}

    def clamp(self, axis: str, value: float) -> float:
        lo, hi = self.limits[axis]
        return min(max(value, lo), hi)

    def in_limits(self, x: float, y: float, z: float) -> bool:
        for axis, v in (("x", x), ("y", y), ("z", z)):
            lo, hi = self.limits[axis]
            if not (lo <= v <= hi):
                return False
        return True


@dataclass(frozen=True)
class Swab:
    id: int
    row: int
    side: str  # "right" | "left"
    x: float
    y: float
    z_grip: float
    calibrated: bool = False

    @property
    def label(self) -> str:
        return f"{self.row}{'R' if self.side == 'right' else 'L'}"


@dataclass(frozen=True)
class Slot:
    id: int
    x: float
    y: float
    calibrated: bool = False


@dataclass(frozen=True)
class Calibration:
    raw: dict[str, Any]
    swabs: list[Swab] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "Calibration":
        swabs = [Swab(**s) for s in raw.get("swab_rack", {}).get("swabs", [])]
        slots = [Slot(**s) for s in raw.get("slide_slots", {}).get("slots", [])]
        return cls(raw=raw, swabs=swabs, slots=slots)

    # --- substrate and pattern area ---
    @property
    def substrate(self) -> dict[str, float]:
        s = self.raw.get("substrate", {})
        return {
            "width_mm": float(s.get("width_mm", 25.0)),
            "length_mm": float(s.get("length_mm", 75.0)),
            "pattern_size_mm": float(s.get("pattern_size_mm", 25.0)),
        }

    @property
    def pattern_area(self) -> tuple[float, float]:
        """The square the DXF is fitted into, in slide-local mm."""
        side = self.substrate["pattern_size_mm"]
        return side, side

    def pattern_fits(self) -> bool:
        s = self.substrate
        return s["pattern_size_mm"] <= min(s["width_mm"], s["length_mm"])

    @property
    def rack_rows(self) -> int:
        return int(self.raw.get("swab_rack", {}).get("rows", 15))

    @property
    def swabs_per_slide(self) -> int:
        return int(self.raw.get("swab_rack", {}).get("swabs_per_slide", 2))

    @property
    def slot_grid(self) -> tuple[int, int]:
        g = self.raw.get("slide_slots", {})
        return int(g.get("rows", 3)), int(g.get("cols", 3))

    @property
    def load_cell(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["load_cell"].items()}

    @property
    def disposal(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["disposal"].items()}

    @property
    def surface_offset_mm(self) -> float:
        return float(self.raw.get("slide_surface_offset_mm", 0.0))

    @property
    def gripper(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.raw["gripper"].items()}

    @property
    def detection(self) -> dict[str, Any]:
        return dict(self.raw.get("detection", {}))

    def swab(self, swab_id: int) -> Swab:
        for s in self.swabs:
            if s.id == swab_id:
                return s
        raise KeyError(f"no swab with id {swab_id}")

    def slot(self, slot_id: int) -> Slot:
        for s in self.slots:
            if s.id == slot_id:
                return s
        raise KeyError(f"no slide slot with id {slot_id}")

    def uncalibrated(self, slot_ids: list[int], swab_ids: list[int]) -> list[str]:
        """Names of everything a job needs that has not been taught yet."""
        missing: list[str] = []
        for sid in slot_ids:
            try:
                if not self.slot(sid).calibrated:
                    missing.append(f"slide slot {sid}")
            except KeyError:
                missing.append(f"slide slot {sid} (not in calibration.yaml)")
        for wid in swab_ids:
            try:
                if not self.swab(wid).calibrated:
                    missing.append(f"swab {self.swab(wid).label}")
            except KeyError:
                missing.append(f"swab id {wid} (not in calibration.yaml)")
        if not self.load_cell.get("scale_counts_per_g"):
            missing.append("load cell scale")
        det = self.detection.get("camera_to_machine", {})
        if len(det.get("points_px") or []) < 4:
            missing.append("camera to machine mapping")
        return missing


def load_machine(path: Path | None = None) -> Machine:
    return Machine(_load(path or CONFIG_DIR / "machine.yaml"))


def load_calibration(path: Path | None = None) -> Calibration:
    """Real calibration if it exists; otherwise the simulated one in sim mode,
    so a fresh clone can dry-run. The example file is the last resort and is
    deliberately incomplete, so preflight refuses rather than inventing values.
    """
    if path is not None:
        return Calibration.from_raw(_load(path))
    for candidate in (
        CONFIG_DIR / "calibration.yaml",
        *( (CONFIG_DIR / "calibration.sim.yaml",) if sim_mode() else () ),
        CONFIG_DIR / "calibration.example.yaml",
    ):
        if candidate.exists():
            return Calibration.from_raw(_load(candidate))
    raise FileNotFoundError(f"no calibration file found in {CONFIG_DIR}")


def save_calibration(cal_raw: dict[str, Any], path: Path | None = None) -> Path:
    """Write calibration.yaml atomically: a half-written file is worse than none."""
    p = path or CONFIG_DIR / "calibration.yaml"
    tmp = p.with_suffix(".yaml.tmp")
    with tmp.open("w") as fh:
        yaml.safe_dump(cal_raw, fh, sort_keys=False)
    os.replace(tmp, p)
    return p
