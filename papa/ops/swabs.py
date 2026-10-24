"""Swab inventory, the pick rule, and wear tracking.

The rack is two columns by N rows, left of the bed. A left-side swab is
pickable only once the right-side swab in the same row is gone, because
reaching the left one first collides with the right one. That single rule
gives the right-first order, and still works on a partly loaded rack.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from papa.config import Calibration, Swab


class RackEmpty(RuntimeError):
    """No pickable swab left. The job pauses and asks for a reload."""


@dataclass
class Wear:
    """Per-swab counters. Either limit reached triggers a mid-slide swap."""
    path_mm: float = 0.0
    contact_s: float = 0.0

    def add(self, path_mm: float = 0.0, contact_s: float = 0.0) -> None:
        self.path_mm += path_mm
        self.contact_s += contact_s

    def worn(self, max_path_mm: float, max_contact_s: float) -> bool:
        return self.path_mm >= max_path_mm or self.contact_s >= max_contact_s

    def fraction(self, max_path_mm: float, max_contact_s: float) -> float:
        a = self.path_mm / max_path_mm if max_path_mm else 0.0
        b = self.contact_s / max_contact_s if max_contact_s else 0.0
        return min(1.0, max(a, b))


@dataclass
class Rack:
    """Which positions hold a swab, which are spent, and what is pickable next.

    `loaded` comes from the camera, not from a count kept in software: a swab
    someone removed by hand should not still be in the plan.
    """
    cal: Calibration
    loaded: set[str] = field(default_factory=set)   # labels like "3R"
    used: set[str] = field(default_factory=set)

    @classmethod
    def from_detection(cls, cal: Calibration, present_labels: list[str]) -> "Rack":
        return cls(cal=cal, loaded=set(present_labels))

    def _by_label(self, label: str) -> Swab | None:
        for s in self.cal.swabs:
            if s.label == label:
                return s
        return None

    def gone(self, label: str) -> bool:
        """Used, or never loaded. Either way the position is clear."""
        return label in self.used or label not in self.loaded

    def pickable(self, label: str) -> bool:
        if self.gone(label):
            return False
        row, side = int(label[:-1]), label[-1]
        if side == "L":
            return self.gone(f"{row}R")
        return True

    def order(self) -> list[str]:
        """Pick order: for each row, right then left."""
        out: list[str] = []
        for row in range(1, self.cal.rack_rows + 1):
            for side in ("R", "L"):
                label = f"{row}{side}"
                if label in self.loaded and label not in self.used:
                    out.append(label)
        return out

    def next_label(self) -> str:
        for label in self.order():
            if self.pickable(label):
                return label
        raise RackEmpty("no pickable swab remains; reload the rack")

    def take(self, label: str) -> Swab:
        s = self._by_label(label)
        if s is None:
            raise KeyError(f"rack position {label} is not in calibration.yaml")
        if not s.calibrated:
            raise ValueError(f"rack position {label} has not been calibrated")
        self.used.add(label)
        return s

    def available(self) -> int:
        return len(self.order())

    def reserve(self, slide_ids: list[int], per_slide: int) -> dict[int, list[str]]:
        """Assign swabs to slides in pick order, for the preflight count.

        Reserving is a plan, not a promise: if wear needs more than reserved,
        the next rack swab is taken at run time.
        """
        plan: dict[int, list[str]] = {}
        avail = self.order()
        for i, sid in enumerate(slide_ids):
            plan[sid] = avail[i * per_slide:(i + 1) * per_slide]
        return plan
