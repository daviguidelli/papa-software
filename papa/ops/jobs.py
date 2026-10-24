"""The job file: the queue plus the resume point, written after every block.

A crash or a power cut must not lose a long queue, so state is persisted on
every block boundary rather than at the end of a slide.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from papa.config import REPO_ROOT

STATE_DIR = Path(os.environ.get("PAPA_STATE_DIR", REPO_ROOT / "var"))


@dataclass
class SlideProgress:
    slot_id: int
    done: bool = False
    block_index: int = 0          # next block to scribe
    swabs_used: list[str] = field(default_factory=list)


@dataclass
class Job:
    dxf_name: str
    slot_ids: list[int]
    swabs_per_slide: int
    pattern_size_mm: float
    total_blocks: int
    created_at: float = field(default_factory=time.time)
    state: str = "queued"         # queued|running|paused|done|error
    current_slot: int | None = None
    current_swab: str | None = None
    message: str = ""
    slides: list[SlideProgress] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.slides:
            self.slides = [SlideProgress(slot_id=s) for s in self.slot_ids]

    # --- progress ---

    def slide(self, slot_id: int) -> SlideProgress:
        for s in self.slides:
            if s.slot_id == slot_id:
                return s
        raise KeyError(f"slot {slot_id} is not in this job")

    def next_slide(self) -> SlideProgress | None:
        for s in self.slides:
            if not s.done:
                return s
        return None

    @property
    def fraction(self) -> float:
        if not self.slides or not self.total_blocks:
            return 0.0
        done = sum(
            (self.total_blocks if s.done else min(s.block_index, self.total_blocks))
            for s in self.slides
        )
        return done / (self.total_blocks * len(self.slides))

    # --- persistence ---

    def save(self, path: Path | None = None) -> Path:
        p = path or STATE_DIR / "job.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, p)        # atomic: a torn job file is worse than none
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> "Job | None":
        p = path or STATE_DIR / "job.json"
        if not p.exists():
            return None
        raw = json.loads(p.read_text())
        slides = [SlideProgress(**s) for s in raw.pop("slides", [])]
        job = cls(**raw)
        job.slides = slides
        return job
