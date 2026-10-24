"""Pi Camera Module 3 capture.

Detection runs on still frames, not video: a still at scan time and another
just before each slide starts is all the state machine needs.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Camera(Protocol):
    def frame(self) -> np.ndarray: ...


class PiCamera:
    def __init__(self, size: tuple[int, int] = (1920, 1080)) -> None:
        from picamera2 import Picamera2

        self._cam = Picamera2()
        cfg = self._cam.create_still_configuration(main={"size": size, "format": "RGB888"})
        self._cam.configure(cfg)
        self._cam.start()

    def frame(self) -> np.ndarray:
        """One RGB frame as HxWx3 uint8."""
        return self._cam.capture_array("main")

    def close(self) -> None:
        self._cam.stop()
