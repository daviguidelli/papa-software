"""Raspberry Pi HQ Camera (IMX477) capture. Focus is a manual ring on the lens; there is no autofocus.

Detection runs on still frames, not video: a still at scan time and another
just before each slide starts is all the state machine needs.
"""
from __future__ import annotations

import threading
from typing import Any, Protocol

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
        """One RGB frame as HxWx3 uint8.

        Picamera2's "RGB888" is laid out B,G,R in memory (OpenCV order), so it
        is flipped here. Skipping this swaps red and blue and breaks detection.
        """
        return np.ascontiguousarray(self._cam.capture_array("main")[:, :, ::-1])

    def still(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """One RGB photo at `size` (e.g. the full 4056 x 3040), switching the
        sensor mode for this capture only; the next frame() is back at the
        live size. None or the live size is the same as frame()."""
        if size is None or tuple(size) == tuple(self._cam.camera_configuration()["main"]["size"]):
            return self.frame()
        cfg = self._cam.create_still_configuration(main={"size": tuple(size), "format": "RGB888"})
        arr = self._cam.switch_mode_and_capture_array(cfg, "main")
        return np.ascontiguousarray(arr[:, :, ::-1])

    def close(self) -> None:
        self._cam.stop()


class CameraStuck(RuntimeError):
    """A capture did not finish in time: the camera stopped sending frames."""


class LockedCamera:
    """One frame at a time. The live view and a scan share the one sensor, and
    Picamera2 captures are not meant to overlap.

    Each capture has a deadline. A capture that never returns (a stalled
    sensor or driver) would otherwise hold the lock forever and freeze every
    later picture, so after a deadline the camera is marked stuck and every
    call fails at once with CameraStuck until the process restarts."""

    FRAME_S = 10.0
    STILL_S = 25.0

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "stuck", None)

    def _run(self, fn: Any, deadline: float) -> np.ndarray:
        if self.stuck:
            raise CameraStuck(self.stuck)
        if not self._lock.acquire(timeout=deadline):
            raise CameraStuck("the camera is still busy with an earlier picture")
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["v"] = fn()
            except BaseException as exc:                  # noqa: BLE001
                box["e"] = exc
            finally:
                self._lock.release()

        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(deadline)
        if t.is_alive():
            object.__setattr__(self, "stuck", f"a picture did not arrive within {deadline:g} s: "
                                              "the camera stopped sending frames")
            raise CameraStuck(self.stuck)
        if "e" in box:
            raise box["e"]
        return box["v"]

    def frame(self) -> np.ndarray:
        return self._run(self._inner.frame, self.FRAME_S)

    def still(self, size: tuple[int, int] | None = None) -> np.ndarray:
        inner = self._inner
        return self._run(lambda: inner.still(size) if hasattr(inner, "still") else inner.frame(), self.STILL_S)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)
