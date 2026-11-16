"""A capture that never returns must not freeze the camera for good."""
from tests.test_validate import api  # noqa: F401  (fixture)
import threading
import time

import numpy as np
import pytest

from papa.hw.camera import CameraStuck, LockedCamera


class Hangs:
    def __init__(self):
        self.release = threading.Event()

    def frame(self):
        self.release.wait()
        return np.zeros((2, 2, 3), np.uint8)


class Works:
    def frame(self):
        return np.ones((2, 2, 3), np.uint8)


def test_hung_capture_marks_the_camera_stuck_and_fails_fast(monkeypatch):
    monkeypatch.setattr(LockedCamera, "FRAME_S", 0.2)
    inner = Hangs()
    cam = LockedCamera(inner)
    with pytest.raises(CameraStuck):
        cam.frame()
    t0 = time.monotonic()
    with pytest.raises(CameraStuck):
        cam.frame()                      # no second wait
    assert time.monotonic() - t0 < 0.1 and cam.stuck
    inner.release.set()


def test_normal_captures_and_errors_pass_through():
    cam = LockedCamera(Works())
    assert cam.frame().sum() == 12 and cam.still().sum() == 12 and not cam.stuck


def test_stuck_camera_answers_503_and_does_not_hang(api, monkeypatch):
    cam = LockedCamera(Works())
    object.__setattr__(cam, "stuck", "a picture did not arrive within 10 s")
    api.server.hub.devices["camera"]._obj = cam
    r = api.get("/api/camera/snapshot", params={"width": 160})
    assert r.status_code == 503 and "did not arrive" in r.json()["error"]
    assert api.get("/api/devices").json()["devices"]["camera"]["stuck"] is True
