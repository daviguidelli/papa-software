"""The Substrates tab's backend: camera points, scan, patterns, placed scrub."""
import numpy as np
import pytest

from papa.sim.arrays import SimArray
from papa.sim.mocks import MockCamera
from papa.vision import correction as corr
from tests.test_validate import api, saved  # noqa: F401  (fixture)

ARRAYS = (SimArray((155.0, 235.0), 12.0, 10.0, 0.0), SimArray((216.0, 236.0), 12.0, 10.0, 1.5),
          SimArray((155.0, 176.0), 12.0, 10.0, -2.0))
SKETCH = {"app": "papa-bridge-sketcher", "substrate": {"w": 12, "h": 10},
          "bridges": [{"id": "B1", "cx": 4, "cy": 5, "l": 1, "w": 10}]}


def teach(api, pts):
    """Pair pixels with gantry positions through the endpoint, like the GUI does."""
    mock = api.server.hub.printer._inner
    mock.homed = "xyz"
    for px, mm in pts:
        mock.x, mock.y = mm
        r = api.post("/api/align/camera/teach", json={"px": list(px), "photo_size": [1920, 1080]})
        assert r.status_code == 200, r.text
    return r.json()


def consistent_points():
    """Pixels and mm that agree exactly through the lens model (0.2 mm per plate px)."""
    c = corr.get_correction((1920, 1080))
    plate = np.array([[300.0, 200.0], [2000, 220], [1980, 1150], [320, 1130], [1100, 700]])
    th = np.radians(1.0)
    A = 0.2 * np.array([[np.cos(th), np.sin(th)], [np.sin(th), -np.cos(th)]])
    mm = plate @ A.T + [0, 330]
    return list(zip(c.from_plate(plate).tolist(), mm.tolist()))


@pytest.fixture
def aligned(api):
    api.post("/api/align/camera/delete", json={"all": True})   # the sim file's points are only for the old tab
    teach(api, consistent_points())
    api.server.hub.devices["camera"] = api.server.Device(
        "camera", lambda: MockCamera(arrays=ARRAYS, axes=api.server.hub.machine.axes,
                                     printer_axes=lambda: corr.PrinterAxes.from_detection(
                                         api.server.hub.cal.detection)))
    return api


def test_scan_needs_camera_points(api):
    api.post("/api/align/camera/delete", json={"all": True})
    r = api.post("/api/align/scan", json={"save": False})
    assert r.status_code == 400 and "Teach the camera points" in r.json()["error"]


def test_teaching_saves_points_and_reports_the_fit(api):
    api.post("/api/align/camera/delete", json={"all": True})
    j = teach(api, consistent_points())
    assert len(j["camera"]["points"]) == 5
    assert max(j["camera"]["fit"]["errors_mm"]) < 0.01 and not j["camera"]["fit"]["warnings"]
    c2m = saved(api)["detection"]["camera_to_machine"]
    assert len(c2m["points_px"]) == 5 and c2m["photo_size"] == [1920, 1080]
    r = api.post("/api/align/camera/teach", json={"px": [1, 2], "photo_size": [2028, 1520]})
    assert r.status_code == 400 and "clear them" in r.json()["error"]
    j = api.post("/api/align/camera/delete", json={"index": 4}).json()
    assert len(j["camera"]["points"]) == 4


def test_scan_places_every_array(aligned):
    j = aligned.post("/api/align/scan", json={"save": False}).json()
    subs = j["scan"]["substrates"]
    assert [s["id"] for s in subs] == ["S1", "S2", "S3"]
    for s, a in zip(subs, ARRAYS):
        th = np.radians(a.rot_deg)                     # true centre: local (6, 5) from the corner
        centre = (a.origin_mm[0] + 6 * np.cos(th) + 5 * np.sin(th), a.origin_mm[1] + 6 * np.sin(th) - 5 * np.cos(th))
        assert s["center_mm"] == pytest.approx(list(centre), abs=0.15)
        assert s["measured_mm"] == pytest.approx([12, 10], abs=0.35)
        assert s["rotation_deg"] == pytest.approx(a.rot_deg, abs=0.5)
        assert s["pattern_source"] == "scrub tab" and s["crop"].startswith("data:image/jpeg")
    assert j["scan"]["image"].startswith("data:image/jpeg;base64,")
    assert any("Measured" in p for p in subs[0]["problems"])        # 12 x 10 array vs the 25 x 25 default


def test_patterns_set_copy_and_clear(aligned):
    aligned.post("/api/align/scan", json={"save": False})
    j = aligned.post("/api/align/pattern", json={"targets": ["S1"], "sketch": SKETCH, "name": "comb"}).json()
    assert j["patterns"]["S1"]["bridges"] == 1 and j["scan"]["stale"]
    j = aligned.post("/api/align/pattern", json={"targets": ["S2", "S3"], "copy_from": "S1"}).json()
    subs = {s["id"]: s for s in j["scan"]["substrates"]}
    assert all(subs[i]["pattern_name"] == "comb" and not subs[i]["warnings"] for i in ("S1", "S2", "S3"))
    assert saved(aligned)["alignment"]["patterns"]["S3"]["area"] == {"w": 12.0, "h": 10.0}
    j = aligned.post("/api/align/pattern", json={"targets": ["S3"], "clear": True}).json()
    assert "S3" not in j["patterns"]
    assert aligned.post("/api/align/pattern", json={"targets": ["X1"], "clear": True}).status_code == 400


def test_placed_scrub_ends_at_the_rotated_corner(aligned):
    srv = aligned.server
    aligned.post("/api/align/scan", json={"save": False})
    aligned.post("/api/align/pattern", json={"targets": ["S1", "S2", "S3"], "sketch": SKETCH})
    mock = srv.hub.printer._inner
    mock.speed_factor = 0
    mock.homed = "xyz"
    mock.z = 40.0
    aligned.post("/api/guard/floor", json={"z": 39.0})
    assert aligned.post("/api/scrub/start-here", json={}).status_code == 200
    s2 = next(s for s in aligned.get("/api/align").json()["scan"]["substrates"] if s["id"] == "S2")
    assert s2["problems"] == []
    r = aligned.post("/api/align/scrub", json={"id": "S2"})
    assert r.status_code == 200, r.text
    srv.hub.thread.join(timeout=30)
    st = aligned.get("/api/scrub/status").json()
    assert st["finished"] and not st["error"]
    # the serpentine ends at local (12, 10): machine +X along local x, -Y along local y, turned
    th = np.radians(s2["rotation_deg"])
    ox, oy = s2["origin_mm"]
    dx, dy = 12.0, -10.0
    want = (ox + dx * np.cos(th) - dy * np.sin(th), oy + dx * np.sin(th) + dy * np.cos(th))
    assert (mock.x, mock.y) == pytest.approx(want, abs=1e-3)
    assert aligned.post("/api/align/scrub", json={"id": "S9"}).status_code == 404


def test_mismatched_substrate_needs_force(aligned):
    aligned.post("/api/align/scan", json={"save": False})
    mock = aligned.server.hub.printer._inner
    mock.homed, mock.z, mock.speed_factor = "xyz", 40.0, 0
    aligned.post("/api/guard/floor", json={"z": 39.0})
    aligned.post("/api/scrub/start-here", json={})
    r = aligned.post("/api/align/scrub", json={"id": "S1"})             # 25 x 25 sketch on a 12 x 10 array
    assert r.status_code == 400 and "Measured" in r.json()["error"]
    aligned.post("/api/align/settings", json={"size_tolerance_mm": 0.05})
    assert aligned.get("/api/align").json()["settings"]["size_tolerance_mm"] == 0.05
    assert aligned.post("/api/align/settings", json={"max_rotation_deg": 90}).status_code == 400
