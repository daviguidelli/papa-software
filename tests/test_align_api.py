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
    # the mock camera is 1080p over the whole bed (~5 px/mm), coarser than a real
    # full-resolution scan; accuracy itself is tested in test_alignment
    api.post("/api/align/settings", json={"size_tolerance_mm": 0.5})
    api.server.hub.devices["camera"] = api.server.Device(
        "camera", lambda: MockCamera(arrays=ARRAYS, axes=api.server.hub.machine.axes,
                                     printer_axes=lambda: corr.PrinterAxes.from_detection(
                                         api.server.hub.cal.detection)))
    return api


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
        assert s["measured_mm"] == pytest.approx([12, 10], abs=0.45)
        assert s["rotation_deg"] == pytest.approx(a.rot_deg, abs=0.5)
        assert s["pattern_source"] == "scrub tab" and s["crop"].startswith("data:image/jpeg")
    assert j["scan"]["image"].startswith("data:image/jpeg;base64,")
    assert any("Measured" in p for p in subs[0]["problems"])        # 12 x 10 array vs the 25 x 25 default


def test_patterns_set_copy_and_clear(aligned):
    aligned.post("/api/align/scan", json={"save": False})
    j = aligned.post("/api/align/pattern", json={"targets": ["S1"], "sketch": SKETCH, "name": "comb"}).json()
    assert j["patterns"]["S1"]["bridges"] == 1 and not j["scan"]["stale"]      # redrawn right away
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


def test_capture_saves_original_and_corrected(api, tmp_path, monkeypatch):
    monkeypatch.setattr(api.server, "STATE_DIR", tmp_path)
    j = api.post("/api/camera/capture", json={"check": True}).json()
    assert (tmp_path / "captures" / j["original"]).exists() and (tmp_path / "captures" / j["corrected"]).exists()
    assert j["corrected_size"] != j["photo_size"] and j["image"].startswith("data:image/jpeg;base64,")
    assert j["check"] is None                               # no checkerboard in the mock frame
    r = api.get(f"/api/camera/captures/{j['corrected']}")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    assert api.get("/api/camera/captures/nope.jpg").status_code == 404


def test_live_view_can_be_corrected(api):
    plain = api.get("/api/camera/snapshot", params={"width": 4056}).content
    corrected = api.get("/api/camera/snapshot", params={"width": 4056, "corrected": 1}).content
    import cv2
    a = cv2.imdecode(np.frombuffer(plain, np.uint8), cv2.IMREAD_COLOR)
    b = cv2.imdecode(np.frombuffer(corrected, np.uint8), cv2.IMREAD_COLOR)
    assert a.shape[:2] == (1080, 1920) and b.shape[:2] == (1362, 2323)


def test_sketcher_gets_each_substrate_straightened(aligned):
    aligned.post("/api/align/scan", json={"save": False})
    j = aligned.get("/api/align/substrate/S2/sketch").json()
    assert j["w"] == pytest.approx(12, abs=0.45) and j["h"] == pytest.approx(10, abs=0.45) and j["bridges"] == []
    import base64

    import cv2
    img = cv2.imdecode(np.frombuffer(base64.b64decode(j["image"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == round(j["w"] * j["px_per_mm"]) and img.shape[0] == round(j["h"] * j["px_per_mm"])
    # the straightened area is the array: mostly copper-coloured, not plate
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    assert ((hsv[:, :, 0] < 30) & (hsv[:, :, 1] > 60)).mean() > 0.3
    aligned.post("/api/align/pattern", json={"targets": ["S2"], "sketch": SKETCH})
    j = aligned.get("/api/align/substrate/S2/sketch").json()
    assert (j["w"], j["h"]) == (12.0, 10.0) and len(j["bridges"]) == 1
    assert aligned.get("/api/align/substrate/S9/sketch").status_code == 404


def test_teaching_on_the_corrected_view_stores_photo_pixels(api):
    api.post("/api/align/camera/delete", json={"all": True})
    c = corr.get_correction((1920, 1080))
    photo = [700.0, 400.0]
    view = c.to_corrected([photo])[0].tolist()
    mock = api.server.hub.printer._inner
    mock.homed, mock.x, mock.y = "xyz", 100.0, 120.0
    j = api.post("/api/align/camera/teach", json={"px": view, "view": "corrected",
                                                  "image_size": list(c.output_size)}).json()
    p = j["camera"]["points"][0]
    assert p["px"] == pytest.approx(photo, abs=0.01) and p["px_view"] == pytest.approx(view, abs=0.01)
    assert saved(api)["detection"]["camera_to_machine"]["photo_size"] == [1920, 1080]


def test_scan_without_camera_points_labels_arrays_for_the_sketcher(api):
    api.post("/api/align/camera/delete", json={"all": True})
    pa = corr.PrinterAxes.from_detection({"camera_to_machine": {
        "points_px": [p for p, _ in consistent_points()], "points_mm": [m for _, m in consistent_points()]}})
    api.server.hub.devices["camera"] = api.server.Device(
        "camera", lambda: MockCamera(arrays=ARRAYS, axes=api.server.hub.machine.axes, printer_axes=lambda: pa))
    j = api.post("/api/align/scan", json={"save": False}).json()
    assert j["scan"]["pixel_only"] and [s["id"] for s in j["scan"]["substrates"]] == ["S1", "S2", "S3"]
    sk = api.get("/api/align/substrate/S2/sketch").json()
    assert sk["size_known"] is False and sk["image"].startswith("data:image/jpeg")
    assert sk["w"] == 10.0 and sk["h"] == pytest.approx(10.0 * 10 / 12, abs=0.6)   # measured aspect
    api.post("/api/align/pattern", json={"targets": ["S2"], "sketch": SKETCH})
    assert saved(api)["alignment"]["patterns"]["S2"]["drawn_on"] == "camera view"
    sk = api.get("/api/align/substrate/S2/sketch").json()
    assert (sk["w"], sk["h"], len(sk["bridges"])) == (12.0, 10.0, 1)
    # once the points are taught, the same substrate keeps its name and its sketch
    teach(api, consistent_points())
    api.post("/api/align/settings", json={"size_tolerance_mm": 0.5})     # coarse mock camera
    j = api.post("/api/align/scan", json={"save": False}).json()
    s2 = next(s for s in j["scan"]["substrates"] if s["id"] == "S2")
    assert s2["pattern_source"] == "own" and not s2["warnings"]
    assert s2["center_mm"] == pytest.approx([216.0 + 6 * np.cos(np.radians(1.5)) + 5 * np.sin(np.radians(1.5)),
                                            236.0 + 6 * np.sin(np.radians(1.5)) - 5 * np.cos(np.radians(1.5))], abs=0.2)


def test_corners_can_be_set_added_and_removed_by_hand(aligned):
    j = aligned.post("/api/align/scan", json={"save": False}).json()
    assert len(j["scan"]["substrates"]) == 3
    aligned.post("/api/align/pattern", json={"targets": ["S3"], "sketch": SKETCH, "name": "third"})
    v = aligned.get("/api/align/view", params={"sid": "S2"}).json()
    assert len(v["corners"]) == 4 and v["image"].startswith("data:image/jpeg")
    to_view = lambda pts: [[x / v["scale"] + v["offset"][0], y / v["scale"] + v["offset"][1]] for x, y in pts]  # noqa: E731
    before = next(s for s in j["scan"]["substrates"] if s["id"] == "S2")["center_mm"]
    moved = [[x + 2, y] for x, y in v["corners"]]            # shift the outline 2 picture px right
    j = aligned.post("/api/align/substrate/S2/corners", json={"corners": to_view(moved)}).json()
    s2 = next(s for s in j["scan"]["substrates"] if s["id"] == "S2")
    assert s2["manual"] and s2["center_mm"][0] > before[0] + 0.05
    # add one by hand: a copy of S1's corners moved well clear
    w = aligned.get("/api/align/view").json()
    s1 = aligned.get("/api/align/view", params={"sid": "S1"}).json()
    full = [[(x / s1["scale"] + s1["offset"][0] + 150) * w["scale"], (y / s1["scale"] + s1["offset"][1]) * w["scale"]]
            for x, y in s1["corners"]]
    j = aligned.post("/api/align/substrate/add", json={"corners": [[x / w["scale"], y / w["scale"]] for x, y in full]}).json()
    assert j["added"] == "S4" and len(j["scan"]["substrates"]) == 4
    assert aligned.post("/api/align/substrate/add", json={"corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}).status_code == 400
    # remove S2: S3 becomes S2 and keeps its sketch
    j = aligned.post("/api/align/substrate/S2/remove").json()
    assert [s["id"] for s in j["scan"]["substrates"]] == ["S1", "S2", "S3"]
    assert j["patterns"]["S2"]["name"] == "third" and "S3" not in j["patterns"]
