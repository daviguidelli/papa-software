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


def _data_url(path):
    import base64
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def test_reference_flow_moves_the_gcode_onto_each_substrate(aligned, monkeypatch, tmp_path):
    a = aligned
    monkeypatch.setattr(a.server, "STATE_DIR", tmp_path)
    j = a.get("/api/reference").json()
    assert not j["ready"] and j["photo"] is None
    j = a.post("/api/reference/photo", json={"image": _data_url("tests/data/reference_substrate.png")}).json()
    assert j["array_found"] and j["photo"].startswith("data:image/jpeg")
    xs = [p[0] for p in j["array"]]
    assert min(xs) == pytest.approx(321, abs=6) and max(xs) == pytest.approx(662, abs=6)
    j = a.post("/api/reference/corners", json={"corners": [[663, 806], [323, 463], [663, 463], [323, 806]]}).json()
    assert j["red_corner"] == 2 and j["corners"][0] == [323, 463]          # sorted TL, TR, BR, BL
    a.post("/api/reference/placement", json={"x_min": 162.5, "y_min": 162.1, "size": 25})
    g = "G21\nG90\nG0 Z10\nG0 X162.65 Y174.6\nG1 Z4.5\nG1 X187.35 Y174.6 F600\nG0 Z10\n"
    r = a.post("/api/reference/gcode", json={"text": "G20\nG1 X1", "name": "bad.gcode"})
    assert r.status_code == 400
    j = a.post("/api/reference/gcode", json={"text": g, "name": "clear.gcode"}).json()
    assert j["ready"] and j["gcode"]["moves"] >= 1 and j["gcode"]["x"] == [162.65, 187.35]

    subs = a.post("/api/align/scan", json={"save": False}).json()["scan"]["substrates"]
    for s, arr in zip(subs, ARRAYS):
        m = s["mount"]
        assert m and m["source"] == "estimated from the array" and m["gcode_ready"]
        assert m["rotation_deg"] == pytest.approx(arr.rot_deg, abs=0.7)    # reference array is ~square to its glass
    r = a.get("/api/align/substrate/S2/gcode")
    assert r.status_code == 200 and "S2_clear.gcode" in r.headers["content-disposition"]
    body = r.text
    assert "; PAPA: reference toolpath moved onto substrate S2" in body and "G1 Z4.5" in body

    # glass corners by hand override the estimate and survive a reference change
    v = a.get("/api/align/view", params={"sid": "S1", "what": "mount"}).json()
    cv = (np.asarray(v["corners"]) / v["scale"] + v["offset"]).tolist()
    j = a.post("/api/align/substrate/S1/mount", json={"corners": cv}).json()
    s1 = next(s for s in j["scan"]["substrates"] if s["id"] == "S1")
    assert s1["mount"]["source"] == "set by hand"
    j = a.post("/api/reference/array", json={"corners": [[320, 505], [662, 505], [662, 775], [320, 775]]}).json()
    assert j["array"][0] == [320, 505]
    s1 = next(s for s in a.get("/api/align").json()["scan"]["substrates"] if s["id"] == "S1")
    assert s1["mount"]["source"] == "set by hand"


def test_reference_sketch_lands_on_each_glass(aligned, monkeypatch, tmp_path):
    import json

    from papa.ops.geometry import to_machine

    a = aligned
    monkeypatch.setattr(a.server, "STATE_DIR", tmp_path)
    a.post("/api/reference/photo", json={"image": _data_url("tests/data/reference_substrate.png")})
    a.post("/api/reference/corners", json={"corners": [[323, 463], [663, 463], [663, 806], [323, 806]]})
    a.post("/api/reference/placement", json={"x_min": 162.5, "y_min": 162.1, "size": 25})
    sk = json.load(open("tests/data/reference_bridges.json"))
    assert a.post("/api/reference/sketch", json={"sketch": {"bridges": []}}).status_code == 400
    j = a.post("/api/reference/sketch", json={"sketch": sk, "name": "bridges_2026-11-16"}).json()
    assert j["ready"] and j["sketch"]["bridges"] == 8 and j["sketch"]["warning"] is None

    subs = a.post("/api/align/scan", json={"save": False}).json()["scan"]["substrates"]
    assert all(s["glass_placed"] and s["pattern_name"].startswith("Reference sketch") for s in subs)
    assert all(s["pattern_mm"] == [25, 25] for s in subs)
    sc = a.server.hub.align_scan
    axes = a.server.hub.machine.axes
    for i, s in enumerate(subs):
        mt = sc["mount_info"][i]["match"]
        for b in sk["bridges"]:                       # reference bridge, moved like the glass
            ref = np.array([162.5 + b["cx"], 162.1 + 25 - b["cy"]])
            want = mt.R @ ref + mt.t
            got = to_machine(axes, *s["origin_mm"], b["cx"], b["cy"], s["rotation_deg"])
            assert got == pytest.approx(want.tolist(), abs=1e-3)       # API rounds to 4 dp

    r = a.get("/api/align/substrate/S1/gcode")
    assert r.status_code == 400 and "contact Z" in r.json()["error"]
    r = a.get("/api/align/substrate/S1/gcode", params={"z": 4.5})
    assert r.status_code == 200 and "S1_scrub.gcode" in r.headers["content-disposition"]
    assert "modified for this glass" in r.text and "G1 Z4.500" in r.text

    # the known, unmodified toolpath, and the comparison against it
    assert a.get("/api/reference/gcode").status_code == 400            # no contact Z yet
    r = a.get("/api/reference/gcode", params={"z": 4.5})
    assert r.status_code == 200 and "REFERENCE toolpath" in r.text and "G1 Z4.500" in r.text
    c = a.get("/api/align/substrate/S2/compare").json()
    mt = sc["mount_info"][1]["match"]
    assert len(c["bridges"]) == 8 and c["ref_segs"] and len(c["ref_segs"]) == len(c["new_segs"])
    for b in c["bridges"]:
        assert b["new"] == pytest.approx((mt.R @ np.array(b["ref"]) + mt.t).tolist(), abs=2e-3)
    og = c["on_glass"]                                     # glass estimated from the array: reference layout
    assert og is not None and og["measured"] is False and og["swapped"] == []
    assert og["rotation_deg"] == pytest.approx(0, abs=0.05) and og["offset_mm"] == pytest.approx([0, 0], abs=0.05)
    assert "Array on its glass" in a.get("/api/align/substrate/S2/report", params={"z": 4.5}).text
    b1 = next(b for b in c["bridges"] if b["id"] == "B1")
    assert b1["ref"] == pytest.approx([162.5 + 12.29, 162.1 + 25 - 0.916], abs=2e-3)
    for (x0, y0, x1, y1), (u0, v0, u1, v1) in zip(c["ref_segs"], c["new_segs"]):   # same path, moved like the glass
        assert [u0, v0] == pytest.approx((mt.R @ np.array([x0, y0]) + mt.t).tolist(), abs=5e-3)
    g = a.get("/api/align/substrate/S2/gcode", params={"z": 4.5}).text
    assert "REFERENCE toolpath, modified for this glass" in g and ";   B8: X" in g

    # a G-code file loaded as well does not replace the sketch
    a.post("/api/reference/gcode", json={"text": "G21\nG90\nG0 X170 Y170\nG1 Z4.5\nG1 X180 F600\n", "name": "other.gcode"})
    s2 = next(s for s in a.get("/api/align").json()["scan"]["substrates"] if s["id"] == "S2")
    assert s2["gcode_from"] == "plan"
    g = a.get("/api/align/substrate/S2/gcode", params={"z": 4.5})
    assert "S2_scrub.gcode" in g.headers["content-disposition"] and "modified for this glass" in g.text
    a.post("/api/reference/gcode", json={"clear": True})

    # this substrate's own geometry, drawn on its glass from the reference bridges
    j = a.get("/api/align/substrate/S3/sketch").json()
    assert j["frame"] == "glass" and j["from_reference"] and (j["w"], j["h"]) == (25, 25) and len(j["bridges"]) == 8
    edited = [dict(b) for b in j["bridges"]]
    edited[1]["cx"] += 0.4                                   # B2 moved 0.4 mm along x
    edited[4]["l"] = 1.5                                     # B5 longer
    del edited[6]                                            # B7 gone
    edited.append({"id": "B9", "cx": 5.0, "cy": 20.0, "l": 1, "w": 1})
    for i, b in enumerate(edited):                           # the sketcher renumbers on save
        b["id"] = f"B{i + 1}"
    a.post("/api/align/pattern", json={"targets": ["S3"], "name": "S3 edit", "frame": "glass",
                                       "sketch": {"substrate": {"w": 25, "h": 25}, "bridges": edited}})
    s3 = next(s for s in a.get("/api/align").json()["scan"]["substrates"] if s["id"] == "S3")
    assert s3["glass_placed"] and s3["pattern_source"] == "glass" and s3["pattern_name"] == "S3 edit (on the glass)"
    c = a.get("/api/align/substrate/S3/compare").json()
    assert c["source"] == "glass" and c["edited"] == 2 and c["added"] == 1 and [r["id"] for r in c["removed"]] == ["B7"]
    by = {b.get("ref_id"): b for b in c["bridges"]}
    assert by["B2"]["edit"] == pytest.approx([0.4, 0.0], abs=1e-6) and by["B5"]["resized"] == pytest.approx([0.5, 0.0])
    assert by["B8"]["id"] == "B7" and by["B8"]["edit_mm"] == pytest.approx(0, abs=1e-6)   # renumbered, still paired
    mt = a.server.hub.align_scan["mount_info"][2]["match"]
    b2_moved_ref = (mt.R @ np.array([162.5 + 12.247 + 0.4, 162.1 + 25 - 4.442]) + mt.t).tolist()
    assert by["B2"]["new"] == pytest.approx(b2_moved_ref, abs=2e-3)         # glass move + the 0.4 mm edit
    g = a.get("/api/align/substrate/S3/gcode", params={"z": 4.5}).text
    assert "own bridge edits" in g and "2 changed" in g and "B7: REMOVED" in g and "NEW (not in the reference)" in g

    # the stand-alone comparison report
    r = a.get("/api/align/substrate/S3/report", params={"z": 4.5})
    assert r.status_code == 200 and "PAPA_S3_vs_reference.html" in r.headers["content-disposition"]
    t = r.text
    assert t.startswith("<!DOCTYPE html>") and "<svg" in t and "REFERENCE_scrub.gcode" in t and "S3_scrub.gcode" in t
    assert t.count("<tr>") >= 10 and "removed" in t and "data:image/jpeg;base64," in t

    # a substrate's own sketch still wins over the reference
    a.post("/api/align/pattern", json={"targets": ["S2"], "sketch": SKETCH, "name": "own"})
    s2 = next(s for s in a.get("/api/align").json()["scan"]["substrates"] if s["id"] == "S2")
    assert not s2["glass_placed"] and s2["pattern_name"] == "own"


def test_scan_photo_download_and_code_version(aligned):
    j = aligned.post("/api/align/scan", json={"save": False}).json()
    assert j["scan"]["timing"]["code"]
    r = aligned.get("/api/align/photo")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"


def test_last_scan_survives_a_restart(aligned):
    import apps.server as srv

    aligned.post("/api/align/scan", json={"save": False})
    n = len(srv.hub.align_scan["views"])
    assert n and (srv._last_scan_dir() / "state.json").is_file()
    srv.hub.align_scan = None
    srv._restore_scan()
    assert srv.hub.align_scan is not None and len(srv.hub.align_scan["views"]) == n
    assert aligned.get("/api/align").json()["scan"]["substrates"]


def test_slots_number_each_holder_position(aligned):
    """A 1 x 4 slot grid over the whole picture: each array gets the slot its
    centre falls in, S1, S2, ... follow the slots, and the numbers reach the
    table, the picture, the G-code header and survive a new scan."""
    a = aligned
    a.post("/api/align/slots", json={"clear": True})
    j = a.post("/api/align/scan", json={"save": False}).json()
    assert all(s.get("slot") is None for s in j["scan"]["substrates"])
    srv, sc = a.server, a.server.hub.align_scan
    cen = np.array([np.asarray(v).mean(0) for v in sc["views"]])
    (x0, y0), (x1, y1) = cen.min(0) - 40, cen.max(0) + 40       # a grid just around the arrays
    box = srv._view_to_drawn(sc, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    dh, dw = sc["drawn"].shape[:2]
    pts = (box / [dw, dh]).tolist()
    r = a.post("/api/align/slots", json={"rows": 1, "cols": 4, "points": pts})
    assert r.status_code == 200, r.text
    assert r.json()["slots"]["rows"] == 1 and len(r.json()["slots"]["corners"]) == 4
    j = a.post("/api/align/scan", json={"save": False}).json()
    subs = j["scan"]["substrates"]
    slots = [s["slot"] for s in subs]
    assert all(isinstance(k, int) and 1 <= k <= 4 for k in slots)
    assert slots == sorted(slots)                                 # S1, S2, ... in slot order
    assert len(set(slots)) > 1                                     # arrays in different places differ
    assert slots == srv._slot_of_views(srv.hub.align_scan["views"])
    g = a.get(f"/api/align/substrate/{subs[0]['id']}/gcode", params={"z": 4.5})
    if g.status_code == 200:
        assert f"slot {slots[0]}" in g.text.splitlines()[0] or any(f"slot {slots[0]}" in l for l in g.text.splitlines()[:3])
    # changing only rows/cols keeps the clicked corners
    r = a.post("/api/align/slots", json={"rows": 2, "cols": 4})
    assert r.json()["slots"]["rows"] == 2
    a.post("/api/align/slots", json={"clear": True})
    assert a.get("/api/align/slots").json()["slots"] is None


def test_slots_need_four_corners(aligned):
    aligned.post("/api/align/scan", json={"save": False})
    r = aligned.post("/api/align/slots", json={"rows": 3, "cols": 3, "points": [[0, 0], [1, 1]]})
    assert r.status_code == 400
    aligned.post("/api/align/slots", json={"clear": True})
    assert aligned.post("/api/align/slots", json={"rows": 3, "cols": 3}).status_code == 400


def test_template_gcode_repeated_on_other_slots(aligned):
    """The slot map gives every slot's centre and the X/Y step; one substrate's
    G-code repeated on other slots is the same moves shifted by the slot step."""
    import re

    a = aligned
    a.post("/api/align/scan", json={"save": False})
    srv, sc = a.server, a.server.hub.align_scan
    cen = np.array([np.asarray(v).mean(0) for v in sc["views"]])
    (x0, y0), (x1, y1) = cen.min(0) - 40, cen.max(0) + 40
    box = srv._view_to_drawn(sc, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    dh, dw = sc["drawn"].shape[:2]
    j = a.post("/api/align/slots", json={"rows": 2, "cols": 3, "points": (box / [dw, dh]).tolist()}).json()
    sm = j["slot_map"]
    assert len(sm["cells"]) == 6 and sm["col_step_mm"] and sm["row_step_mm"]
    assert any(c["sid"] for c in sm["cells"])
    src = next(c for c in sm["cells"] if c["sid"])
    other = next(c for c in sm["cells"] if c["slot"] != src["slot"])
    one = a.get(f"/api/align/substrate/{src['sid']}/gcode", params={"z": 4.5})
    assert one.status_code == 200, one.text
    r = a.get("/api/align/slots/gcode", params={"source": src["sid"], "targets": f"{src['slot']},{other['slot']}", "z": 4.5})
    assert r.status_code == 200, r.text
    assert f"slot {other['slot']}: template shifted" in r.text

    def xs(text):
        return [float(m) for m in re.findall(r"^G[01][^;\n]*X(-?[\d.]+)", text, re.M)]

    blocks = r.text.split("; ===== slot ")
    x_same, x_moved = xs(blocks[1]), xs(blocks[2])
    dx = other["centre_mm"][0] - src["centre_mm"][0]
    assert x_same and len(x_same) == len(x_moved)
    assert np.allclose(np.array(x_moved) - np.array(x_same), dx, atol=0.002)
    assert a.get("/api/align/slots/gcode", params={"source": src["sid"], "targets": "99", "z": 4.5}).status_code == 400
    a.post("/api/align/slots", json={"clear": True})


def test_slot_corners_any_click_order(aligned):
    """Corners clicked in reading order (TL, TR, BL, BR) or starting anywhere
    give the same grid as top-left, top-right, bottom-right, bottom-left; an
    already saved twisted grid is untwisted."""
    import json

    a = aligned
    a.post("/api/align/scan", json={"save": False})
    tl, tr, br, bl = [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]
    ref = a.post("/api/align/slots", json={"rows": 3, "cols": 3, "points": [tl, tr, br, bl]}).json()["slots"]["corners"]
    for order in ([tl, tr, bl, br], [br, bl, tl, tr], [tr, tl, bl, br]):
        got = a.post("/api/align/slots", json={"rows": 3, "cols": 3, "points": order}).json()["slots"]["corners"]
        assert np.allclose(got, ref, atol=0.05), order
    srv = a.server
    c = ref
    srv._slots_file().write_text(json.dumps({"rows": 3, "cols": 3, "corners": [c[0], c[1], c[3], c[2]]}))
    assert np.allclose(srv._slots()["corners"], ref, atol=0.05)
    a.post("/api/align/slots", json={"clear": True})
