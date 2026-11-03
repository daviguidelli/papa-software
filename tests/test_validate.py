"""The validation page's backend: Z floor, locked Z homing, servo lock, load cell
position and probe, slide corner and trace. All against mocks."""
import importlib
import shutil

import pytest
import yaml
from fastapi.testclient import TestClient

from papa import config as cfgmod
from papa.ops.guard import Guard, GuardedPrinter, ZFloorError
from papa.sim.mocks import MockMoonraker


@pytest.fixture
def api(monkeypatch, tmp_path):
    cfg = tmp_path / "config"
    shutil.copytree(cfgmod.REPO_ROOT / "config", cfg)
    (cfg / "calibration.yaml").unlink(missing_ok=True)
    monkeypatch.setattr(cfgmod, "CONFIG_DIR", cfg)
    monkeypatch.setenv("PAPA_SIM", "1")
    monkeypatch.delenv("PAPA_REAL", raising=False)
    monkeypatch.delenv("PAPA_ALLOW_Z_HOME", raising=False)
    server = importlib.reload(importlib.import_module("apps.server"))
    server.hub.guard = Guard(tmp_path / "z_floor.json")
    client = TestClient(server.app)
    client.cfg = cfg
    client.server = server
    return client


def saved(client):
    return yaml.safe_load((client.cfg / "calibration.yaml").read_text())


def test_guard_blocks_down_until_a_floor_exists(tmp_path):
    p = GuardedPrinter(MockMoonraker(z=50.0), Guard(tmp_path / "f.json"))
    p.move(z=60.0)
    p.move_relative(dz=5.0)
    with pytest.raises(ZFloorError, match="Set the Z floor"):
        p.move(z=40.0)
    with pytest.raises(ZFloorError):
        p.move_relative(dz=-0.05)
    p.guard.set_floor(30.0)
    p.move(z=35.0)
    with pytest.raises(ZFloorError, match="below the floor"):
        p.move(z=29.0)
    with pytest.raises(ZFloorError):
        p.move_relative(dz=-6.0)


def test_floor_persists(tmp_path):
    Guard(tmp_path / "f.json").set_floor(12.5)
    assert Guard(tmp_path / "f.json").floor == 12.5


def test_jog_xy_never_sends_z_and_z_down_needs_a_floor(api):
    r = api.post("/api/machine/jog", json={"dx": 5, "speed_mm_s": 20})
    assert r.status_code == 200
    z0 = r.json()["z"]
    r = api.post("/api/machine/jog", json={"dz": -1, "speed_mm_s": 5})
    assert r.status_code == 400 and "Z floor" in r.json()["error"]
    assert api.post("/api/guard/floor", json={"z": z0 - 3}).json()["floor"] == z0 - 3
    assert api.post("/api/machine/jog", json={"dz": -1}).status_code == 200
    assert api.post("/api/machine/jog", json={"dz": -10}).status_code == 400
    assert api.post("/api/machine/jog", json={"dz": 2}).status_code == 200


def test_floor_outside_limits_is_refused(api):
    assert api.post("/api/guard/floor", json={"z": -5}).status_code == 400
    assert api.post("/api/guard/floor", json={"z": 999}).status_code == 400


def test_z_homing_is_locked_by_default(api, monkeypatch):
    assert api.post("/api/machine/home", json={"axes": "Z"}).status_code == 403
    assert api.post("/api/machine/home", json={}).status_code == 403
    assert api.post("/api/machine/home", json={"axes": "X Y"}).status_code == 200
    monkeypatch.setenv("PAPA_ALLOW_Z_HOME", "1")
    assert api.post("/api/machine/home", json={"axes": "Z"}).status_code == 200


def test_servo_lock_writes_calibration_and_survives_reload(api):
    r = api.post("/api/gripper/pulse", json={"us": 1450})
    assert r.status_code == 200 and r.json()["pulse_us"] == 1450
    assert api.post("/api/gripper/pulse", json={"us": 100}).status_code == 400
    assert api.post("/api/gripper/lock", json={"which": "open", "us": 1200}).status_code == 200
    r = api.post("/api/gripper/lock", json={"which": "closed", "us": 1800})
    assert r.json()["locked"] == {"open_us": 1200, "closed_us": 1800}
    assert saved(api)["gripper"] == {"open_us": 1200, "closed_us": 1800, "open_set": True, "closed_set": True}
    api.post("/api/gripper/close")
    assert api.get("/api/gripper").json()["pulse_us"] == 1800
    assert api.post("/api/gripper/lock", json={"which": "open", "us": 1800}).status_code == 400
    assert api.post("/api/gripper/lock", json={"which": "sideways", "us": 1500}).status_code == 400


def test_load_cell_xy_needs_homing_then_saves(api):
    assert api.post("/api/loadcell/position", json={"capture_xy": True}).status_code == 400
    api.post("/api/machine/home", json={"axes": "XY"})
    api.post("/api/machine/jog", json={"dx": 30, "dy": 40, "speed_mm_s": 50})
    r = api.post("/api/loadcell/position", json={"capture_xy": True, "z_safe": 16})
    assert r.status_code == 200
    lc = saved(api)["load_cell"]
    assert lc["xy_set"] is True and lc["z_safe"] == 16 and lc["z_safe_set"] is True
    assert (lc["x"], lc["y"]) == (30.0, 40.0)


def test_probe_refuses_until_ready_then_finds_contact(api, monkeypatch):
    assert api.post("/api/loadcell/probe").status_code == 400          # no floor
    api.post("/api/guard/floor", json={"z": 5})
    assert api.post("/api/loadcell/probe").status_code == 400          # not captured
    monkeypatch.setenv("PAPA_ALLOW_Z_HOME", "1")
    api.post("/api/machine/home", json={"axes": "XYZ"})
    api.post("/api/machine/jog", json={"dx": 30, "dy": 40, "speed_mm_s": 50})
    api.post("/api/loadcell/position", json={"capture_xy": True, "z_safe": 16})
    r = api.post("/api/loadcell/probe")
    assert r.status_code == 200, r.text
    assert 11.5 < r.json()["contact_z"] < 12.0                          # mock surface is at 12


def test_probe_is_stopped_by_the_floor(api, monkeypatch):
    monkeypatch.setenv("PAPA_ALLOW_Z_HOME", "1")
    api.post("/api/machine/home", json={"axes": "XYZ"})
    api.post("/api/loadcell/position", json={"capture_xy": True, "z_safe": 16})
    api.post("/api/guard/floor", json={"z": 13})                        # above the surface
    r = api.post("/api/loadcell/probe")
    assert r.status_code == 400 and "below the floor" in r.json()["error"]


def test_surface_offset_saves_and_rejects_nonsense(api):
    assert api.post("/api/calibration/surface_offset", json={"offset_mm": -7.25}).status_code == 200
    assert saved(api)["slide_surface_offset_mm"] == -7.25
    assert api.post("/api/calibration/surface_offset", json={"offset_mm": 500}).status_code == 400


def test_slide_size_corner_and_trace(api):
    bad = api.post("/api/slide/size", json={"width_mm": 25, "length_mm": 75, "pattern_size_mm": 30})
    assert bad.status_code == 400
    ok = api.post("/api/slide/size", json={"width_mm": 26, "length_mm": 76, "pattern_size_mm": 20})
    assert ok.status_code == 200 and ok.json()["substrate"]["pattern_size_mm"] == 20
    assert api.post("/api/slide/corner", json={"id": 2}).status_code == 400   # not homed
    api.post("/api/machine/home", json={"axes": "XY"})
    api.post("/api/machine/jog", json={"dx": 60, "dy": 200, "speed_mm_s": 100})
    s = api.post("/api/slide/corner", json={"id": 2}).json()
    assert s["slot"]["calibrated"] is True
    # machine.yaml maps local +x to +X and local +y to -Y
    assert s["slot"]["pattern"] == [[60, 200], [80, 200], [80, 180], [60, 180]]
    assert s["slot"]["outline"][2] == [86, 124]
    assert api.post("/api/slide/corner", json={"id": 99}).status_code == 400
    assert api.post("/api/slide/trace", json={"id": 3}).status_code == 400    # not captured
    r = api.post("/api/slide/trace", json={"id": 2, "which": "pattern", "speed_mm_s": 30})
    assert r.status_code == 200
    log = api.server.hub.printer._inner.log
    assert sum(1 for l in log if l.startswith("G1") and "X80.0000 Y180.0000" in l) == 1
    assert (r.json()["x"], r.json()["y"]) == (60, 200)


def test_trace_refuses_a_square_that_leaves_the_machine(api):
    api.post("/api/machine/home", json={"axes": "XY"})
    api.post("/api/machine/jog", json={"dx": 340, "dy": 10, "speed_mm_s": 100})
    api.post("/api/slide/corner", json={"id": 1})
    r = api.post("/api/slide/trace", json={"id": 1, "which": "outline"})
    assert r.status_code == 400 and "outside the machine limits" in r.json()["error"]


def test_saving_never_turns_sim_values_into_the_real_calibration(api):
    api.post("/api/calibration/surface_offset", json={"offset_mm": -1})
    assert saved(api)["swab_rack"]["swabs"][0]["x"] == 112.40          # example file, not sim's 112.4/30.0 set
    assert len(saved(api)["slide_slots"]["slots"]) == 1


# ---------- live camera ----------

def test_snapshot_is_a_jpeg_and_respects_width(api):
    import cv2
    import numpy as np

    r = api.get("/api/camera/snapshot?width=640")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 640 and img.shape[0] == 360          # 16:9 kept


def test_stream_is_multipart_and_carries_several_frames(api):
    with api.stream("GET", "/api/camera/stream?width=320&fps=15&frames=3") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace; boundary=frame")
        body = b"".join(r.iter_bytes())
    assert body.count(b"--frame") == 3 and body.count(b"\xff\xd8") >= 3


def test_stream_slot_is_released_after_the_client_finishes(api):
    for _ in range(6):                                           # more than MAX_STREAMS
        with api.stream("GET", "/api/camera/stream?width=160&frames=1") as r:
            assert r.status_code == 200
            b"".join(r.iter_bytes())


def test_missing_camera_is_a_clean_503(api):
    def boom():
        raise RuntimeError("no libcamera on this machine")
    api.server.hub.devices["camera"] = api.server.Device("camera", boom)
    r = api.get("/api/camera/stream")
    assert r.status_code == 503 and "no libcamera" in r.json()["error"]
    assert api.get("/api/camera/snapshot").status_code == 503


def test_remote_camera_host_down_is_a_502_not_a_hang(api, monkeypatch):
    monkeypatch.setenv("PAPA_CAMERA_URL", "http://127.0.0.1:9")
    for path in ("/api/camera/snapshot", "/api/camera/stream"):
        r = api.get(path)
        assert r.status_code == 502 and "unreachable" in r.json()["error"]
    assert api.get("/api/devices").json()["devices"]["camera"]["remote"] == "http://127.0.0.1:9"


def test_locked_camera_passes_attributes_through():
    from papa.hw.camera import LockedCamera
    from papa.sim.mocks import MockCamera

    cam = LockedCamera(MockCamera())
    cam.slide_slots = (1, 2)
    assert cam._inner.slide_slots == (1, 2) and cam.frame().shape == (1080, 1920, 3)


# ---------- hold-to-jog, fine servo, probe settings ----------

import time as _time


def _home(api):
    api.post("/api/machine/home", json={"axes": "X Y"})
    api.server.hub.printer.move(z=100.0)


def _hold_for(api, seconds, body):
    end = _time.monotonic() + seconds
    while _time.monotonic() < end:
        r = api.post("/api/machine/hold", json=body)
        if r.status_code == 400:                         # the hold ended itself (limit, floor)
            break
        assert r.status_code == 200, r.text
        _time.sleep(0.1)


def test_hold_moves_while_beating_and_stops_without_beats(api):
    _home(api)
    x0 = api.get("/api/machine/position").json()["x"]
    _hold_for(api, 0.8, {"dx": 1, "speed_mm_s": 20})
    _time.sleep(0.6)                                     # dead-man window passes
    assert not api.get("/api/machine/hold").json()["moving"]
    x1 = api.get("/api/machine/position").json()["x"]
    moved = x1 - x0
    assert 6 < moved < 30                                # about 0.8 s at 20 mm/s plus look-ahead
    _time.sleep(0.3)
    assert api.get("/api/machine/position").json()["x"] == x1


def test_hold_stop_endpoint_stops_immediately(api):
    _home(api)
    api.post("/api/machine/hold", json={"dx": -1, "speed_mm_s": 10})
    _time.sleep(0.2)
    r = api.post("/api/machine/hold/stop")
    assert r.status_code == 200 and r.json()["moving"] is False


def test_hold_z_down_needs_floor_and_stops_at_it(api):
    _home(api)
    api.post("/api/machine/hold", json={"dz": -1, "speed_mm_s": 5})
    _time.sleep(0.3)
    assert api.get("/api/machine/hold").json()["why"] == "Z floor is not set"
    r = api.post("/api/machine/hold", json={"dz": -1, "speed_mm_s": 5})
    assert r.status_code == 400 and "Z floor is not set" in r.json()["error"]    # sticky until released
    assert api.post("/api/machine/hold/stop").json()["why"] == "Z floor is not set"
    api.post("/api/guard/floor", json={"z": 99.0})
    _hold_for(api, 1.0, {"dz": -1, "speed_mm_s": 5})
    _time.sleep(0.1)
    z = api.get("/api/machine/position").json()["z"]
    assert z >= 99.0 - 1e-6
    assert "floor" in api.get("/api/machine/hold").json()["why"]


def test_hold_rejects_diagonals_and_silly_speeds(api):
    _home(api)
    assert api.post("/api/machine/hold", json={"dx": 1, "dy": 1, "speed_mm_s": 5}).status_code == 400
    assert api.post("/api/machine/hold", json={"dz": -1, "speed_mm_s": 50}).status_code == 400
    assert api.post("/api/machine/hold", json={"dx": 1, "speed_mm_s": 0}).status_code == 400


def test_step_jog_is_refused_while_a_button_is_held(api):
    _home(api)
    api.post("/api/machine/hold", json={"dx": 1, "speed_mm_s": 10})
    r = api.post("/api/machine/jog", json={"dx": 1})
    api.post("/api/machine/hold/stop")
    assert r.status_code == 409


def test_servo_pulse_must_sit_inside_the_window(api):
    assert api.get("/api/gripper").json()["window"] == [1300, 1700]
    assert api.post("/api/gripper/pulse", json={"us": 1000}).status_code == 400
    assert api.post("/api/gripper/pulse", json={"us": 1501, "slew_us_s": 50}).status_code == 200
    assert api.post("/api/gripper/window", json={"lo": 900, "hi": 2100}).status_code == 200
    assert api.post("/api/gripper/pulse", json={"us": 1000}).status_code == 200
    assert api.post("/api/gripper/window", json={"lo": 100, "hi": 2100}).status_code == 400
    assert api.post("/api/gripper/pulse", json={"us": 1500, "slew_us_s": 5000}).status_code == 400


def test_lock_marks_each_state_and_persists_flags(api):
    api.post("/api/gripper/lock", json={"which": "open", "us": 1400})
    g = saved(api)["gripper"]
    assert g["open_us"] == 1400 and g["open_set"] is True and not g.get("closed_set")


def test_ramp_points_are_small_whole_steps_ending_on_target():
    from papa.hw.gripper import ramp_points

    pts = ramp_points(1500, 1510, 50)
    assert pts[-1][0] == 1510 and all(abs(b[0] - a[0]) <= 1 for a, b in zip(pts, pts[1:]))
    assert ramp_points(1500, 1500, 50) == []
    down = ramp_points(1600, 1400, 100)
    assert down[-1][0] == 1400 and down[0][0] < 1600


def test_real_servo_refuses_open_and_close_until_both_locked():
    from papa.hw.gripper import GripperNotCalibrated, ServoGripper

    g = ServoGripper.__new__(ServoGripper)
    g.calibrated = False
    with pytest.raises(GripperNotCalibrated):
        g.open()
    with pytest.raises(GripperNotCalibrated):
        g.close()


def _ready_probe(api, z_start=100.0):
    _home(api)
    api.server.hub.printer._inner.homed = "xyz"
    api.post("/api/guard/floor", json={"z": 60.0})
    api.post("/api/loadcell/position", json={"capture_xy": True})
    api.post("/api/loadcell/position", json={"z_safe": z_start})


def test_probe_goes_to_start_z_before_descending_and_uses_given_speed(api):
    _ready_probe(api)
    mock = api.server.hub.printer._inner
    mock.log.clear()
    r = api.post("/api/loadcell/probe", json={"start_z": 90.0, "speed_mm_s": 0.5, "step_mm": 0.1})
    assert r.status_code in (200, 409), r.text
    g = [l for l in mock.log if l.startswith("G1")]
    first_z = next(l for l in g if " Z" in l and "F" in l)
    assert "Z90.0000" in first_z and "F600" in first_z           # 10 mm/s, never travel speed
    assert any(l.startswith("G1 Z-0.1000 F30") for l in g)       # 0.5 mm/s descent steps


def test_probe_rejects_unsafe_overrides(api):
    _ready_probe(api)
    assert api.post("/api/loadcell/probe", json={"start_z": 50.0}).status_code == 400   # under floor
    assert api.post("/api/loadcell/probe", json={"speed_mm_s": 30}).status_code == 400
    assert api.post("/api/loadcell/probe", json={"step_mm": 5}).status_code == 400
    assert api.post("/api/loadcell/probe", json={"start_z": 9999}).status_code == 400
