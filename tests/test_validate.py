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
    assert saved(api)["gripper"] == {"open_us": 1200, "closed_us": 1800}
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
