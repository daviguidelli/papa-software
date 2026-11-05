"""Autonomous grip and dispose: taught positions, the safe-Z-first move order,
refusals, and stop. All against mocks."""
import time

import pytest

from tests.test_validate import api, saved  # noqa: F401  (fixture)


def wait_done(client, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get("/api/auto/status").json()
        if not s["running"]:
            return s
        time.sleep(0.02)
    raise AssertionError("auto action did not finish")


def moves(client):
    printer = client.server.hub.printer._inner
    return [ln for ln in printer.log if ln.upper().startswith(("G0 ", "G1 "))]


@pytest.fixture
def taught(api):
    api.post("/api/machine/home", json={"axes": "XY"})
    api.post("/api/machine/set_z", json={"z": 100.0, "up_mm": 100.0})
    api.post("/api/guard/floor", json={"z": 20.0})
    api.post("/api/machine/move", json={"x": 40, "y": 60, "z": 40, "speed_mm_s": 100})
    api.post("/api/auto/teach", json={"target": "pickup"})
    api.post("/api/machine/move", json={"x": 300, "y": 50, "z": 45, "speed_mm_s": 100})
    api.post("/api/auto/teach", json={"target": "bin"})
    api.post("/api/machine/move", json={"z": 90, "speed_mm_s": 100})
    api.post("/api/auto/teach", json={"target": "safe_z"})
    api.post("/api/auto/settings", json={"grip_wait_s": 0.0, "release_wait_s": 0.0})
    return api


def test_nothing_runs_until_positions_are_taught(api):
    r = api.post("/api/auto/grip")
    assert r.status_code == 400 and "safe Z" in r.json()["error"]
    info = api.get("/api/auto").json()
    assert info["ready"] == {"grip": False, "dispose": False}
    assert info["centre"] == {"x": 175.0, "y": 175.0}


def test_teach_needs_xy_homed(api):
    r = api.post("/api/auto/teach", json={"target": "pickup"})
    assert r.status_code == 400 and "Home X and Y" in r.json()["error"]
    assert api.post("/api/auto/teach", json={"target": "nope"}).status_code == 400


def test_teach_saves_to_calibration(taught):
    s = saved(taught)["autonomous"]
    assert s["pickup"] == {"x": 40.0, "y": 60.0, "z": 40.0}
    assert s["bin"]["x"] == 300.0
    assert s["safe_z"] == 90.0
    assert taught.get("/api/auto").json()["ready"] == {"grip": True, "dispose": True}


def test_grip_goes_up_then_xy_then_down_then_lifts_to_centre(taught):
    taught.post("/api/machine/move", json={"x": 200, "y": 200, "z": 30, "speed_mm_s": 100})
    before = len(moves(taught))
    assert taught.post("/api/auto/grip").status_code == 200
    s = wait_done(taught)
    assert s["finished"] and not s["error"], s
    new = moves(taught)[before:]
    assert "Z90" in new[0] and "X" not in new[0]          # rise to safe Z first, no XY
    assert "X40" in new[1] and "Y60" in new[1] and "Z" not in new[1]
    assert "Z40" in new[2] and "X" not in new[2]          # then down onto the swab
    assert "Z70" in new[3]                                # 40 + 30 mm lift
    assert "X175" in new[4] and "Y175" in new[4]          # centre of the gantry
    g = taught.server.hub.gripper
    assert g.log[-1] == "close" and "open" in g.log


def test_dispose_goes_to_bin_and_opens(taught):
    taught.post("/api/machine/move", json={"x": 100, "y": 100, "z": 30, "speed_mm_s": 100})
    before = len(moves(taught))
    assert taught.post("/api/auto/dispose").status_code == 200
    s = wait_done(taught)
    assert s["finished"] and not s["error"], s
    new = moves(taught)[before:]
    assert "Z90" in new[0] and "X" not in new[0]
    assert "X300" in new[1] and "Y50" in new[1]
    assert "Z45" in new[2]
    assert "Z90" in new[3]
    assert taught.server.hub.gripper.log[-1] == "open"


def test_no_lift_when_already_above_safe_z(taught):
    taught.post("/api/machine/move", json={"x": 200, "y": 200, "z": 95, "speed_mm_s": 100})
    before = len(moves(taught))
    taught.post("/api/auto/dispose")
    wait_done(taught)
    assert "X300" in moves(taught)[before]


def test_motion_is_refused_while_it_runs(taught):
    taught.post("/api/auto/settings", json={"grip_wait_s": 2.0})
    taught.post("/api/auto/grip")
    time.sleep(0.2)
    r = taught.post("/api/machine/jog", json={"dx": 1})
    assert r.status_code == 409
    taught.post("/api/auto/stop")
    s = wait_done(taught)
    assert "stopped" in s["error"]
    assert any(st["state"] == "skipped" for st in s["steps"])


def test_not_homed_stops_before_any_move(taught):
    taught.server.hub.printer._inner.homed = ""
    before = len(moves(taught))
    taught.post("/api/auto/grip")
    s = wait_done(taught)
    assert "Home X and Y" in s["error"]
    assert len(moves(taught)) == before


def test_safe_z_below_target_is_refused(taught):
    taught.post("/api/machine/move", json={"z": 30, "speed_mm_s": 100})
    taught.post("/api/auto/teach", json={"target": "safe_z"})
    r = taught.post("/api/auto/grip")
    assert r.status_code == 400 and "Safe Z must be at or above" in r.json()["error"]


def test_settings_are_range_checked(taught):
    assert taught.post("/api/auto/settings", json={"lift_mm": 5000}).status_code == 400
    assert taught.post("/api/auto/settings", json={"lift_mm": 25}).json()["settings"]["lift_mm"] == 25.0


def test_redeclaring_z_makes_taught_positions_stale(taught):
    assert taught.get("/api/auto").json()["ready"]["grip"] is True
    taught.post("/api/machine/set_z", json={"z": 50.0, "up_mm": 100.0})
    info = taught.get("/api/auto").json()
    assert info["ready"] == {"grip": False, "dispose": False}
    r = taught.post("/api/auto/grip")
    assert r.status_code == 400 and "Z reference changed" in r.json()["error"]


def test_reteaching_after_a_new_z_reference_works_again(taught):
    taught.post("/api/machine/set_z", json={"z": 50.0, "up_mm": 100.0})
    taught.post("/api/guard/floor", json={"z": 10.0})
    taught.post("/api/machine/move", json={"x": 40, "y": 60, "z": 30, "speed_mm_s": 100})
    taught.post("/api/auto/teach", json={"target": "pickup"})
    taught.post("/api/machine/move", json={"z": 45, "speed_mm_s": 100})
    taught.post("/api/auto/teach", json={"target": "safe_z"})
    assert taught.get("/api/auto").json()["ready"]["grip"] is True


def test_no_rise_when_already_over_the_target(taught):
    taught.post("/api/machine/move", json={"x": 40, "y": 60, "z": 41, "speed_mm_s": 100})
    before = len(moves(taught))
    taught.post("/api/auto/grip")
    wait_done(taught)
    new = moves(taught)[before:]
    assert not any("Z90" in m for m in new[:3])
    assert any("Z40" in m for m in new[:3])


def test_restore_z_keeps_taught_positions_valid(taught):
    taught.get("/api/machine/position")
    last = taught.get("/api/guard").json()["last_z"]
    assert last is not None
    printer = taught.server.hub.printer._inner
    printer.homed = "xy"                                   # Klipper forgot Z
    assert taught.post("/api/machine/restore_z").status_code == 200
    assert taught.get("/api/guard").json()["floor"] == 20.0
    assert taught.get("/api/auto").json()["ready"]["grip"] is True


def test_restore_z_refuses_when_z_is_already_set(taught):
    r = taught.post("/api/machine/restore_z")
    assert r.status_code == 400 and "already set" in r.json()["error"]


def test_calibration_save_keeps_autonomous_data(taught):
    cal = taught.get("/api/calibration").json()
    cal.pop("autonomous", None)
    assert taught.post("/api/calibration", json=cal).status_code == 200
    assert taught.get("/api/auto").json()["ready"]["grip"] is True


def test_settings_persist(taught):
    taught.post("/api/auto/settings", json={"xy_speed_mm_s": 35, "z_speed_mm_s": 3})
    s = taught.get("/api/auto").json()["settings"]
    assert s["xy_speed_mm_s"] == 35 and s["z_speed_mm_s"] == 3
