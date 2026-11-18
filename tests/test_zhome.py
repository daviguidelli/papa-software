"""Z home on the load cell, the relative heights, park, and routine 2."""
import time

import pytest

from tests.test_align_api import aligned  # noqa: F401  (fixture)
from tests.test_validate import api, saved  # noqa: F401  (fixture)


def moves(client):
    return [ln for ln in client.server.hub.printer._inner.log if ln.upper().startswith(("G0 ", "G1 "))]


def kin(client):
    return [ln for ln in client.server.hub.printer._inner.log if ln.startswith("SET_KINEMATIC")]


@pytest.fixture
def homed_cell(aligned):
    c = aligned
    mock = c.server.hub.printer._inner
    mock.speed_factor = 0
    c.post("/api/machine/home", json={"axes": "XY"})
    c.post("/api/machine/set_z", json={"z": 100.0, "up_mm": 100.0})
    c.post("/api/guard/floor", json={"z": 20.0})
    c.post("/api/machine/move", json={"x": 30, "y": 40, "z": 60, "speed_mm_s": 100})
    c.post("/api/loadcell/position", json={"capture_xy": True, "capture_z": True})
    c.server.hub.loadcell.contact_z = 58.0
    c.post("/api/machine/move", json={"x": 40, "y": 60, "z": 40, "speed_mm_s": 100})
    c.post("/api/auto/teach", json={"target": "pickup"})
    c.post("/api/machine/move", json={"x": 300, "y": 50, "z": 45, "speed_mm_s": 100})
    c.post("/api/auto/teach", json={"target": "bin"})
    c.post("/api/machine/move", json={"z": 90, "speed_mm_s": 100})
    c.post("/api/auto/teach", json={"target": "safe_z"})
    c.post("/api/auto/settings", json={"grip_wait_s": 0.0, "release_wait_s": 0.0, "pickup_step_y": 5.0})
    c.post("/api/machine/move", json={"x": 100, "y": 200, "z": 42, "speed_mm_s": 100})
    c.post("/api/scrub/start-here")
    c.post("/api/scrub/settings", json={"swab_mm": 2.0, "scrub_mm_s": 30.0, "travel_mm_s": 150.0, "z_mm_s": 10.0})
    c.post("/api/zhome/heights", json={"slide_mm": -2.0, "pickup_mm": 3.0, "bin_mm": 5.0, "safe_mm": 30.0})
    return c


def test_zhome_needs_the_setup_first(api):
    assert api.post("/api/zhome/run").status_code == 400
    j = api.get("/api/zhome").json()
    assert "Home Z on the load cell first." in j["problems"]


def test_zhome_declares_the_contact_and_derives_every_height(homed_cell):
    c = homed_cell
    r = c.post("/api/zhome/run")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["contact_z_before"] == pytest.approx(58.0 - 1.1 / 0.00980665 / 600, abs=0.02)
    assert j["floor"] == pytest.approx(47.5)             # slide at 48, 0.5 below it
    cal = saved(c)
    assert cal["autonomous"]["pickup"]["z"] == 53.0 and cal["autonomous"]["bin"]["z"] == 55.0
    assert cal["autonomous"]["safe_z"] == 80.0
    assert (cal["autonomous"]["pickup"]["x"], cal["autonomous"]["pickup"]["y"]) == (40.0, 60.0)
    assert cal["scrub"]["start"]["z"] == 48.0
    assert c.get("/api/machine/position").json()["z"] == pytest.approx(80.0)
    assert c.get("/api/zhome").json()["problems"] == []
    assert c.get("/api/auto").json()["ready"] == {"grip": True, "dispose": True}


def test_changing_a_height_after_zhome_moves_the_taught_z_and_floor(homed_cell):
    c = homed_cell
    c.post("/api/zhome/run")
    c.post("/api/zhome/heights", json={"slide_mm": -1.5})
    assert saved(c)["scrub"]["start"]["z"] == 48.5
    assert c.get("/api/guard").json()["floor"] == pytest.approx(48.0)


def test_park_only_goes_up(homed_cell):
    c = homed_cell
    c.post("/api/zhome/run")
    c.post("/api/machine/move", json={"z": 60, "speed_mm_s": 100})
    assert c.post("/api/park").status_code == 200
    assert c.get("/api/machine/position").json()["z"] == pytest.approx(80.0)
    c.post("/api/machine/move", json={"z": 95, "speed_mm_s": 100})
    c.post("/api/park")
    assert c.get("/api/machine/position").json()["z"] == pytest.approx(95.0)


def wait2(c, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        s = c.get("/api/routine2/status").json()
        if not s["running"]:
            return s
        time.sleep(0.03)
    raise AssertionError("routine 2 did not finish")


def test_routine2_checks_scribes_swaps_and_z_is_left_alone(homed_cell):
    c = homed_cell
    c.post("/api/zhome/run")
    c.post("/api/align/scan", json={"save": False})
    c.post("/api/scrub/sketch", json={"substrate": {"w": 25, "h": 25}, "bridges": []})
    c.server.hub.loadcell.contact_z = 50.187 + 0.3      # a swab 0.3 mm longer than the one Z was homed with
    before = len(kin(c))
    assert c.get("/api/routine2").json()["ready"]
    r = c.post("/api/routine2/run", json={"slides": ["S1", "S2"], "force": True})
    assert r.status_code == 200, r.text
    s = wait2(c)
    assert s["finished"] and not s["error"], s
    names = [x["name"] for x in s["steps"]]
    assert [n for n in names if n.startswith("Scrub")] == ["Scrub slide A", "Scrub slide B"]
    chk = "Check Z on the load cell (Z unchanged)"
    assert names.index(chk) < names.index("Scrub slide A") < names.index("[position 1] Close the jaws")
    assert names.index("[position 1] Close the jaws") < names.index(chk, names.index("Scrub slide A")) < names.index("Scrub slide B")
    assert names.count(chk) == 2
    assert sum(n.startswith("[position") and n.endswith("Close the jaws") for n in names) == 2
    assert names[-1] == "Park at safe Z"
    assert s["last_check"]["drift_mm"] == pytest.approx(0.3, abs=0.02)
    assert len(kin(c)) == before, "the check must not redeclare Z"
    assert c.get("/api/machine/position").json()["z"] >= 80.0 - 1e-6, "parked at or above safe Z"
    ys = {ln.split("Y")[1].split()[0] for ln in moves(c) if " X" in ln and " Y" in ln}
    assert "60.0000" in ys and "65.0000" in ys, "position 1 is the pickup, position 2 is one pickup step further"
    assert any("Z48.0000" in ln for ln in moves(c)), "scribing happens at the slide height"


def test_routine2_refuses_without_zhome(aligned):
    r = aligned.post("/api/routine2/run", json={"force": True})
    assert r.status_code == 400
