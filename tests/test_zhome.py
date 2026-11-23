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


def test_routine2_from_hand_captured_slot_corners_and_pickup2(homed_cell):
    c = homed_cell
    c.post("/api/zhome/run")
    c.server.hub.loadcell.contact_z = 50.187            # the mock cell, in the frame Z home declared
    c.post("/api/machine/move", json={"x": 150, "y": 100, "speed_mm_s": 100})
    c.post("/api/slide/corner", json={"id": 1})
    c.post("/api/machine/move", json={"x": 200, "y": 100, "speed_mm_s": 100})
    c.post("/api/slide/corner", json={"id": 2})
    c.post("/api/scrub/sketch", json={"substrate": {"w": 25, "h": 25}, "bridges": []})
    c.post("/api/machine/move", json={"x": 40, "y": 77, "speed_mm_s": 100})
    j = c.post("/api/auto/pickup2").json()
    assert j["settings"]["pickup_step"] == {"x": 0.0, "y": 17.0}
    r = c.post("/api/routine2/run", json={"slides": ["1", "2"]})
    assert r.status_code == 200, r.text
    s = wait2(c)
    assert s["finished"] and not s["error"], s
    xs = {ln.split("X")[1].split()[0] for ln in moves(c) if " X" in ln}
    assert "150.0000" in xs and "200.0000" in xs
    assert any("Z48.0000" in ln for ln in moves(c))


def test_routine2_slot_needs_a_captured_corner(homed_cell):
    c = homed_cell
    c.post("/api/zhome/run")
    r = c.post("/api/routine2/run", json={"slides": ["8", "9"]})
    assert r.status_code == 400 and "top-left corner" in r.json()["error"]


def test_loadcell_calibrate_and_tare_save_themselves(api):
    import yaml
    r = api.post("/api/loadcell/tare")
    assert r.status_code == 200 and r.json()["saved"]
    assert "tare_counts" in saved(api)["load_cell"]
    api.server.hub.loadcell.pressed_g = 100.0 if hasattr(api.server.hub.loadcell, "pressed_g") else None
    r = api.post("/api/loadcell/calibrate", json={"grams": 100})
    if r.status_code == 200:
        assert r.json()["saved"]
        assert saved(api)["load_cell"]["scale_counts_per_g"] == pytest.approx(r.json()["scale_counts_per_g"])
    assert api.post("/api/loadcell/calibrate", json={"grams": 0}).status_code == 400


def test_setup_positions_show_up(homed_cell):
    c = homed_cell
    pos = c.get("/api/zhome").json()["positions"]
    assert pos["swab1"] == {"x": 40.0, "y": 60.0} and pos["bin"]["x"] == 300.0
    assert pos["swab2"] == {"x": 40.0, "y": 65.0}
    assert pos["cell"] == {"x": 30.0, "y": 40.0}
    assert c.get("/api/zhome").json()["loadcell"]["scaled"] in (True, False)


def test_zhome_after_restart_starts_a_fixed_height_above_the_contact(homed_cell):
    c = homed_cell
    c.post("/api/zhome/run")
    c.post("/api/machine/move", json={"z": 90, "speed_mm_s": 100})
    c.server.hub.loadcell.contact_z = 50.187
    r = c.post("/api/zhome/run")
    assert r.status_code == 200, r.text
    assert abs(r.json()["contact_z_before"] - (50.0 - 0.0)) < 0.5, "the second Z home finds the same contact"


def test_routine2_on_the_3x3_mount_slide1_is_slot2_empty_slots_skipped(homed_cell):
    """Slide 1's corner is grid slot 2; each other slot's corner is that corner
    moved by the mount spacing (1.995 in per column in +X, 2.011 in per row in
    -Y). Only the slots given as filled run, in slot order, each with a Z
    check, a scrub at the slide height, dispose and the next swab."""
    c = homed_cell
    c.post("/api/zhome/run")
    c.server.hub.loadcell.contact_z = 50.187
    c.post("/api/machine/move", json={"x": 150, "y": 200, "speed_mm_s": 100})
    c.post("/api/slide/corner", json={"id": 1})
    c.post("/api/scrub/sketch", json={"substrate": {"w": 10, "h": 10}, "bridges": []})
    c.post("/api/machine/move", json={"x": 40, "y": 77, "speed_mm_s": 100})
    c.post("/api/auto/pickup2")
    col, row = 1.995 * 25.4, 2.011 * 25.4
    info = c.get("/api/routine3x3").json()
    s1, s2, s5 = (next(x for x in info["cells"] if x["slot"] == k) for k in (1, 2, 5))
    assert s2["corner"] == pytest.approx([150, 200])
    assert s1["corner"] == pytest.approx([150 - col, 200], abs=1e-3)
    assert s5["corner"] == pytest.approx([150, 200 - row], abs=1e-3)
    r = c.post("/api/routine3x3/run", json={"slots": [2, 3]})       # the scan found slots 2 and 3 filled
    assert r.status_code == 200, r.text
    assert r.json()["runs"] == [2, 3] and 1 in r.json()["skips"]
    s = wait2(c)
    assert s["finished"] and not s["error"], s
    names = [x["name"] for x in s["steps"]]
    assert [n for n in names if n.startswith("Scrub")] == ["Scrub slot 2", "Scrub slot 3"]
    assert sum(n.endswith("Check Z on the load cell (Z unchanged)") for n in names) == 2
    assert names[-1] == "Park at safe Z"
    xs = {ln.split("X")[1].split()[0] for ln in moves(c) if " X" in ln}
    assert "150.0000" in xs and f"{150 + col:.4f}" in xs           # slot 2 and one column over
    assert any("Z48.0000" in ln for ln in moves(c))


def test_routine3x3_needs_a_scan_or_slots(homed_cell):
    c = homed_cell
    c.post("/api/zhome/run")
    c.post("/api/machine/move", json={"x": 150, "y": 200, "speed_mm_s": 100})
    c.post("/api/slide/corner", json={"id": 1})
    c.server.hub.align_scan = None
    j = c.get("/api/routine3x3").json()
    assert not j["ready"] and any("Scan the substrates" in p for p in j["problems"])
    assert c.post("/api/routine3x3/run", json={}).status_code == 400


def test_grid_check_visits_each_slot_corner_high_and_never_lowers(homed_cell):
    """The X/Y dry run: rises to the slide height + 30 mm (or safe Z), visits
    each slot's corner (Slide 1 = slot 2, mount spacing), never moves below
    that height, never grips or scrubs."""
    import time as _t
    c = homed_cell
    c.post("/api/zhome/run")
    c.post("/api/machine/move", json={"x": 150, "y": 200, "speed_mm_s": 100})
    c.post("/api/slide/corner", json={"id": 1})
    c.post("/api/scrub/sketch", json={"substrate": {"w": 10, "h": 8}, "bridges": []})
    c.server.hub.align_scan = None
    before = len(moves(c))
    r = c.post("/api/routine3x3/check", json={"slots": "all", "height_mm": 30, "dwell_s": 0, "outline": True})
    assert r.status_code == 200, r.text
    z = r.json()["z"]
    assert z >= 48.0 + 30 - 1e-6                         # slide height 48 + 30
    for _ in range(400):
        s = c.get("/api/routine3x3/check").json()
        if not s["running"]:
            break
        _t.sleep(0.02)
    assert s["error"] is None and s["done"] == list(range(1, 10)), s
    new = moves(c)[before:]
    zs = [float(ln.split("Z")[1].split()[0]) for ln in new if " Z" in ln]
    assert all(v >= z - 1e-6 for v in zs), "never below the check height"
    assert c.get("/api/machine/position").json()["z"] >= z - 1e-6
    col, row = 1.995 * 25.4, 2.011 * 25.4
    xs = {ln.split("X")[1].split()[0] for ln in new if " X" in ln}
    ys = {ln.split("Y")[1].split()[0] for ln in new if " Y" in ln}
    assert f"{round(150 - col, 3):.4f}" in xs and f"{round(150 + col, 3):.4f}" in xs    # slots 1 and 3
    assert f"{round(200 - 2 * row, 3):.4f}" in ys                                       # bottom row
    r = c.post("/api/routine3x3/check", json={"slots": "filled"})
    assert r.status_code == 400                                                   # no scan: nothing known filled


def test_grid_rows_can_be_flipped_to_plus_y(homed_cell):
    c = homed_cell
    c.post("/api/machine/move", json={"x": 182, "y": 91, "speed_mm_s": 100})
    c.post("/api/slide/corner", json={"id": 1})
    s7 = lambda j: next(x for x in j["cells"] if x["slot"] == 7)["corner"]
    assert s7(c.get("/api/routine3x3").json())[1] == pytest.approx(91 - 2 * 2.011 * 25.4, abs=1e-3)
    j = c.post("/api/routine3x3/flip", json={"axis": "row"}).json()
    assert s7(j)[1] == pytest.approx(91 + 2 * 2.011 * 25.4, abs=1e-3)
    assert j["spacing"]["source"] == "entered"
    j = c.post("/api/routine3x3/flip", json={"axis": "col"}).json()
    assert s7(j)[0] == pytest.approx(182 + 2.011 * 0 + 1.995 * 25.4, abs=1e-3)   # slot 7 is column 1: now +X of slot 2
    c.post("/api/master", json={"clear": "spacing"})


def test_grid_check_without_known_z_asks_then_moves_xy_only(api):
    c = api
    c.post("/api/machine/home", json={"axes": "XY"})
    c.post("/api/machine/move", json={"x": 182, "y": 91, "speed_mm_s": 100})
    c.post("/api/slide/corner", json={"id": 1})
    r = c.post("/api/routine3x3/check", json={"slots": [3], "dwell_s": 0})
    assert r.status_code == 409 and "Z is not known" in r.json()["error"]
    r = c.post("/api/routine3x3/check", json={"slots": [3], "dwell_s": 0, "z_unknown_ok": True})
    assert r.status_code == 200, r.text
    c.server.hub.thread.join(timeout=10)
    s = c.get("/api/routine3x3/check").json()
    assert s["error"] is None and s["done"] == [3] and s["z"] is None
    p = c.get("/api/machine/position").json()
    assert p["x"] == pytest.approx(182 + 1.995 * 25.4, abs=1e-3)


def test_grid_anchor_slot_can_be_changed(homed_cell):
    c = homed_cell
    c.post("/api/machine/move", json={"x": 182, "y": 91, "speed_mm_s": 100})
    c.post("/api/slide/corner", json={"id": 1})
    j = c.post("/api/routine3x3/anchor", json={"slot": 1}).json()
    assert j["anchor_slot"] == 1
    cell = lambda k: next(x for x in j["cells"] if x["slot"] == k)["corner"]
    assert cell(1) == pytest.approx([182, 91])
    assert cell(2) == pytest.approx([182 + 1.995 * 25.4, 91], abs=1e-3)
    assert c.get("/api/routine3x3").json()["anchor_slot"] == 1          # saved
    assert c.post("/api/routine3x3/anchor", json={"slot": 12}).status_code == 400
    c.post("/api/routine3x3/anchor", json={"slot": 2})
