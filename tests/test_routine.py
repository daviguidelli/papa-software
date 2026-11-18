import time

from tests.test_autonomous import taught, wait_done  # noqa: F401
from tests.test_validate import api  # noqa: F401


def wait_routine(client, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get("/api/routine1/status").json()
        if not s["running"]:
            return s
        time.sleep(0.03)
    raise AssertionError("routine did not finish")


def set_scrub(client):
    client.post("/api/machine/move", json={"x": 100, "y": 200, "z": 42, "speed_mm_s": 100})
    client.post("/api/scrub/start-here")
    client.post("/api/scrub/settings", json={"swab_mm": 2.0, "scrub_mm_s": 30.0,
                                             "travel_mm_s": 150.0, "z_mm_s": 10.0})
    client.post("/api/scrub/sketch", json={"substrate": {"w": 25, "h": 25},
                                          "bridges": [{"id": 1, "cx": 12, "cy": 12, "l": 3, "w": 3}]})


def test_routine_not_ready_without_scrub_start(taught):
    r = taught.post("/api/routine1/run")
    assert r.status_code == 400 and "Set start" in r.json()["error"]


def test_full_routine_grips_scrubs_and_disposes(taught):
    set_scrub(taught)
    mock = taught.server.hub.printer._inner
    mock.speed_factor = 0
    assert taught.get("/api/routine1").json()["ready"]
    assert taught.post("/api/routine1/run").status_code == 200
    s = wait_routine(taught)
    assert s["finished"] and not s["error"], s
    names = [x["name"] for x in s["steps"]]
    assert names.index("Close the jaws") < names.index("Scrub the pattern") < names.index("Open the jaws", names.index("Scrub the pattern"))
    assert all(x["state"] == "done" for x in s["steps"])
    assert abs(mock.x - 300.0) < 1e-6 and abs(mock.y - 50.0) < 1e-6
    log = mock.log
    xy_before_scrub = [l for l in log if l.startswith("G1 X125")]
    assert xy_before_scrub


def test_routine_scrub_fault_keeps_swab_and_skips_dispose(taught):
    set_scrub(taught)
    mock = taught.server.hub.printer._inner
    mock.speed_factor = 3.0
    assert taught.post("/api/routine1/run").status_code == 200
    time.sleep(0.4)
    while True:
        st = taught.get("/api/routine1/status").json()
        if any(x["name"] == "Scrub the pattern" and x["state"] == "running" for x in st["steps"]) or not st["running"]:
            break
        time.sleep(0.02)
    taught.post("/api/routine1/stop")
    s = wait_routine(taught)
    assert s["error"] and not s["finished"]
    states = {x["name"]: x["state"] for x in s["steps"]}
    assert states.get("Lower into the bin") == "skipped"


def test_grip_shift_moves_gantry_in_minus_x_while_jaws_close(taught):
    taught.post("/api/auto/settings", json={"grip_shift_mm": 2.0, "grip_close_s": 0.4})
    mock = taught.server.hub.printer._inner
    mock.speed_factor = 0
    grip = taught.server.hub.gripper
    assert taught.post("/api/auto/grip").status_code == 200
    end = time.time() + 10
    while time.time() < end and taught.get("/api/auto/status").json()["running"]:
        time.sleep(0.02)
    s = taught.get("/api/auto/status").json()
    assert s["finished"], s
    assert any(l.startswith("G1 X38.0000 Y60.0000") for l in mock.log)
    assert grip.state == "closed"


def test_grip_shift_outside_limits_is_refused(taught):
    taught.post("/api/auto/settings", json={"grip_shift_mm": 10.0})
    taught.post("/api/machine/move", json={"x": 1, "y": 60, "z": 40, "speed_mm_s": 100})
    taught.post("/api/auto/teach", json={"target": "pickup"})
    r = taught.post("/api/auto/grip")
    assert r.status_code == 400 and "grip shift" in r.json()["error"].lower()
