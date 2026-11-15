import time

from papa.ops import scrub as sc
from tests.test_validate import api  # noqa: F401


def _st(**kw):
    d = dict(sc.DEFAULTS)
    d.update(kw)
    return d


def test_empty_plan_is_odd_serpentine_top_left_to_bottom_right():
    p = sc.plan(_st())
    assert p["rows"] % 2 == 1
    assert p["segments"][0]["x0"] == 0 and p["segments"][0]["y0"] == 0
    assert p["end"] == [25.0, 25.0]
    assert p["hops"] == 0
    assert p["uncovered_pct"] == 0


def test_bridge_cuts_rows_and_rounds_corners():
    b = [{"id": 1, "cx": 12.5, "cy": 12.5, "l": 4, "w": 4}]
    p = sc.plan(_st(bridges=b))
    r = p["radius_mm"]
    for s in p["segments"]:
        if s["kind"] == "hop":
            continue
        for t in (0, 0.25, 0.5, 0.75, 1):
            x = s["x0"] + (s["x1"] - s["x0"]) * t
            y = s["y0"] + (s["y1"] - s["y0"]) * t
            assert sc._dist_to_rect(x, y, b[0]) >= r - 1e-3
    assert p["hops"] >= 1
    assert p["end"] == [25.0, 25.0]
    assert p["uncovered_pct"] < 25


def test_blocked_span_at_corner_is_narrower_than_at_edge():
    b = [{"cx": 10, "cy": 10, "l": 2, "w": 2}]
    r = 0.6
    full = sc._blocked(10, b, r)[0]
    top = sc._blocked(9 - r + 0.05, b, r)[0]
    assert (full[1] - full[0]) > (top[1] - top[0])


def test_problems_need_start_and_floor(api):
    m = api.server.hub.machine
    assert sc.problems(_st(), m, None, None)
    start = {"x": 100, "y": 200, "z": 40, "ref": None}
    assert sc.problems(_st(start=start), m, None, None)
    assert not sc.problems(_st(start=start), m, None, 39.0)
    assert sc.problems(_st(start=start), m, None, 41.0)
    assert sc.problems(_st(start={"x": 340, "y": 200, "z": 40}), m, None, 39.0)


def test_endpoints_and_full_run(api):
    srv = api.server
    mock = srv.hub.printer._inner
    mock.speed_factor = 0
    mock.homed = "xyz"
    mock.x, mock.y, mock.z = 100.0, 200.0, 60.0
    assert api.post("/api/scrub/start-here").status_code == 200
    r = api.post("/api/scrub/run")
    assert r.status_code == 400
    api.post("/api/guard/floor", json={"z": 55.0})
    api.post("/api/scrub/settings", json={"swab_mm": 2.0, "scrub_mm_s": 30.0, "travel_mm_s": 150.0, "z_mm_s": 10.0})
    api.post("/api/scrub/sketch", json={"substrate": {"w": 25, "h": 25},
                                       "bridges": [{"id": 1, "cx": 12, "cy": 12, "l": 3, "w": 3}]})
    plan = api.get("/api/scrub/plan").json()
    assert plan["hops"] >= 1
    assert api.post("/api/scrub/run").status_code == 200
    for _ in range(300):
        s = api.get("/api/scrub/status").json()
        if not s["running"]:
            break
        time.sleep(0.05)
    assert s["finished"], s
    assert abs(mock.x - 125.0) < 1e-6 and abs(mock.y - 175.0) < 1e-6
    assert abs(mock.z - 63.0) < 1e-6


def test_run_blocked_while_busy_and_stop(api):
    srv = api.server
    mock = srv.hub.printer._inner
    mock.homed = "xyz"
    mock.x, mock.y, mock.z = 100.0, 200.0, 60.0
    mock.speed_factor = 5.0
    api.post("/api/scrub/start-here")
    api.post("/api/guard/floor", json={"z": 55.0})
    assert api.post("/api/scrub/run").status_code == 200
    assert api.post("/api/scrub/run").status_code == 409
    api.post("/api/scrub/stop")
    for _ in range(300):
        if not api.get("/api/scrub/status").json()["running"]:
            break
        time.sleep(0.05)
    s = api.get("/api/scrub/status").json()
    assert s["error"] and not s["finished"]


def test_last_z_is_saved_after_every_z_move(tmp_path):
    from papa.ops.guard import Guard, GuardedPrinter
    from papa.sim.mocks import MockMoonraker
    g = Guard(tmp_path / "f.json")
    p = GuardedPrinter(MockMoonraker(z=50.0, homed="xyz"), g)
    p.move(z=60.0)
    assert Guard(tmp_path / "f.json").last_z == 60.0
    p.move_relative(dz=2.5)
    assert Guard(tmp_path / "f.json").last_z == 62.5
