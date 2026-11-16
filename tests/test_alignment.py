"""Camera-based per-substrate alignment: geometry, fit, detection and the scrub path.

No server here (that is in test_validate), so these run without FastAPI.
"""
import math

import numpy as np
import pytest

from papa.ops import alignment as al
from papa.ops import scrub as sc
from papa.ops.geometry import to_local, to_machine
from papa.sim.arrays import SimArray, render
from papa.sim.mocks import MockMoonraker
from papa.vision import correction as corr

AXES = {"local_x": "+X", "local_y": "-Y"}


def sim_axes(rot_deg=1.2, mm_per_px=0.11):
    """A consistent camera -> printer mapping (plate px -> mm), Y mirrored."""
    th, s = math.radians(rot_deg), mm_per_px
    ref = np.array([[300, 200], [2000, 220], [1980, 1150], [320, 1130]], float)
    A = np.array([[s * math.cos(th), s * math.sin(th)], [s * math.sin(th), -s * math.cos(th)]])
    return corr.PrinterAxes(ref, ref @ A.T + [20, 240])


# ---- geometry ----

@pytest.mark.parametrize("axes", [
    {"local_x": "+X", "local_y": "-Y"}, {"local_x": "+X", "local_y": "+Y"},
    {"local_x": "+Y", "local_y": "+X"}, {"local_x": "-Y", "local_y": "-X"},
])
def test_to_machine_and_back(axes):
    for rot in (0.0, 2.5, -7.0):
        mx, my = to_machine(axes, 100, 50, 3.0, 4.0, rot)
        lx, ly = to_local(axes, 100, 50, mx, my, rot)
        assert (lx, ly) == pytest.approx((3.0, 4.0))


def test_documented_axis_meaning():
    assert to_machine({"local_x": "+Y", "local_y": "+X"}, 0, 0, 1, 0) == (0, 1)
    assert to_machine({"local_x": "+X", "local_y": "-Y"}, 10, 10, 2, 3) == (12, 7)


def test_rotation_turns_from_plus_x_toward_plus_y():
    x, y = to_machine(AXES, 0, 0, 10, 0, 90)
    assert (x, y) == pytest.approx((0, 10))


# ---- fit ----

def corners_for(origin, w, h, rot):
    return np.array([to_machine(AXES, origin[0], origin[1], x, y, rot)
                     for x, y in ((0, 0), (w, 0), (w, h), (0, h))])


@pytest.mark.parametrize("rot", [0.0, 1.3, -4.0, 30.0])
def test_fit_recovers_rotation_and_origin_any_corner_order(rot):
    c = corners_for((120.0, 80.0), 12, 10, rot)[[2, 0, 3, 1]]     # shuffled
    p = al.place("S1", AXES, c, {"area": {"w": 12, "h": 10}}, {"max_rotation_deg": 45})
    assert p.rotation_deg == pytest.approx(rot, abs=1e-9)
    assert p.origin_mm == pytest.approx([120.0, 80.0], abs=1e-9)
    assert p.measured_mm == pytest.approx([12, 10])
    assert p.fit_rms_mm == pytest.approx(0, abs=1e-9)
    assert p.warnings == []


def test_size_difference_is_shared_around_the_array():
    c = corners_for((50.0, 50.0), 12.4, 10.0, 0.0)     # 0.4 mm wider than sketched
    p = al.place("S1", AXES, c, {"area": {"w": 12, "h": 10}}, {})
    assert p.origin_mm == pytest.approx([50.2, 50.0])
    assert p.size_error_mm == pytest.approx([0.4, 0.0])
    assert any("Measured 12.40 x 10.00" in w for w in p.warnings)


def test_turned_ninety_and_too_much_rotation_warn():
    c = corners_for((50.0, 50.0), 10, 12, 0.0)
    p = al.place("S1", AXES, c, {"area": {"w": 12, "h": 10}}, {})
    assert any("90 deg" in w for w in p.warnings)
    p = al.place("S1", AXES, corners_for((50, 50), 12, 10, 15), {"area": {"w": 12, "h": 10}}, {})
    assert any("more than the 10 deg" in w for w in p.warnings)


def test_numbering_reads_rows_top_to_bottom_in_the_local_frame():
    # local +y is machine -Y, so the top row has the larger machine Y
    centers = [np.array(v) for v in ((60, 150), (100, 200), (60, 200), (100, 150))]
    order = al.number(AXES, centers, row_mm=5)
    assert [tuple(centers[i]) for i in order] == [(60, 200), (100, 200), (60, 150), (100, 150)]


# ---- printer axes ----

def test_printer_axes_fit_and_report():
    pa = sim_axes()
    rep = pa.report()
    assert rep["mirrored"] is True
    assert max(rep["errors_mm"]) < 1e-9
    assert rep["mm_per_square"] == pytest.approx(0.11 * 135)
    pts = np.array([[500.0, 600.0], [1700, 300]])
    assert pa.mm_to_plate(pa.plate_to_mm(pts)) == pytest.approx(pts)


def test_printer_axes_flags_a_bad_point():
    pa = sim_axes()
    ref = np.array([[300, 200], [2000, 220], [1980, 1150], [320, 1130], [1100, 700]], float)
    mm = pa.plate_to_mm(ref)
    mm[4] += [1.5, 0]
    rep = corr.PrinterAxes(ref, mm).report()
    assert any("off the fit" in w for w in rep["warnings"])


def test_from_detection_uses_the_lens_model():
    pa = sim_axes()
    c = corr.get_correction((1920, 1080))
    plate = np.array([[400.0, 300.0], [1900, 320], [1880, 1100]])
    det = {"camera_to_machine": {"points_px": c.from_plate(plate).tolist(),
                                 "points_mm": pa.plate_to_mm(plate).tolist()}}
    pa2 = corr.PrinterAxes.from_detection(det)
    assert max(pa2.errors_mm) < 1e-6
    with pytest.raises(corr.AxesError):
        corr.PrinterAxes.from_detection({"camera_to_machine": {"points_px": [[1, 2]], "points_mm": [[1, 2]]}})


# ---- detection on a rendered frame ----

TRUTH = [SimArray((60, 200), 12, 10, 0.0), SimArray((95, 201), 12, 10, 1.5),
         SimArray((130, 199), 12, 10, -2.0), SimArray((60, 170), 12, 10, 3.0),
         SimArray((95, 170), 12, 10, -0.7)]


@pytest.fixture(scope="module")
def frame():
    pa = sim_axes()
    return pa, render(TRUTH, AXES, pa)


def test_scan_finds_every_array_with_pose(frame):
    pa, rgb = frame
    res = al.scan(rgb, AXES, pa, {"size_tolerance_mm": 0.3}, {"area": {"w": 12, "h": 10}, "bridges": []})
    assert [p.id for p in res.placements] == ["S1", "S2", "S3", "S4", "S5"]
    for p, t in zip(res.placements, TRUTH):
        assert p.origin_mm == pytest.approx(list(t.origin_mm), abs=0.1)
        assert p.rotation_deg == pytest.approx(t.rot_deg, abs=0.4)
        assert p.measured_mm == pytest.approx([12, 10], abs=0.3)
        assert p.warnings == []


def test_scan_uses_own_pattern_or_the_scrub_tab(frame):
    pa, rgb = frame
    own = {"name": "wide", "area": {"w": 12, "h": 10}, "bridges": [{"cx": 6, "cy": 5, "l": 2, "w": 1}]}
    res = al.scan(rgb, AXES, pa, {"patterns": {"S2": own}}, {"area": {"w": 12, "h": 10}, "bridges": []})
    d = res.as_dict()
    assert [s["pattern_source"] for s in d["substrates"]] == ["scrub tab", "own", "scrub tab", "scrub tab", "scrub tab"]
    assert len(d["substrates"][0]["corners_px"]) == 4


def test_scan_without_arrays_says_so():
    pa = sim_axes()
    res = al.scan(render([], AXES, pa), AXES, pa, {}, {"area": {"w": 12, "h": 10}})
    assert res.placements == [] and "No electrode arrays" in res.warnings[0]


# ---- the placed toolpath ----

def test_scrub_runs_the_plan_rotated_about_the_array_corner():
    class M:
        axes = AXES
        limits = {"x": [0, 350], "y": [0, 350], "z": [0, 340]}

        def in_limits(self, x, y, z):
            return 0 <= x <= 350 and 0 <= y <= 350 and 0 <= z <= 340

    origin, rot = (120.0, 80.0), 3.0
    p = al.place("S1", AXES, corners_for(origin, 6, 4, rot), {"area": {"w": 6, "h": 4}}, {})
    base = sc.merged({"start": {"x": 0, "y": 0, "z": 40, "ref": None}, "swab_mm": 1.0})
    st = al.scrub_settings(base, {"area": {"w": 6, "h": 4}, "bridges": []}, p)
    assert st["start"]["z"] == 40 and st["rotation_deg"] == pytest.approx(rot)
    assert not sc.problems(st, M(), None, 39.0)

    mock = MockMoonraker()
    mock.speed_factor = 0
    mock.homed = "xyz"
    runner = sc.ScrubRunner(lambda: mock, M())
    plan = sc.plan(st)
    runner.execute(st, plan, lambda: None)
    xy = [(mock.x, mock.y)]
    moves = [line for line in mock.log if line.startswith(("G0", "G1")) and " X" in line]
    assert moves, "no XY moves were sent"
    # every scrub end point lands where the rotated local point should be
    want = [to_machine(AXES, origin[0], origin[1], s["x1"], s["y1"], rot) for s in plan["segments"]]
    got = []
    for line in moves:
        toks = dict((t[0], float(t[1:])) for t in line.split()[1:] if t[0] in "XY")
        got.append((toks["X"], toks["Y"]))
    for w in want:
        assert min(math.dist(w, g) for g in got) < 2e-3
    assert xy


def test_detects_the_real_array_in_lab_lighting():
    """The corrected view of the PAPA plate from the real camera, with one
    array on its holder (dimmer than the simulation: the holder reads about
    140 grey), yellow caution labels and a blue part that must not count."""
    import cv2
    from pathlib import Path

    from papa.vision import electrodes as el

    img = cv2.imread(str(Path(__file__).parent / "data" / "real_plate_corrected.jpg"))
    found = el.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    assert len(found) == 1
    c = found[0].corners
    sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    assert 60 < min(sides) and max(sides) < 90          # about 80 x 65 px in this picture
    assert 440 < c[:, 0].mean() < 560 and 410 < c[:, 1].mean() < 520


def test_detects_the_real_array_when_its_lines_look_dark():
    """Same camera, other light: the electrode lines read near black rather
    than copper, so colour alone finds nothing. Holes, slots and label text
    on the same picture must not count."""
    import cv2
    from pathlib import Path

    from papa.vision import electrodes as el

    img = cv2.imread(str(Path(__file__).parent / "data" / "real_plate_dark_lines.jpg"))
    found = el.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    assert len(found) == 1
    c = found[0].corners
    assert 300 < c[:, 0].mean() < 390 and 200 < c[:, 1].mean() < 270
    sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    assert 50 < min(sides) and max(sides) < 85


def test_full_resolution_search_is_fast_and_as_exact():
    """A full-resolution scan is searched on a smaller copy and measured in a
    window around each array: much faster, same outline."""
    import time
    from pathlib import Path

    import cv2

    from papa.vision import electrodes as el

    small = cv2.imread(str(Path(__file__).parent / "data" / "real_plate_dark_lines.jpg"))
    big = cv2.cvtColor(cv2.resize(small, (small.shape[1] * 3, small.shape[0] * 3), interpolation=cv2.INTER_CUBIC),
                       cv2.COLOR_BGR2RGB)
    t = time.monotonic()
    fast = el.detect(big)
    took = time.monotonic() - t
    slow = el.detect(big, coarse_width=10**6)
    assert len(fast) == len(slow) == 1
    assert np.abs(fast[0].corners - slow[0].corners).max() < 1.5      # px, on a ~210 px wide array
    assert took < 3.0
