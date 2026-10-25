"""Line spacing against kerf: will two paths come out as two lines or one?"""
import pytest

from papa.toolpath.dxf_loader import Stroke
from papa.toolpath.spacing import closest_approach, report


def line(x0, y0, x1, y1):
    return Stroke(points=[(x0, y0), (x1, y1)])


def test_two_parallel_lines_report_their_gap():
    s = closest_approach([line(0, 0, 10, 0), line(0, 0.5, 10, 0.5)], step_mm=0.1)
    assert s.min_distance_mm == pytest.approx(0.5, abs=0.02)
    assert s.between_strokes == (0, 1) or s.between_strokes == (1, 0)


def test_a_lone_straight_line_is_not_close_to_itself():
    # every point is near the next one; that is not a merge risk
    s = closest_approach([line(0, 0, 20, 0)], step_mm=0.1)
    assert s.min_distance_mm == float("inf")


def test_a_path_doubling_back_does_count():
    # a hairpin genuinely can fuse, so it must not be ignored
    hairpin = Stroke(points=[(0, 0), (10, 0), (10, 0.3), (0, 0.3)])
    s = closest_approach([hairpin], step_mm=0.1, ignore_along_stroke_mm=1.0)
    assert s.min_distance_mm == pytest.approx(0.3, abs=0.02)


def test_a_wide_kerf_fuses_close_lines():
    strokes = [line(0, 0, 10, 0), line(0, 0.4, 10, 0.4)]
    s = closest_approach(strokes, step_mm=0.1)
    assert s.merges_at(0.6)
    assert not s.merges_at(0.2)
    assert s.margin_mm(0.2) == pytest.approx(0.2, abs=0.02)
    assert s.margin_mm(0.6) < 0


def test_report_warns_when_lines_would_merge():
    r = report([line(0, 0, 10, 0), line(0, 0.4, 10, 0.4)], kerf_mm=0.6, step_mm=0.1)
    assert r["merges"] is True
    assert "one line" in r["warning"]


def test_report_is_honest_when_kerf_is_unknown():
    r = report([line(0, 0, 10, 0), line(0, 2, 10, 2)], kerf_mm=None, step_mm=0.1)
    assert r["min_distance_mm"] == pytest.approx(2.0, abs=0.02)
    assert "not measured yet" in r["warning"]
    assert "merges" not in r


def test_comfortable_spacing_passes_without_a_warning():
    r = report([line(0, 0, 10, 0), line(0, 3, 10, 3)], kerf_mm=0.5, step_mm=0.2)
    assert r["merges"] is False
    assert r["warning"] is None
    assert r["margin_mm"] == pytest.approx(2.5, abs=0.05)


def test_empty_pattern_does_not_blow_up():
    assert closest_approach([]).min_distance_mm == float("inf")
    assert report([], kerf_mm=0.5)["min_distance_mm"] is None


def test_crossing_lines_are_counted_not_reported_as_a_zero_gap():
    # an X: the two lines are meant to meet, and must not mask a real near-miss
    cross = [line(0, 0, 10, 10), line(0, 10, 10, 0), line(20, 0, 20, 10),
             line(20.7, 0, 20.7, 10)]
    s = closest_approach(cross, step_mm=0.1)
    assert s.crossings > 0
    assert s.min_distance_mm == pytest.approx(0.7, abs=0.05)
