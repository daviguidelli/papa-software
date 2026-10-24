import pytest

from papa.config import Calibration
from papa.ops.swabs import Rack, RackEmpty, Wear


def make_cal(rows=3):
    swabs = []
    sid = 1
    for r in range(1, rows + 1):
        for side in ("right", "left"):
            swabs.append({"id": sid, "row": r, "side": side,
                          "x": 100.0 - (0 if side == "right" else 14),
                          "y": 20.0 + r * 10, "z_grip": 12.0, "calibrated": True})
            sid += 1
    return Calibration.from_raw({
        "swab_rack": {"rows": rows, "swabs_per_slide": 2, "swabs": swabs},
        "slide_slots": {"rows": 1, "cols": 1, "slots": []},
        "load_cell": {"x": 0, "y": 0, "z_safe": 25, "scale_counts_per_g": 1, "tare_counts": 0},
        "disposal": {"x": 0, "y": 0, "z_release": 10},
        "gripper": {"open_us": 1000, "closed_us": 2000},
    })


def test_left_is_blocked_until_the_right_of_that_row_is_gone():
    rack = Rack.from_detection(make_cal(), ["1R", "1L", "2R", "2L"])
    assert rack.pickable("1R")
    assert not rack.pickable("1L"), "reaching left first would hit the right swab"
    assert rack.next_label() == "1R"
    rack.take("1R")
    assert rack.pickable("1L")
    assert rack.next_label() == "1L"


def test_a_row_missing_its_right_swab_frees_the_left_one():
    rack = Rack.from_detection(make_cal(), ["1L", "2R"])
    assert rack.pickable("1L"), "never loaded counts as gone"
    assert rack.next_label() == "1L"


def test_order_is_right_then_left_row_by_row():
    rack = Rack.from_detection(make_cal(), ["1R", "1L", "2R", "2L", "3R"])
    assert rack.order() == ["1R", "1L", "2R", "2L", "3R"]


def test_reserve_assigns_in_pick_order():
    rack = Rack.from_detection(make_cal(), ["1R", "1L", "2R", "2L"])
    plan = rack.reserve([3, 7], per_slide=2)
    assert plan == {3: ["1R", "1L"], 7: ["2R", "2L"]}


def test_running_out_raises_rather_than_scribing_with_nothing():
    rack = Rack.from_detection(make_cal(), ["1R"])
    rack.take(rack.next_label())
    with pytest.raises(RackEmpty):
        rack.next_label()


def test_uncalibrated_position_is_refused():
    cal = make_cal()
    raw = dict(cal.raw)
    raw["swab_rack"]["swabs"][0]["calibrated"] = False
    rack = Rack.from_detection(Calibration.from_raw(raw), ["1R"])
    with pytest.raises(ValueError, match="calibrated"):
        rack.take("1R")


def test_wear_trips_on_either_limit():
    w = Wear()
    w.add(path_mm=99)
    assert not w.worn(100, 120)
    w.add(path_mm=2)
    assert w.worn(100, 120)
    w2 = Wear()
    w2.add(contact_s=121)
    assert w2.worn(100, 120)
    assert w2.fraction(100, 120) == 1.0
