from papa.config import load_calibration, load_machine


def test_machine_yaml_loads_and_clamps():
    m = load_machine()
    assert m.moonraker_url.startswith("http")
    assert m.clamp("z", -5) == 0
    assert m.clamp("x", 1e6) == m.limits["x"][1]
    assert not m.in_limits(-1, 0, 0)


def test_example_calibration_describes_the_substrate():
    c = load_calibration()
    s = c.substrate
    assert s["width_mm"] > 0 and s["length_mm"] > 0
    assert c.pattern_area[0] == c.pattern_area[1] == s["pattern_size_mm"]
    assert c.pattern_fits()
    assert c.rack_rows >= 1 and 1 <= c.swabs_per_slide <= 3
