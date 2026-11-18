import pytest

from papa.ops.zcal import GripFailed, OverForce, calibrate_z
from papa.sim.mocks import MockLoadCell, MockMoonraker

ARGS = dict(
    cell_x=30.0, cell_y=40.0, z_safe=25.0,
    target_force_n=1.1, abort_force_n=3.0,
    step_mm=0.05, max_travel_mm=8.0,
    approach_speed_mm_s=1.0, travel_speed_mm_s=100.0,
    settle_readings=3, backoff_mm=0.5,
)


def rig(**kw):
    p = MockMoonraker(z=100.0, speed_factor=0)
    lc = MockLoadCell(printer=p, noise_g=0.0, **kw)
    return p, lc


def test_probe_descends_and_stops_near_the_surface():
    p, lc = rig(contact_z=20.0)
    r = calibrate_z(p, lc, **ARGS)
    # overshoot is bounded by the settle readings plus one step, no more
    bound = (ARGS["settle_readings"] + 4) * ARGS["step_mm"]
    assert 20.0 - bound <= r.stepped_z <= 20.0 + 1e-6
    # interpolated to the 1.1 N crossing: stiffness 600 g/mm -> 1.1 N is 112.2 g -> 0.187 mm into the swab
    assert r.contact_z == pytest.approx(20.0 - 1.1 / 0.00980665 / 600.0, abs=0.01)
    assert r.force_n >= ARGS["target_force_n"]
    assert r.steps > 0


def test_scribe_z_adds_the_surface_offset():
    p, lc = rig(contact_z=20.0)
    r = calibrate_z(p, lc, **ARGS)
    assert r.scribe_z(0.4) == pytest.approx(r.contact_z + 0.4)


def test_no_swab_reads_as_a_failed_grip_with_no_extra_sensor():
    p, lc = rig(contact_z=20.0)
    lc.has_swab = False
    with pytest.raises(GripFailed):
        calibrate_z(p, lc, **ARGS)
    assert p.z == 25.0, "must retreat to the safe Z before giving up"


def test_surface_out_of_reach_is_a_failed_grip_not_a_crash():
    p, lc = rig(contact_z=1.0)   # further than max_travel from the start
    with pytest.raises(GripFailed):
        calibrate_z(p, lc, **ARGS)


def test_a_stiff_contact_aborts_to_protect_the_cell():
    p, lc = rig(contact_z=20.0, stiffness_g_per_mm=100000.0)
    with pytest.raises(OverForce):
        calibrate_z(p, lc, **ARGS)
    assert p.z == 25.0


def test_abort_force_must_exceed_target():
    p, lc = rig()
    bad = dict(ARGS, abort_force_n=1.1)
    with pytest.raises(ValueError):
        calibrate_z(p, lc, **bad)


def test_backwards_cell_stops_the_descent_early_with_a_clear_message():
    p, lc = rig(contact_z=20.0)
    lc.stiffness_g_per_mm = -600.0
    with pytest.raises(GripFailed, match="sign is backwards"):
        calibrate_z(p, lc, **ARGS)
    assert p.z == 25.0
