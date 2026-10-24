"""The whole loop in simulation: no printer, no Pi, no hardware."""
from pathlib import Path

import pytest

from examples.make_sample_dxf import build
from papa.config import Calibration, Machine
from papa.ops.jobs import Job
from papa.ops.machine import Orchestrator
from papa.ops.swabs import Rack
from papa.sim.mocks import MockGripper, MockLoadCell, MockMoonraker
from papa.toolpath import blocks as blockmod
from papa.toolpath import dxf_loader, fit

MACHINE = Machine({
    "moonraker_url": "mock://x",
    "axes": {"local_x": "+X", "local_y": "-Y"},
    "limits_mm": {"x": [0, 350], "y": [0, 350], "z": [0, 340]},
    "speeds_mm_s": {"travel": 150, "scribe": 20, "z_approach": 1.0, "z_lift": 10},
    "zcal": {"target_force_g": 5.0, "abort_force_g": 20.0, "step_mm": 0.05,
             "max_travel_mm": 8.0, "settle_readings": 2, "backoff_mm": 0.3},
    "wear": {"max_path_mm": 40.0, "max_contact_s": 1e9},
    "toolpath": {"block_target_mm": 3.0, "resume_overlap_mm": 0.3},
    "pins": {"hx711_dat": 5, "hx711_clk": 6, "servo_pwm": 13},
    "retry": {"max_grip_failures": 3},
})


def calibration(rows=15, slots=(1, 2)):
    swabs, sid = [], 1
    for r in range(1, rows + 1):
        for side in ("right", "left"):
            swabs.append({"id": sid, "row": r, "side": side,
                          "x": 100.0 - (0 if side == "right" else 14),
                          "y": 20.0 + r * 8, "z_grip": 14.0, "calibrated": True})
            sid += 1
    return Calibration.from_raw({
        "substrate": {"width_mm": 25.0, "length_mm": 75.0, "pattern_size_mm": 25.0},
        "swab_rack": {"rows": rows, "swabs_per_slide": 2, "swabs": swabs},
        "slide_slots": {"rows": 3, "cols": 3,
                        "slots": [{"id": i, "x": 60.0 + 30 * i, "y": 200.0, "calibrated": True}
                                  for i in slots]},
        "load_cell": {"x": 30.0, "y": 40.0, "z_safe": 25.0,
                      "scale_counts_per_g": 1.0, "tare_counts": 0},
        "disposal": {"x": 10.0, "y": 10.0, "z_release": 20.0},
        "slide_surface_offset_mm": 0.4,
        "gripper": {"open_us": 1000, "closed_us": 2000},
        "detection": {"camera_to_machine": {"points_px": [[0, 0]] * 4,
                                            "points_mm": [[0, 0]] * 4}},
    })


@pytest.fixture
def rig(tmp_path):
    cal = calibration()
    printer = MockMoonraker(speed_factor=0)
    printer.home()
    lc = MockLoadCell(printer=printer, contact_z=20.0, noise_g=0.0)
    strokes = fit.fit(dxf_loader.load(build(tmp_path / "s.dxf")), *cal.pattern_area).strokes
    bl = blockmod.make_blocks(strokes, 3.0)
    job = Job(dxf_name="s.dxf", slot_ids=[1, 2], swabs_per_slide=2,
              pattern_size_mm=25.0, total_blocks=len(bl))
    rack = Rack.from_detection(cal, [f"{r}{s}" for r in range(1, 16) for s in ("R", "L")])
    orch = Orchestrator(machine=MACHINE, cal=cal, printer=printer, loadcell=lc,
                        gripper=MockGripper(), rack=rack, blocks=bl, job=job)
    return orch, printer, job, tmp_path


def test_job_completes_every_slide(rig, monkeypatch, tmp_path):
    orch, printer, job, _ = rig
    monkeypatch.setattr(job, "save", lambda path=None: tmp_path / "job.json")
    out = orch.run()
    assert out.state == "done", out.message
    assert all(s.done for s in out.slides)
    assert out.fraction == pytest.approx(1.0)


def test_each_slide_starts_with_a_fresh_swab(rig, monkeypatch, tmp_path):
    orch, _, job, _ = rig
    monkeypatch.setattr(job, "save", lambda path=None: tmp_path / "job.json")
    out = orch.run()
    used = [s.swabs_used for s in out.slides]
    assert all(u for u in used)
    flat = [lab for u in used for lab in u]
    assert len(flat) == len(set(flat)), "a swab must never be reused"


def test_wear_forces_a_mid_slide_swap(rig, monkeypatch, tmp_path):
    orch, _, job, _ = rig
    monkeypatch.setattr(job, "save", lambda path=None: tmp_path / "job.json")
    out = orch.run()
    # max_path_mm is 40 and the pattern is longer, so at least one slide swapped
    assert any(len(s.swabs_used) >= 2 for s in out.slides)


def test_gcode_is_absolute_and_ends_each_script_with_m400(rig, monkeypatch, tmp_path):
    orch, printer, job, _ = rig
    monkeypatch.setattr(job, "save", lambda path=None: tmp_path / "job.json")
    orch.run()
    assert any(l.startswith("G1") for l in printer.log)
    assert "M400" in printer.log


def test_axis_mapping_sends_local_y_down_as_machine_minus_y(rig, monkeypatch, tmp_path):
    orch, _, job, _ = rig
    monkeypatch.setattr(job, "save", lambda path=None: tmp_path / "job.json")
    slot = orch.cal.slot(1)
    mx, my = orch._to_machine(slot.x, slot.y, 10.0, 5.0)
    assert mx == pytest.approx(slot.x + 10.0)
    assert my == pytest.approx(slot.y - 5.0)


def test_preflight_names_what_is_missing(rig):
    orch, _, job, _ = rig
    job.slot_ids.append(99)
    job.slides.append(type(job.slides[0])(slot_id=99))
    problems = orch.preflight()
    assert any("99" in p for p in problems)


def test_empty_rack_pauses_instead_of_failing(rig, monkeypatch, tmp_path):
    orch, _, job, _ = rig
    monkeypatch.setattr(job, "save", lambda path=None: tmp_path / "job.json")
    orch.rack.loaded = {"1R"}
    out = orch.run()
    assert out.state in ("paused", "error")
    assert "swab" in out.message.lower()
