"""Hardware interfaces, with a single place that decides real or mock.

PAPA_SIM=1 swaps every driver. Nothing else in the codebase checks for
simulation, so the state machine cannot behave differently in a dry run.
"""
from __future__ import annotations

from typing import Any

from papa.config import Calibration, Machine, sim_mode


def make_printer(machine: Machine) -> Any:
    if sim_mode():
        from papa.sim.mocks import MockMoonraker

        return MockMoonraker()
    from papa.hw.moonraker import Moonraker

    return Moonraker(machine.moonraker_url, machine.timeout_s)


def make_loadcell(machine: Machine, cal: Calibration, printer: Any) -> Any:
    if sim_mode():
        from papa.sim.mocks import MockLoadCell

        return MockLoadCell(printer=printer)
    from papa.hw.loadcell import HX711LoadCell

    lc = cal.load_cell
    pins = machine.pins
    return HX711LoadCell(
        dat_pin=pins["hx711_dat"],
        clk_pin=pins["hx711_clk"],
        scale_counts_per_g=lc.get("scale_counts_per_g") or 1.0,
        tare_counts=int(lc.get("tare_counts") or 0),
    )


def make_gripper(machine: Machine, cal: Calibration) -> Any:
    if sim_mode():
        from papa.sim.mocks import MockGripper

        return MockGripper()
    from papa.hw.gripper import ServoGripper

    g = cal.gripper
    return ServoGripper(machine.pins["servo_pwm"], g["open_us"], g["closed_us"])


def make_camera() -> Any:
    if sim_mode():
        from papa.sim.mocks import MockCamera

        return MockCamera()
    from papa.hw.camera import PiCamera

    return PiCamera()
