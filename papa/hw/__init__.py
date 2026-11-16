"""Hardware interfaces, with a single place that decides real or mock.

Selection is per device, so the four can be mixed. `PAPA_SIM=1` mocks
everything; `PAPA_REAL` names the exceptions. The combination that matters
before anything is wired:

    PAPA_SIM=1 PAPA_REAL=printer python -m apps.server

That drives a real printer over the network from a laptop while the load cell,
gripper and camera stay mocked, because those three live on GPIO pins a laptop
does not have. Nothing else in the codebase checks for simulation, so the state
machine cannot behave differently in a dry run.
"""
from __future__ import annotations

from typing import Any

from papa.config import Calibration, Machine, is_real


def make_printer(machine: Machine) -> Any:
    if not is_real("printer"):
        from papa.sim.mocks import MockMoonraker

        return MockMoonraker()
    from papa.hw.moonraker import Moonraker

    return Moonraker(machine.moonraker_url, machine.timeout_s)


def make_loadcell(machine: Machine, cal: Calibration, printer: Any) -> Any:
    if not is_real("loadcell"):
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
    if not is_real("gripper"):
        from papa.sim.mocks import MockGripper

        return MockGripper(open_us=cal.gripper["open_us"], closed_us=cal.gripper["closed_us"])
    from papa.hw.gripper import ServoGripper

    g = cal.gripper
    return ServoGripper(
        machine.pins["servo_pwm"], g["open_us"], g["closed_us"],
        calibrated=bool(g.get("open_set")) and bool(g.get("closed_set")),
    )


def _sim_arrays() -> tuple:
    """Four electrode arrays near the simulated slide slots, a little turned."""
    from papa.sim.arrays import SimArray

    return (SimArray((155.0, 235.0), 12.0, 10.0, 0.0), SimArray((216.0, 236.0), 12.0, 10.0, 1.5),
            SimArray((155.0, 176.0), 12.0, 10.0, -2.0), SimArray((215.0, 174.5), 12.0, 10.0, 0.8))


def _sim_printer_axes() -> Any:
    """Where the simulated arrays physically are, seen by the simulated camera:
    a fixed camera-to-printer geometry (0.2 mm per plate pixel, turned 1 deg),
    independent of what has been taught, like a real camera."""
    import numpy as np

    from papa.vision.correction import PrinterAxes

    plate = np.array([[300.0, 200.0], [2000.0, 220.0], [1980.0, 1150.0]])
    th = np.radians(1.0)
    A = 0.2 * np.array([[np.cos(th), np.sin(th)], [np.sin(th), -np.cos(th)]])
    return PrinterAxes(plate, plate @ A.T + [0.0, 330.0])


def make_camera(machine: Machine | None = None) -> Any:
    if not is_real("camera"):
        import os

        from papa.sim.mocks import MockCamera

        if os.environ.get("PAPA_SIM_ARRAYS") != "1":
            return MockCamera()
        return MockCamera(arrays=_sim_arrays(), axes=machine.axes if machine is not None else None,
                          printer_axes=_sim_printer_axes)
    from papa.hw.camera import PiCamera

    size = (machine.raw.get("camera_size") if machine is not None else None) or [1920, 1080]
    return PiCamera((int(size[0]), int(size[1])))
