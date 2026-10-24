# PAPA software

Control software for **PAPA**, an automated subtractive thin-film patterning
platform built on a Sovol SV08 frame. It scribes a DXF pattern, scaled into a
square on each slide, off blue-film glass slides using cotton swabs.

AME 40463 Senior Design · Fall 2026 · Group 01

## How the two boards divide the work

The Raspberry Pi runs the only operation script. The SV08 mainboard never runs
our code: it is a **motion server** the Pi commands over Ethernet through
Klipper's Moonraker API. That is what lets one script own the whole sequence
even though motion and sensing live on different boards.

| Owner | Responsibilities |
| --- | --- |
| Raspberry Pi 4 | Orchestration and state machine, DXF to toolpath, camera and detection, load cell, gripper servo, the GUI, calibration and job files |
| SV08 mainboard | X, Y, Z1–Z4 motion, homing, speed and acceleration limits, G-code execution, reporting position |
| Link | Ethernet, Moonraker HTTP on port 7125. Every move script ends with `M400`, so a call returns only once motion has finished |

Two setup details, both outside this repo: Moonraker must list the Pi under
`trusted_clients`, and a direct Pi-to-mainboard cable has no router, so both
ends need fixed IPs on the same subnet.

## Run it without a printer

Nothing here needs the machine to develop against. `PAPA_SIM=1` swaps every
hardware driver for a mock and the state machine runs unchanged — the Z probe
really descends and really stops, and every G-code line is logged.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                  # 34 tests, no hardware
PAPA_SIM=1 python -m apps.server        # GUI + API on :8080
```

On the Pi, use `requirements-pi.txt` (adds picamera2 and lgpio) and drop
`PAPA_SIM`.

## Layout

```
papa/
  config.py          machine.yaml + calibration.yaml loading, limits, clamping
  hw/                moonraker, loadcell (HX711), gripper (servo), camera
                     __init__.py decides real vs mock — the only such check
  vision/presence.py blue filter: film in a slot, stem in a rack position
  toolpath/          dxf_loader -> fit -> blocks
  ops/               zcal, swabs (pick rule + wear), jobs, machine (the loop)
  sim/mocks.py       mock Moonraker, load cell, gripper, camera
apps/
  server.py          one FastAPI app behind all three GUI tabs
  static/            the exported GUI
config/              machine.yaml, calibration.example.yaml
tests/               runs entirely in simulation
examples/            sample DXF generator
```

## Two config files, on purpose

`machine.yaml` is set once by whoever builds the machine: Moonraker URL, axis
mapping, limits, speeds, probe force and wear limits, GPIO pins.

`calibration.yaml` is written by the Calibration tab and is the only file an
operator ever changes: substrate size, the pattern square, slot and rack
counts, every taught position, the load cell scale, and the blue thresholds.
Counts and dimensions are data, so a different mounting base is a config
change rather than a code change.

## Coordinate frames

- **Machine frame** — Klipper absolute XYZ in mm, valid after homing.
- **Slide-local frame** — origin at the slide's top-left corner, +x right,
  +y down. The toolpath is computed once here; each slot's taught origin turns
  it into machine coordinates at run time, so no per-slide G-code exists.

Which machine axis local +y runs along lives in `machine.yaml` under `axes`,
so a rotated mounting base needs no code change.

## The loop

```
preflight -> pick swab -> grip -> Z calibrate -> scribe block
          -> worn? -> (dispose, pick, grip, Z calibrate, resume with overlap)
          -> slide done? -> dispose -> next slide
```

Blocks are streamed rather than uploaded as one file, for two reasons: wear is
checked between blocks, and the index of the last finished block is the resume
point. State is saved after **every** block, so a pause, a crash or a power cut
resumes in the right place.

Z calibration runs after every swab pickup, because each swab sits slightly
differently in the gripper. It descends in small steps and reads force between
them — the HX711 gives only 10–80 samples/s, so watching force during a
continuous descent would find contact too late. No force within the maximum
travel means the swab is missing or the grip failed, which gives grip detection
with no extra sensor.

## The swab rack

Two columns by N rows, left of the bed. A left-side swab is pickable only once
the right-side swab in that row is gone (used, or never loaded). That single
rule gives the right-first order, avoids the interference, and still works on a
partly loaded rack. The camera reports which positions hold a swab.

## Detection

The camera answers one question: is blue present where it should be? Filtering
is done in HSV, not raw RGB — a hue range survives the lights brightening or
dimming, a fixed RGB cutoff does not. Regions come from the taught origins, so
adding a slot or a swab needs no extra setup.

Two checks once the camera is mounted: the 2.54 mm stem should span about ten
pixels or more, and the lighting should be fixed and glare-free, since glare on
glass shifts the blue.

## The GUI

One page, three tabs, one backend. Only **Operation** is meant for end users
(upload a DXF, confirm the scanned slides, run). **Calibration** teaches
positions and holds every tunable value. **Jog and test** is manual control and
hardware checks. Only one tab may own the hardware at a time: while a job runs,
the calibration and jog endpoints refuse.

## Status

Toolpath, config, detection, Z calibration, the swab rule and the full loop run
and are covered by tests in simulation. Hardware drivers are written but
untested against real hardware. Nothing has run on the printer yet.
