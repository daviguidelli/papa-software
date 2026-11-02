# Validation page

`/validate` tests one system at a time, before the full machine is integrated.
Same page on the Pi (all real) or on a laptop (`PAPA_SIM=1 PAPA_REAL=printer`,
load cell and servo mocked). The header says REAL or MOCK for each device, so a
mocked reading is never mistaken for hardware.

```bash
python -m apps.server          # Pi, then open http://<pi-ip>:8080/validate
PAPA_SIM=1 PAPA_REAL=printer python -m apps.server   # laptop, real printer only
```

## Safety, always on

- **One jog pad.** Every tab uses it. Arrows are machine axes.
- **Z floor.** Z cannot move down until the operator sets a floor, then never
  below it. Stored in `var/z_floor.json`. It wraps the printer object, so jobs
  obey it too, and job preflight fails while it is unset.
- **Z homing is locked.** `/api/machine/home` refuses Z or "all axes" unless the
  server starts with `PAPA_ALLOW_Z_HOME=1`. Home XY always works. Raw G-code
  through `printer.script()` is not inspected.

## Tabs and what they write to `calibration.yaml`

| Tab | Does | Writes |
| --- | --- | --- |
| Load cell | live force, tare, scale, capture XY and approximate start Z, one-shot probe (the same `calibrate_z` a job runs) | `load_cell.x/y/z_safe`, `xy_set`, `z_safe_set`, `tare_counts`, `scale_counts_per_g` |
| Glass offset | cell contact Z (probe) and glass contact Z (jogged), offset = glass - cell | `slide_surface_offset_mm` (scribe Z = contact Z + offset) |
| Servo | drive to a raw pulse width, lock OPEN and CLOSED, test both | `gripper.open_us`, `gripper.closed_us` |
| Slide | capture the top-left corner per slot, edit slide and pattern-square size, trace the square or outline in XY | `slide_slots.slots[]`, `substrate.*` |

The camera is deliberately absent. Saving a value never copies the simulation
calibration into `calibration.yaml`; with no real file yet, it starts from the
example file.

## Order that works

1. Home XY, jog down in small steps, set the Z floor.
2. Servo: no swab, jaws clear, find and lock OPEN and CLOSED.
3. Load cell: tare, scale, capture XY, set start Z, grip a swab, probe.
4. Glass offset: use the probe result, touch the glass, lock the offset.
5. Slide: capture the corner, set the size, trace the square.
