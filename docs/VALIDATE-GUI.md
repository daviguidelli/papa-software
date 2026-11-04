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

## Camera tab (live footage)

Shows the Pi camera live in the browser. The stream (MJPEG) only runs while the tab is open.

* **Pi-served page** (`http://<pi-ip>:8080/validate`): the camera is read directly on the Pi. Chip says REAL.
* **Laptop-served page** (`PAPA_SIM=1 PAPA_REAL=printer python -m apps.server`): the camera is mocked unless you point it at the Pi:
  `PAPA_CAMERA_URL=http://<pi-ip>:8080 PAPA_SIM=1 PAPA_REAL=printer python -m apps.server`
  The laptop then proxies the Pi's stream and the chip says REMOTE. If the Pi is down you get a clear 502 message, not a hang.
* A stamp burned into the frame shows REAL/MOCK/REMOTE, the time and the source frame size, so a frozen or fake picture is obvious.
* Controls: width, frames/s, thirds grid with centre cross, pause/resume, **Save snapshot** (1920 px JPEG).
* Max 3 simultaneous streams; the camera sits behind a lock so a stream and a snapshot never touch it at once.
* Optional `camera_size: [w, h]` in `machine.yaml` sets the capture size (default 1920x1080). The teammate scripts (`camCal`, `substrateRecognition`) use the full 4056x3040 4:3 sensor, so the framing differs. Align the two before trusting pixel coordinates from one in the other.

## Hold-to-jog

Sidebar, **Hold** mode: press and keep a button down and that axis keeps moving at the XY or Z mm/s in the boxes; let go and it stops. One axis at a time.

* The page sends a heartbeat every 100 ms. If it stops (button released, tab hidden, Wi-Fi drops) the server stops within 0.35 s. At most about a quarter second of motion is still queued in Klipper when that happens, so at 4 mm/s Z coasts roughly 1 mm. Use a low Z speed near anything hard.
* Caps: XY 100 mm/s, Z 10 mm/s. Z down is disabled until a floor is set, and a held Z stops short of the floor with a message. Axis limits stop it too.
* While a button is held, step jogs and other motion are refused (409).
* E-STOP also cancels a hold.

## Servo: fine calibration

* Window: manual pulses can only go inside the **servo window**, which starts at 1300 to 1700 µs. Widen it in small steps (a confirmation appears).
* Speed: moves ramp at 20, 50, 100 or 250 µs per second (or jump, the job speed). A 1 µs nudge at 50 µs/s takes 20 ms; a 100 µs move takes 2 s.
* Nudges: ±1, 2, 5, 10, 25 µs.
* The first command after power-up cannot be ramped, because the horn's resting spot is unknown. It jumps. Do it with empty, clear jaws, near the middle of the window.
* OPEN and CLOSED stay disabled, in the GUI and in jobs on the real servo, until **both** have been locked on this tab (`open_set` and `closed_set` in calibration.yaml).
* Procedure: find the fully open value, lock it. Find the value where a swab is held firmly without crushing it, lock it. Then Test OPEN and Test CLOSED.

## Load cell probe: start Z, speed, step

The Probe card has **Start Z**, **Descent mm/s** and **Step mm**. Order: rise to start Z if below it, travel over the cell at that height, settle onto the start Z, tare, then step down. Z never moves at travel speed. Start Z must be inside the machine's Z limits and not below the Z floor. Limits: 0.05 to 10 mm/s, 0.01 to 1 mm.

## Substrate detection (Camera tab)

**Detect substrates** takes one frame and runs the team's red-tape method (`substrateRecognition`), ported to `papa/vision/substrates.py`: red HSV threshold, clean up, circle each marker, pair upper-right with lower-left markers, number the pairs in rows (S1, S2, ...). The result is drawn on the picture and listed in a table in pixels.

* Enter the real **marker spacing** in pixels (X and Y) to stop neighbours being paired by mistake. Without it the shortest valid pair wins, which fails when substrates sit close together.
* Settings written for the 4056 px wide stills are rescaled to the frame being analysed.
* **Save scan** writes `var/scans/scan_NNNN/` with `original.jpg`, `detected.jpg`, `substrates.json`, like the script does.
* Output is pixels only. Nothing yet converts pixels to printer X/Y.
* The load cell sampler reads at 20 samples per second by default, only while the Load cell tab is open.
