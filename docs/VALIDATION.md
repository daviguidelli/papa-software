# Validation plan

Goal: go from "nothing is wired" to "one slide scribed" without ever debugging
two unknowns at once.

**Rules**
1. One subsystem at a time, in the order below. Each stage has a pass line; do
   not move on until it passes.
2. Change one variable per test. Write down the force, speed and swab with every
   result.
3. Each bench script prints PASS/FAIL and saves a JSON log to `var/bench/`. Keep
   the logs; they are your lab notebook.
4. Everything runs in simulation first: `PAPA_SIM=1 python -m scripts.bench.<name>`.

Run from the repo root. On a laptop only the gantry stage applies
(`PAPA_SIM=1 PAPA_REAL=printer`); the other three need the Pi.

---

## Stage 0 — decisions that change the numbers (before any wiring)

| Question | Why it blocks | Where to look |
| --- | --- | --- |
| What is the TAL221 rating? | A 100 g cell saturates at 0.98 N, below the 1.1 N working force | Label on the cell |
| What force does a scribe need? | The config says 5 gf target / 20 gf abort; 1.1 N is 112 gf | `docs/EXPERIMENTS.md` §2 |
| Is the SV08 running stock Klipper + Moonraker? | The whole motion layer assumes it | `curl http://<ip>:7125/printer/info` |
| Printer IP, and the Pi's fixed IP | Both ends need fixed addresses on a direct cable | `deploy/setup-pi.md` |

**Gate:** if the cell is 100 g, either the working force changes or the cell
does. Settle this first; the abort force in `machine.yaml` has no meaning until
the rating is known.

---

## Stage 1 — Gantry (printer + network only)

`python -m scripts.bench.b1_gantry [--ruler]`

Checks: link, Klipper ready, homing, ±1 mm on each axis reads back, an
out-of-range move is refused, 10 round trips return to the same commanded
point, and a move blocks until finished.

**Pass:** all checks green. With `--ruler`, a commanded 50 mm move measures
50 ±0.5 mm.

Klipper reports the *commanded* position. The script cannot prove the carriage
physically moved; the ruler check can.

| Failure | Likely cause |
| --- | --- |
| No answer | Wrong IP, cable, or different subnet |
| Answers, moves refused | Client missing from Moonraker `trusted_clients` |
| Reads back wrong by a constant factor | Steps per mm / microsteps in the Klipper config |
| Axis moves the wrong way | `axes:` mapping in `machine.yaml` |

---

## Stage 2 — Load cell (Pi + HX711, no motion)

`python -m scripts.bench.b2_loadcell --rating-g <label> [--known-g 50]`

Checks: rating against the working force, reads arrive, rest noise < 0.3 g,
drift < 0.3 g over 30 s, and with a known weight the scale reads within 2 %.

**Pass:** all green. Then copy `scale_counts_per_g` and `tare_counts` into
`calibration.yaml`.

Add weights gently. The cell is easily overloaded, and the HX711 reading past
the rating is neither linear nor safe for the part.

| Failure | Likely cause |
| --- | --- |
| "HX711 not ready" | DAT/CLK swapped (GPIO5/6), no 5 V, bad ground |
| Noise high | Long unshielded leads, a loose cell mount, vibration from the frame |
| Drift | Warm-up (wait 5 min), a mount that flexes |

---

## Stage 3 — Servo gripper (Pi + servo + its own pack)

`python -m scripts.bench.b3_servo --find`, then `--cycles 20`

First run with no swab and the jaws clear. `--find` lets you type pulse widths
from 1500 µs in 50 µs steps until you know where open and closed are; stop the
moment it hums or binds. Put the values in `calibration.yaml`.

**Pass:** smooth motion, no jitter at rest, a swab is held when you pull on it,
and the 4xAA pack stays above 4.8 V after 20 cycles. The pack and the Pi must
share a ground.

---

## Stage 4 — Camera (Pi + HQ Camera mounted)

`python -m scripts.bench.b4_camera [--sample x,y]`

Saves a frame and a blue mask to `var/bench/`. Copy them to the laptop with
`scp` and look at them.

**Pass:**
- exposure mean 60–190, under 2 % blown-out pixels (glare on the glass)
- frame-to-frame difference under 2, so lighting is stable
- the 2.54 mm swab stem spans at least ~10 px
- `--sample` on a real film and a real stem gives hue bands that separate them

The HQ Camera has no autofocus: set focus on the lens ring with a slide in place, then lock the thumbscrew. Lens choice (6 mm or 16 mm) and mounting height set the field of view, so work out pixels-per-mm before fixing the mount.

Fix focus first, then white balance, then exposure; lock all three. Detection
thresholds only mean something under those exact settings.

---

## Stage 5 — Combined (gantry + load cell + gripper)

Only after stages 1–3 pass.

`python -m scripts.bench.b5_combined --cell-x X --cell-y Y --z-safe Z --runs 10`

Load a swab, then probe the cell ten times. `--z-safe` is a height just above
the cell, so the descent has no chance of crossing the bed.

**Pass:** contact height spread ≤ one probe step (0.05 mm). This is the number
that decides scribe depth, so it is worth getting right before anything touches
glass. Record the force at contact too: it tells you how many grams one step
costs.

---

## Stage 6 — Camera ↔ gantry agreement

Not scripted yet. Put the swab tip over a mark on the bed using the Jog page,
read the machine position, and click the same mark in the camera image. Repeat
for four corners; those pairs are the homography. Then check a fifth point that
was not used. Pass: error under 0.5 mm.

## Stage 7 — First scribing

1. Dry run in air: `python -m papa.cli dry-run examples/sample.dxf --slides 1`, no swab, check the G-code.
2. One swab, one sacrificial slide, a short straight line.
3. Kerf comb (`docs/EXPERIMENTS.md` §1).
4. The real pattern, only after the kerf is known.

---

## Variables, and when each stops being unknown

| Variable | Stage | Stored in |
| --- | --- | --- |
| Moonraker URL, axes mapping | 1 | `machine.yaml` |
| Load cell scale and tare | 2 | `calibration.yaml` |
| Cell rating, working force, abort force | 0, 2 | `machine.yaml` |
| Servo open/closed pulse | 3 | `calibration.yaml` |
| Camera focus, exposure, white balance | 4 | to be added |
| HSV bands for film and stem | 4 | `calibration.yaml` |
| Probe step and repeatability | 5 | `machine.yaml` |
| Camera → machine homography | 6 | `calibration.yaml` |
| Surface offset | 5–7 | `calibration.yaml` |
| Kerf, scribe speed, swab wear | 7 | `calibration.yaml`, `machine.yaml` |
