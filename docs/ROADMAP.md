# Roadmap

What is done, what is next, and what is still undecided.

## Done and tested in simulation

- Config loading, limits and clamping; substrate and counts as data
- DXF → chained strokes → uniform fit into the pattern square → resumable blocks
- Stepped Z probe on the load cell, with grip detection from "no force within
  max travel" and an abort force that protects the cell
- Swab pick rule (right before left within a row), wear tracking, mid-slide
  swaps, a fresh swab per slide
- Job file written after every block, so pause, crash and power loss resume
- Blue presence detection in HSV for films and stems
- The whole loop, end to end, with no printer: 34 tests plus a CLI dry run

## Done: the Jog and test page

`apps/static/index.html` is real and wired to the API, verified in a browser
against the mock. Per-device selection (`PAPA_REAL`) means it can drive a real
printer from a laptop with everything else mocked.

## Next: validate against the real printer

One line changes: `moonraker_url` in `config/machine.yaml`. Then home, jog 1 mm,
and check the readout against a ruler. If `/printer/info` answers but moves are
refused, it is almost always `trusted_clients`.

## Next: the Calibration and Operation tabs

Same page, two more panels. The endpoints already exist; the prototype has the
interaction worked out.

## Next: the Pi bench, one device at a time

See `deploy/setup-pi.md`. Order matters: Moonraker reachable → jog → gripper →
load cell → Z probe → camera. Each step is checkable on its own, and each one
is a place where a wiring mistake is cheap to find.

## Then: the printer

Dry run in air with no swab, then one swab on a sacrificial slide. Everything
before that point should already have been proven in simulation or on the
bench.

## Values that need experiments, not opinions

| Value | Where | How to find it |
| --- | --- | --- |
| `scribe.kerf_mm` | calibration.yaml | Comb of line pairs at 2.0-0.1 mm gaps, at the working force. See EXPERIMENTS.md |
| `wear.max_path_mm`, `max_contact_s` | machine.yaml | Scribe until the line quality drops, on a sacrificial slide |
| `zcal.target_force_g` | machine.yaml | Lowest force that removes film cleanly |
| `zcal.abort_force_g` | machine.yaml | Well under the cell's rating — confirm the part first |
| `slide_surface_offset_mm` | calibration.yaml | Measure load cell top vs slide top |
| `detection.*_hsv`, blue fractions | calibration.yaml | Sample real film and stem under the final lighting |
| `speeds_mm_s.scribe` | machine.yaml | Fast enough to be useful, slow enough to cut cleanly |

## Still open

- Aspect ratio when a DXF is not square: currently uniform scale, longest side
  fills the square, centred. Fine unless the pattern must land at exact
  coordinates on the slide.
- How many grip failures before the operator is asked (currently 3).
- Whether a soft stop should finish the current slide or stop at the block.
- Whether a misaligned slide should block the job or just warn.
