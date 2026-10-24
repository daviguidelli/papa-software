"""Command line entry points, all usable with no printer attached.

    python -m papa.cli preview examples/sample.dxf
    python -m papa.cli dry-run examples/sample.dxf --slides 1,2
    python -m papa.cli preflight --slides 1,2
    python -m papa.cli gcode examples/sample.dxf --slide 1 --out var/job.gcode

`dry-run` and `gcode` run the real state machine against the mocks, so the
G-code they produce is the G-code the printer would receive.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PAPA_SIM", "1")

from papa import config as cfgmod            # noqa: E402
from papa.hw import make_gripper, make_loadcell, make_printer  # noqa: E402
from papa.ops.jobs import STATE_DIR, Job     # noqa: E402
from papa.ops.machine import Orchestrator, Status  # noqa: E402
from papa.ops.swabs import Rack              # noqa: E402
from papa.toolpath import blocks as blockmod  # noqa: E402
from papa.toolpath import dxf_loader, fit     # noqa: E402


def _ids(text: str) -> list[int]:
    return [int(t) for t in text.split(",") if t.strip()]


def _build(dxf: str, cal) -> tuple[list, list]:
    strokes = fit.fit(dxf_loader.load(dxf), *cal.pattern_area).strokes
    machine = cfgmod.load_machine()
    bl = blockmod.make_blocks(strokes, machine.toolpath.get("block_target_mm", 3.0))
    return strokes, bl


def _orchestrator(dxf: str, slides: list[int]) -> Orchestrator:
    machine = cfgmod.load_machine()
    cal = cfgmod.load_calibration()
    printer = make_printer(machine)
    printer.home()
    loadcell = make_loadcell(machine, cal, printer)
    if hasattr(loadcell, "contact_z"):
        loadcell.contact_z = cal.load_cell["z_safe"] - 5.0
    gripper = make_gripper(machine, cal)
    strokes, bl = _build(dxf, cal)
    labels = [f"{r}{s}" for r in range(1, cal.rack_rows + 1) for s in ("R", "L")]
    job = Job(
        dxf_name=Path(dxf).name, slot_ids=slides,
        swabs_per_slide=cal.swabs_per_slide,
        pattern_size_mm=cal.pattern_area[0], total_blocks=len(bl),
    )

    def show(s: Status) -> None:
        print(
            f"  [{s.state:>7}] {s.stage:<14} slot {s.slot_id} swab {s.swab or '-':<4}"
            f" scribe {s.scribe_fraction * 100:5.1f}%  wear {s.wear_fraction * 100:5.1f}%"
            f"  overall {s.overall_fraction * 100:5.1f}%",
            end="\r",
        )

    return Orchestrator(
        machine=machine, cal=cal, printer=printer, loadcell=loadcell,
        gripper=gripper, rack=Rack.from_detection(cal, labels),
        blocks=bl, job=job, on_status=show,
    )


def cmd_preview(args: argparse.Namespace) -> int:
    cal = cfgmod.load_calibration()
    strokes, bl = _build(args.dxf, cal)
    w, h = cal.pattern_area
    print(f"{args.dxf}: {len(strokes)} strokes, {len(bl)} blocks")
    print(f"pattern square {w} x {h} mm, total path "
          f"{sum(s.length for s in strokes):.1f} mm")
    out = Path(args.out or STATE_DIR / "preview.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fit.to_svg(strokes, w, h))
    print(f"preview written to {out}")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    orch = _orchestrator(args.dxf, _ids(args.slides))
    problems = orch.preflight()
    if problems:
        print("preflight failed:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("preflight passed")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    orch = _orchestrator(args.dxf, _ids(args.slides))
    print(f"dry run: {len(orch.blocks)} blocks x {len(orch.job.slot_ids)} slides")
    job = orch.run()
    print()
    print(f"state: {job.state}  {job.message}")
    for s in job.slides:
        print(f"  slot {s.slot_id}: done={s.done} swabs={s.swabs_used}")
    lines = getattr(orch.printer, "log", [])
    print(f"{len(lines)} G-code lines issued")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"G-code log written to {args.out}")
    return 0 if job.state == "done" else 1


def cmd_gcode(args: argparse.Namespace) -> int:
    args.slides = str(args.slide)
    args.out = args.out or str(STATE_DIR / "job.gcode")
    return cmd_dry_run(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="papa")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preview", help="parse a DXF and write an SVG preview")
    p.add_argument("dxf")
    p.add_argument("--out")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("preflight", help="check a job without moving anything")
    p.add_argument("dxf")
    p.add_argument("--slides", default="1")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("dry-run", help="run the whole loop against the mocks")
    p.add_argument("dxf")
    p.add_argument("--slides", default="1")
    p.add_argument("--out", help="write the G-code log here")
    p.set_defaults(func=cmd_dry_run)

    p = sub.add_parser("gcode", help="dump the G-code for one slide")
    p.add_argument("dxf")
    p.add_argument("--slide", type=int, default=1)
    p.add_argument("--out")
    p.set_defaults(func=cmd_gcode)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
