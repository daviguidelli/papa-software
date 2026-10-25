"""One FastAPI app, three tabs: Operation, Calibration, Jog and test.

They share this backend, so no logic is duplicated, and only one of them can
own the hardware at a time: while a job runs, the calibration and jog
endpoints refuse, which is what stops someone jogging the gantry mid-scribe.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from papa import config as cfgmod
from papa.hw import make_camera, make_gripper, make_loadcell, make_printer
from papa.ops.jobs import STATE_DIR, Job
from papa.ops.machine import Orchestrator, Status
from papa.ops.swabs import Rack
from papa.toolpath import blocks as blockmod
from papa.toolpath import dxf_loader, fit, spacing as spacingmod

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="PAPA")


class Device:
    """One lazily constructed device.

    Lazy on purpose: a laptop has no HX711, and building every driver at
    startup would mean a missing load cell stops you jogging the gantry. Each
    device is built on first use and a failure is remembered, so the endpoints
    that need it return 503 with the reason while everything else keeps working.
    """

    def __init__(self, name: str, factory: Callable[[], Any]) -> None:
        self.name = name
        self._factory = factory
        self._obj: Any = None
        self.error: str | None = None

    @property
    def built(self) -> bool:
        return self._obj is not None

    def get(self) -> Any:
        if self._obj is None and self.error is None:
            try:
                self._obj = self._factory()
            except Exception as exc:                     # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
        if self.error is not None:
            raise HTTPException(503, f"{self.name} is unavailable ({self.error})")
        return self._obj

    def describe(self) -> dict:
        return {
            "real": cfgmod.is_real(self.name),
            "built": self.built,
            "error": self.error,
        }


class Hub:
    """Owns the hardware handles and the one running job."""

    def __init__(self) -> None:
        self.machine = cfgmod.load_machine()
        self.cal = cfgmod.load_calibration()
        self.devices = {
            "printer": Device("printer", lambda: make_printer(self.machine)),
            "loadcell": Device(
                "loadcell",
                lambda: make_loadcell(self.machine, self.cal, self.printer),
            ),
            "gripper": Device("gripper", lambda: make_gripper(self.machine, self.cal)),
            "camera": Device("camera", make_camera),
        }
        self.orch: Orchestrator | None = None
        self.thread: threading.Thread | None = None
        self.status = Status()
        self.strokes: list[Any] = []
        self.dxf_name = ""
        self.lock = threading.Lock()

    @property
    def printer(self) -> Any:
        return self.devices["printer"].get()

    @property
    def loadcell(self) -> Any:
        return self.devices["loadcell"].get()

    @property
    def gripper(self) -> Any:
        return self.devices["gripper"].get()

    @property
    def camera(self) -> Any:
        return self.devices["camera"].get()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def require_idle(self) -> None:
        if self.busy:
            raise HTTPException(409, "a job is running; the hardware is busy")

    def reload_calibration(self) -> None:
        self.cal = cfgmod.load_calibration()


hub = Hub()


# ---------- pages ----------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/devices")
def devices() -> dict:
    """What this process is actually talking to, and whether it answers.

    The page shows this, so there is never any doubt about whether a reading
    came from hardware or from a mock.
    """
    reachable, url, detail = False, "", None
    try:
        reachable = hub.printer.ping()          # builds it, so describe() comes after
        url = getattr(hub.printer, "base_url", "")
    except HTTPException as exc:
        detail = str(exc.detail)
    out = {name: d.describe() for name, d in hub.devices.items()}
    out["printer"].update(reachable=reachable, url=url)
    if detail:
        out["printer"]["detail"] = detail
    return {"sim": cfgmod.sim_mode(), "devices": out,
            "moonraker_url": hub.machine.moonraker_url,
            "zcal": hub.machine.zcal, "speeds": hub.machine.speeds,
            "limits": hub.machine.limits}


# ---------- operation ----------

@app.post("/api/pattern")
async def upload_pattern(file: UploadFile) -> dict:
    hub.require_idle()
    tmp = STATE_DIR / "upload.dxf"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())
    try:
        strokes = dxf_loader.load(tmp)
    except dxf_loader.UnsupportedEntity as exc:
        raise HTTPException(400, str(exc)) from exc
    w, h = hub.cal.pattern_area
    result = fit.fit(strokes, w, h)
    hub.strokes = result.strokes
    hub.dxf_name = file.filename or "pattern.dxf"
    return {
        "name": hub.dxf_name,
        "strokes": len(result.strokes),
        "scale": result.scale,
        "source_size_mm": result.source_size,
        "fitted_size_mm": result.fitted_size,
        "area_mm": [w, h],
        "svg": fit.to_svg(result.strokes, w, h),
        "spacing": spacingmod.report(result.strokes, hub.cal.kerf_mm),
    }


@app.get("/api/scan")
def scan() -> dict:
    """One still frame, filtered for blue: which slots hold a film and which
    rack positions hold a swab."""
    hub.require_idle()
    from papa.vision import presence

    frame = hub.camera.frame()
    det = hub.cal.detection
    try:
        cmap = presence.CameraMap.from_cfg(det)
    except ValueError as exc:
        raise HTTPException(400, f"camera is not mapped to machine coordinates: {exc}") from exc

    film = presence.HsvBand.from_cfg(det.get("film_hsv", {}))
    stem = presence.HsvBand.from_cfg(det.get("stem_hsv", det.get("film_hsv", {})))
    w, h = hub.cal.pattern_area

    slots = []
    for slot in hub.cal.slots:
        d = presence.slide_present(
            frame, cmap, (slot.x, slot.y), w, h, film,
            min_fraction=float(det.get("slide_min_blue_fraction", 0.35)),
            misaligned_fraction=float(det.get("slide_misaligned_fraction", 0.12)),
        )
        slots.append({"id": slot.id, "present": d.present, "fraction": round(d.fraction, 3),
                      "misaligned": d.misaligned, "calibrated": slot.calibrated})

    swabs = []
    for s in hub.cal.swabs:
        d = presence.swab_present(
            frame, cmap, (s.x, s.y), float(det.get("swab_roi_mm", 4.0)), stem,
            min_fraction=float(det.get("swab_min_blue_fraction", 0.30)),
        )
        swabs.append({"id": s.id, "label": s.label, "present": d.present,
                      "fraction": round(d.fraction, 3), "calibrated": s.calibrated})
    return {"slots": slots, "swabs": swabs}


@app.post("/api/job")
def start_job(payload: dict) -> dict:
    hub.require_idle()
    if not hub.strokes:
        raise HTTPException(400, "upload a DXF first")
    slot_ids = [int(i) for i in payload.get("slot_ids", [])]
    if not slot_ids:
        raise HTTPException(400, "select at least one slide")
    per = int(payload.get("swabs_per_slide", hub.cal.swabs_per_slide))
    present = [s["label"] for s in scan()["swabs"] if s["present"]] if payload.get("rescan") \
        else [s.label for s in hub.cal.swabs]

    bl = blockmod.make_blocks(hub.strokes, hub.machine.toolpath.get("block_target_mm", 3.0))
    job = Job(
        dxf_name=hub.dxf_name, slot_ids=slot_ids, swabs_per_slide=per,
        pattern_size_mm=hub.cal.pattern_area[0], total_blocks=len(bl),
    )
    orch = Orchestrator(
        machine=hub.machine, cal=hub.cal, printer=hub.printer, loadcell=hub.loadcell,
        gripper=hub.gripper, rack=Rack.from_detection(hub.cal, present),
        blocks=bl, job=job, on_status=lambda s: setattr(hub, "status", s),
    )
    problems = orch.preflight()
    if problems:
        raise HTTPException(400, {"preflight": problems})

    hub.orch = orch
    hub.thread = threading.Thread(target=orch.run, daemon=True)
    hub.thread.start()
    return {"started": True, "blocks": len(bl), "slides": len(slot_ids)}


@app.get("/api/status")
def status() -> dict:
    s = hub.status
    job = hub.orch.job if hub.orch else Job.load()
    return {
        "busy": hub.busy, "sim": cfgmod.sim_mode(),
        "status": s.__dict__,
        "job": None if job is None else {
            "dxf": job.dxf_name, "state": job.state, "message": job.message,
            "fraction": round(job.fraction, 4),
            "slides": [sp.__dict__ for sp in job.slides],
        },
    }


@app.post("/api/control/{action}")
def control(action: str) -> dict:
    if hub.orch is None:
        raise HTTPException(400, "no job")
    match action:
        case "pause":
            hub.orch.pause()
        case "resume":
            hub.orch.resume()
        case "soft_stop":
            hub.orch.soft_stop()
        case "hard_stop":
            hub.orch.hard_stop()
        case _:
            raise HTTPException(404, f"unknown action {action}")
    return {"ok": True, "action": action}


# ---------- jog and test (refused while a job runs) ----------

@app.get("/api/machine/position")
def position() -> dict:
    p = hub.printer.position()
    return {"x": p.x, "y": p.y, "z": p.z, "homed": p.homed, "is_homed": p.is_homed}


@app.post("/api/machine/home")
def home(payload: dict | None = None) -> dict:
    hub.require_idle()
    hub.printer.home((payload or {}).get("axes", ""))
    return position()


@app.post("/api/machine/jog")
def jog(payload: dict) -> dict:
    """Relative jog, clamped against limits_mm before anything is sent.

    Clamping here matters because the GUI repeats while a button is held.
    """
    hub.require_idle()
    p = hub.printer.position()
    speed = float(payload.get("speed_mm_s", hub.machine.speeds["travel"]))
    tx = hub.machine.clamp("x", p.x + float(payload.get("dx", 0.0)))
    ty = hub.machine.clamp("y", p.y + float(payload.get("dy", 0.0)))
    tz = hub.machine.clamp("z", p.z + float(payload.get("dz", 0.0)))
    hub.printer.move(x=tx, y=ty, z=tz, speed_mm_s=speed)
    return position()


@app.post("/api/machine/move")
def move(payload: dict) -> dict:
    hub.require_idle()
    x, y, z = (payload.get("x"), payload.get("y"), payload.get("z"))
    if not hub.machine.in_limits(
        float(x if x is not None else 0), float(y if y is not None else 0),
        float(z if z is not None else 0),
    ):
        raise HTTPException(400, "target is outside limits_mm")
    hub.printer.move(x=x, y=y, z=z, speed_mm_s=float(payload.get("speed_mm_s", 50.0)))
    return position()


@app.post("/api/machine/estop")
def estop() -> dict:
    hub.printer.emergency_stop()
    return {"ok": True}


@app.post("/api/machine/restart")
def restart() -> dict:
    hub.printer.firmware_restart()
    return {"ok": True}


@app.post("/api/gripper/{action}")
def gripper(action: str) -> dict:
    hub.require_idle()
    if action == "open":
        hub.gripper.open()
    elif action == "close":
        hub.gripper.close()
    else:
        raise HTTPException(404, action)
    return {"state": getattr(hub.gripper, "state", action)}


@app.get("/api/loadcell")
def loadcell() -> dict:
    return {"grams": round(hub.loadcell.grams(), 3)}


@app.post("/api/loadcell/tare")
def tare() -> dict:
    hub.require_idle()
    return {"tare_counts": hub.loadcell.tare()}


@app.post("/api/loadcell/scale")
def scale(payload: dict) -> dict:
    hub.require_idle()
    g = float(payload.get("grams", 0))
    return {"scale_counts_per_g": hub.loadcell.calibrate_with_known_weight(g)}


# ---------- calibration ----------

@app.get("/api/calibration")
def get_calibration() -> dict:
    return hub.cal.raw


@app.post("/api/calibration")
def put_calibration(payload: dict) -> dict:
    """Stage and write calibration.yaml, then reload it in place."""
    hub.require_idle()
    path = cfgmod.save_calibration(payload)
    hub.reload_calibration()
    return {"written": str(path)}


@app.post("/api/calibration/capture")
def capture(payload: dict) -> dict:
    """Store the live machine position for the target being taught, so nothing
    is typed by hand."""
    hub.require_idle()
    p = hub.printer.position()
    return {"target": payload.get("target"), "id": payload.get("id"),
            "x": p.x, "y": p.y, "z": p.z}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(HTTPException)
async def http_error(_req: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
