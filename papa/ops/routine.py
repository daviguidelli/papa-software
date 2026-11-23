"""Routine 1: grip a swab, scrub the sketch, dispose of the swab.

Built from the same steps the Autonomous tab and the Scrub tab already use. A
fault or a stop during the scrub lifts the swab clear and ends the routine with
the swab still held; nothing moves on its own after that.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

from papa.ops import autonomous as automod
from papa.ops import scrub as scrubmod
from papa.ops.autonomous import AutoStatus, Step


class RoutineError(RuntimeError):
    pass


def problems(auto_st: dict, scrub_st: dict, machine: Any, ref: str | None, floor: float | None) -> list[str]:
    out = automod.problems(auto_st, machine, "grip", ref)
    for m in automod.problems(auto_st, machine, "dispose", ref):
        if m not in out:
            out.append(m)
    out += scrubmod.problems(scrub_st, machine, ref, floor)
    return out


def describe(auto_st: dict, scrub_st: dict, machine: Any, ref: str | None, floor: float | None) -> dict:
    bad = problems(auto_st, scrub_st, machine, ref, floor)
    return {"problems": bad, "ready": not bad}


class RoutineRunner:
    def __init__(self, auto: automod.AutoRunner, scrub: scrubmod.ScrubRunner, machine: Any) -> None:
        self.auto, self.scrub, self.machine = auto, scrub, machine
        self.status = AutoStatus()
        self._abort = threading.Event()
        self.thread: threading.Thread | None = None

    def _check(self) -> None:
        if self._abort.is_set():
            raise automod.Aborted("stopped between moves")

    def start(self, auto_st: dict, scrub_st: dict, ref: str | None, floor: float | None) -> threading.Thread:
        bad = problems(auto_st, scrub_st, self.machine, ref, floor)
        if bad:
            raise RoutineError(" ".join(bad))
        sst = scrubmod.merged(scrub_st)
        plan = scrubmod.plan(sst)
        safe = float(auto_st["safe_z"])

        def do_scrub() -> None:
            try:
                self.scrub.execute(sst, plan, self._check, safe_z=safe)
            except BaseException:
                self.scrub.lift_after_stop()
                raise

        steps: list[tuple[str, Callable[[], None]]] = (
            self.auto.grip_steps(auto_st) + [("Scrub the pattern", do_scrub)] + self.auto.dispose_steps(auto_st)
        )
        self._abort.clear()
        self.auto._abort.clear()
        self.status = AutoStatus(running=True, action="routine1", steps=[Step(n) for n, _ in steps])
        self.thread = threading.Thread(target=self._run, args=(steps,), daemon=True)
        self.thread.start()
        return self.thread

    def stop(self) -> None:
        self._abort.set()
        self.auto._abort.set()

    def _run(self, steps: list[tuple[str, Callable[[], None]]]) -> None:
        s = self.status
        try:
            for i, (_, fn) in enumerate(steps):
                self._check()
                s.steps[i].state = "running"
                try:
                    fn()
                except Exception:
                    s.steps[i].state = "failed"
                    raise
                s.steps[i].state = "done"
            s.finished = True
        except Exception as exc:                          # noqa: BLE001
            s.error = str(exc) or type(exc).__name__
            for st in s.steps:
                if st.state == "pending":
                    st.state = "skipped"
        finally:
            s.running = False


class Routine2Runner(RoutineRunner):
    """Routine 2: scrub two slides.

    Check Z with the swab in the jaws, scrub A, dispose, grip the swab at
    position 1, check Z, scrub B, dispose, grip the swab at position 2 (ready
    for the next run) and park at safe Z. A check only reports the drift: Z is
    never changed.

    Position 1 is the taught pickup; position 2 is one `pickup_step` further.
    """

    def __init__(self, auto: automod.AutoRunner, scrub: scrubmod.ScrubRunner, machine: Any) -> None:
        super().__init__(auto, scrub, machine)
        self.last_check: dict | None = None

    def start2(self, auto_st: dict, slides: list[dict], ref: str | None, floor: float | None,
               check_fn: Callable[[Callable[[], None]], dict], park_fn: Callable[[], None],
               swab_index: int = 0, extra_problems: list[str] | None = None) -> threading.Thread:
        if len(slides) != 2:
            raise RoutineError("Routine 2 scrubs exactly two slides.")
        return self.start_many(auto_st, [("slide A", slides[0]), ("slide B", slides[1])], ref, floor,
                               check_fn, park_fn, swab_index, extra_problems, action="routine2")

    def start_many(self, auto_st: dict, slides: list[tuple[str, dict]], ref: str | None, floor: float | None,
                   check_fn: Callable[[Callable[[], None]], dict], park_fn: Callable[[], None],
                   swab_index: int = 0, extra_problems: list[str] | None = None,
                   action: str = "routine3x3") -> threading.Thread:
        """Routine 2's sequence for any number of slides: for each slide, check
        Z on the load cell, scrub it, dispose of the swab and grip the next one
        (swab n = the taught pickup + n x the next-swab step); park at the end."""
        bad = list(extra_problems or [])
        bad += automod.problems(auto_st, self.machine, "grip", ref)
        for m in automod.problems(auto_st, self.machine, "dispose", ref):
            if m not in bad:
                bad.append(m)
        for label, s in slides:
            bad += [f"{label[:1].upper()}{label[1:]}: {m}" for m in scrubmod.problems(s, self.machine, ref, floor)]
        if not slides:
            bad.append("No slide to scrub.")
        if bad:
            raise RoutineError(" ".join(bad))
        safe = float(auto_st["safe_z"])
        step = auto_st.get("pickup_step") or {"x": 0.0, "y": 0.0}

        def grip_at(n: int) -> list[tuple[str, Callable[[], None]]]:
            t = dict(auto_st["pickup"])
            t["x"] = float(t["x"]) + n * float(step.get("x", 0.0))
            t["y"] = float(t["y"]) + n * float(step.get("y", 0.0))
            if not self.machine.in_limits(t["x"], t["y"], t["z"]):
                raise RoutineError(f"Swab {n + 1} would be outside the machine limits.")
            return self.auto.grip_steps({**auto_st, "pickup": t})

        def scrub_step(label: str, raw: dict) -> tuple[str, Callable[[], None]]:
            sst = scrubmod.merged(raw)
            plan = scrubmod.plan(sst)

            def go() -> None:
                try:
                    self.scrub.execute(sst, plan, self._check, safe_z=safe)
                except BaseException:
                    self.scrub.lift_after_stop()
                    raise
            return (label, go)

        def do_check() -> None:
            self.last_check = check_fn(self._check)

        for n in range(len(slides)):              # every swab position must be reachable before anything moves
            grip_at(swab_index + n)
        steps: list[tuple[str, Callable[[], None]]] = []
        r2 = action == "routine2"                 # Routine 2 keeps its original step names
        for n, (label, raw) in enumerate(slides):
            steps += [("Check Z on the load cell (Z unchanged)" if r2 else f"[{label}] Check Z on the load cell (Z unchanged)",
                       do_check)]
            steps += [scrub_step(f"Scrub {label}", raw)]
            steps += [(m if r2 else f"[{label}] {m}", f) for m, f in self.auto.dispose_steps(auto_st)]
            tag = f"[position {n + 1}]" if r2 else f"[swab {swab_index + n + 1}]"
            steps += [(f"{tag} {m}", f) for m, f in grip_at(swab_index + n)]
        steps += [("Park at safe Z", park_fn)]
        self._abort.clear()
        self.auto._abort.clear()
        self.last_check = None
        self.status = AutoStatus(running=True, action=action, steps=[Step(n) for n, _ in steps])
        self.thread = threading.Thread(target=self._run, args=(steps,), daemon=True)
        self.thread.start()
        return self.thread
