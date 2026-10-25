# Getting started

Written for someone who has never run this. Three stages, each useful on its
own: your laptop alone, your laptop plus the printer, then the Pi.

---

## Stage 1 — on your laptop, no hardware at all

Everything runs against a simulated printer. Nothing can be damaged, so this is
the place to click around.

**Once, to set up:**

```bash
git clone https://github.com/daviguidelli/papa-software
cd papa-software
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

`python3 -m venv .venv` makes a private folder of Python packages for this
project, so installing things here cannot break anything else on your machine.
`source .venv/bin/activate` switches your terminal into it — you will see
`(.venv)` appear in the prompt. You need that line in every new terminal
window; the rest is one-time.

**Check it works:**

```bash
pytest
```

51 tests should pass in about a second. If they do, the whole toolpath,
calibration and job pipeline is intact on your machine.

**Open the GUI:**

```bash
PAPA_SIM=1 python -m apps.server
```

Then open <http://localhost:8080> in a browser. Leave the terminal running —
closing it stops the server. `Ctrl+C` in that terminal shuts it down.

`PAPA_SIM=1` means "simulate every piece of hardware". The chips along the top
of the page will say **Mock printer**, **loadcell mocked**, and so on, so you
can always tell what is real.

Try it: press **Home all**, then the arrow buttons. The X, Y and Z readout
follows the simulated machine. Hold an arrow down and it repeats. Type
`X = 999` into Absolute move and press Move there — it is refused, because the
machine's limit is 350 mm.

---

## Stage 2 — your laptop driving the real printer

This proves the network link before anything is wired to the Pi. The laptop
talks to the printer's mainboard over the network; no Pi is involved.

**1. Find the printer's address.** On the printer's own screen, look for its IP
under network settings — something like `192.168.1.42`.

**2. Check it answers.** In a terminal:

```bash
curl http://192.168.1.42:7125/printer/info
```

You want a reply containing `"state": "ready"`. If you get nothing, the
printer is not reachable from your laptop — fix that before going further.

**3. Point the software at it.** Open `config/machine.yaml` and edit the first
line:

```yaml
moonraker_url: http://192.168.1.42:7125
```

**4. Allow your laptop to send commands.** The printer refuses commands from
machines it does not know. On the printer, in `moonraker.conf`:

```ini
[authorization]
trusted_clients:
    192.168.1.0/24
```

Use your laptop's own address, or the whole subnet as above. Restart Moonraker
after editing. **If `/printer/info` answers but moves get refused, this is
almost always the reason.**

**5. Run it:**

```bash
PAPA_SIM=1 PAPA_REAL=printer python -m apps.server
```

Read that as: simulate everything, *except* the printer is real. The load cell,
gripper and camera stay simulated because they plug into the Pi's GPIO pins,
which a laptop does not have. The top of the page will show the printer's
address instead of "Mock printer".

**6. First moves.** Stand near the power switch.

- **Home all**, and watch it actually home.
- Set the step to **1 mm** and press X+ once. Confirm the readout changed by
  1 mm *and* that the gantry moved about a millimetre. A ruler is fine.
- Only then try 10 mm.

If a move does nothing, look at the **Log** panel at the bottom right: it shows
every request and what came back.

---

## Stage 3 — on the Raspberry Pi

Now the Pi becomes the computer that runs everything, and your laptop is just a
browser pointed at it. The code is identical — only *where it runs* changes.

**1. Install on the Pi** (over SSH, or with a keyboard attached):

```bash
sudo apt install -y python3-venv python3-picamera2 python3-lgpio
git clone https://github.com/daviguidelli/papa-software
cd papa-software
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

`--system-site-packages` lets the project see `picamera2` and `lgpio`, which
come from `apt` rather than `pip` because they need the Pi's own libraries.

**2. Wire the printer link.** Same two steps as Stage 2, but with the Pi's
address: set `moonraker_url`, and put the **Pi's** address in Moonraker's
`trusted_clients`. If the Pi connects to the mainboard with a direct Ethernet
cable there is no router to hand out addresses, so both ends need fixed ones —
see `deploy/setup-pi.md`.

**3. Bring up one device at a time.** Do not plug everything in and hope.

```bash
# printer only, everything else simulated — same as Stage 2
PAPA_SIM=1 PAPA_REAL=printer .venv/bin/python -m apps.server

# add the gripper once it is wired
PAPA_SIM=1 PAPA_REAL=printer,gripper .venv/bin/python -m apps.server

# add the load cell
PAPA_SIM=1 PAPA_REAL=printer,gripper,loadcell .venv/bin/python -m apps.server

# everything real
.venv/bin/python -m apps.server
```

Each step adds one thing that can be wrong. The page's chips tell you which
devices are real, and a device that fails to start returns an error on its own
controls while everything else keeps working — a missing load cell does not
stop you jogging.

**4. Open it from your laptop.** The Pi serves on port 8080, so from any
browser on the same network:

```
http://<pi-address>:8080
```

**5. Start it automatically.** Once it works by hand, install the service file
so it starts on boot:

```bash
sudo cp deploy/papa.service /etc/systemd/system/
sudo systemctl enable --now papa
```

Check it with `systemctl status papa`, and read its output with
`journalctl -u papa -f`.

---

## Things that commonly go wrong

| What you see | What it means |
| --- | --- |
| `/printer/info` answers, moves are refused | The address is missing from Moonraker's `trusted_clients` |
| Nothing answers at all | Wrong IP, printer off, or different network |
| `ModuleNotFoundError: lgpio` | Running with real hardware selected on a machine without GPIO. Use `PAPA_SIM=1 PAPA_REAL=printer` |
| Page loads but says "Unreachable" | The server is running; the printer is not answering it |
| Fonts look plain | The page fetches its fonts from the internet. Harmless, and it falls back cleanly on an offline network |
| A job refuses to start | Preflight lists exactly what is missing — usually an untaught position |
