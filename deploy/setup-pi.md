# Pi setup

Everything here is about the two boards finding each other and the Pi being
allowed to command motion. None of it is in the code, and all of it is easy to
forget.

## 1. Fixed IPs on both ends

A direct Pi-to-mainboard cable has no router, so nothing hands out addresses.
Both ends need a static address on the same subnet. On the Pi:

```bash
sudo nmcli con add type ethernet ifname eth0 con-name papa-link \
  ipv4.method manual ipv4.addresses 192.168.50.1/24
```

Give the mainboard `192.168.50.2` and put that in `config/machine.yaml` as
`moonraker_url: http://192.168.50.2:7125`. Check with
`curl http://192.168.50.2:7125/printer/info`.

If the SV08 is on the lab network instead, use its network address and skip the
static setup — but then confirm the address does not change on reboot.

## 2. Let Moonraker accept the Pi

Moonraker refuses requests from clients it does not trust. In its
`moonraker.conf`:

```ini
[authorization]
trusted_clients:
    192.168.50.1
    127.0.0.1
cors_domains:
    *://localhost:*
```

Restart Moonraker. A 401 or 403 from `/printer/gcode/script` almost always
means this step was missed.

## 3. Confirm the mainboard runs Klipper

The plan assumes stock Klipper plus Moonraker on the SV08. Verify before
trusting any of it:

```bash
curl -s http://192.168.50.2:7125/printer/info | python3 -m json.tool
```

`state: ready` and a Klipper version string means the motion server is real.
If the printer runs Marlin instead, the Moonraker client is the only module
that has to change — which is why it is one file.

## 4. GPIO and power

| Signal | Pin | Note |
| --- | --- | --- |
| HX711 DAT | GPIO5 | load cell data |
| HX711 CLK | GPIO6 | load cell clock |
| Servo PWM | GPIO13 | hardware PWM; jitter drops swabs |

The gripper servo runs from its own 4×AA pack, with grounds common to the Pi.
Powering the servo from the Pi's 5 V rail browns out the Pi under stall
current.

## 5. Install

```bash
sudo apt install -y python3-venv python3-picamera2 python3-lgpio
git clone https://github.com/daviguidelli/papa-software
cd papa-software
python3 -m venv .venv --system-site-packages   # picamera2 comes from apt
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                  # should pass with PAPA_SIM=1
```

Then install the unit file in this folder and enable it.

## 6. First power-on order

Work up from the least dangerous thing, so a wiring mistake is found before
anything is pressing on glass.

1. `PAPA_SIM=1` and open the GUI. Nothing is connected; everything should work.
2. Drop `PAPA_SIM`. Jog and test tab: connect, home, jog 1 mm, check the
   readout matches reality.
3. Gripper open and close with no swab, then with a swab.
4. Live load cell reading. Press the swab by hand and watch the value. Tare,
   then calibrate with a known weight. Confirm the cell's rating and set
   `abort_force_g` well under it.
5. Z probe on the load cell with a swab gripped, slow step size.
6. Camera: map the four points, sample the film and stem blues, test detection.
7. Dry run in air on a slot with no slide.
8. One swab on a sacrificial slide.
