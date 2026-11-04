#!/usr/bin/env bash
# Run once on the Pi, from inside the papa-software folder:
#   bash deploy/install-service.sh                      # printer URL from config/machine.yaml
#   bash deploy/install-service.sh http://10.24.48.56:7125   # printer on the campus network
# Makes the GUI start by itself every time the Pi turns on.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="$(pwd)"
PRINTER_URL="${1:-}"
[ -x "$DIR/.venv/bin/python" ] || { echo "No .venv here. Do the install steps first."; exit 1; }
ENVLINE=""
[ -n "$PRINTER_URL" ] && ENVLINE="Environment=PAPA_MOONRAKER_URL=$PRINTER_URL"
sudo tee /etc/systemd/system/papa.service >/dev/null <<UNIT
[Unit]
Description=PAPA control software
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$DIR
Environment=PYTHONUNBUFFERED=1
$ENVLINE
ExecStart=$DIR/.venv/bin/python -m apps.server
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now papa
sleep 3
systemctl --no-pager --lines=5 status papa || true
echo
echo "Done. Open http://$(hostname -I | awk '{print $1}'):8080/validate"
