#!/usr/bin/env bash
# Runs the GUI in simulation: no printer, no Pi, nothing real. Mac and Linux.
set -e
cd "$(dirname "$0")/.."
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.11 or newer is needed. Install it from python.org, then run this again."; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
echo
echo "Ready. Open  http://localhost:8080/validate  in your browser."
echo "Stop with Ctrl+C."
echo
PAPA_SIM=1 python -m apps.server
