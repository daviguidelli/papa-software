@echo off
rem Runs the GUI in simulation: no printer, no Pi, nothing real. Windows.
cd /d "%~dp0\.."
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (%PY% -m venv .venv)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
set PAPA_SIM=1
echo.
echo Ready. Open  http://localhost:8080/validate  in your browser.
echo Stop with Ctrl+C.
echo.
python -m apps.server
