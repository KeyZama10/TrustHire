@echo off
REM Windows shim so `make demo` works without GNU make. From PowerShell run:  .\make demo
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%~dp0scripts\tasks.py" %*
) else (
  python "%~dp0scripts\tasks.py" %*
)
