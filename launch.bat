@echo off
echo Professional Contingency Analysis Suite
echo ======================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"
if exist "dist\Professional_Contingency_Analysis_Suite.exe" (
    start "" "dist\Professional_Contingency_Analysis_Suite.exe"
) else (
    echo Error: Executable not found!
    echo Please run build_complete.py first.
    pause
)
