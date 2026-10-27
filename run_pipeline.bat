@echo off
chcp 65001 >nul
cd /d "%~dp0"
set LOKY_MAX_CPU_COUNT=4
python tools\pipeline_modeling.py
if errorlevel 1 (echo Pipeline FAILED, check the error above & pause & exit /b 1)
start "" "reports\dashboard.html"
pause
