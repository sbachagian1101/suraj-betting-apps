@echo off
setlocal
title Horse Race Predictor

REM ---------------------------------------------------------------------
REM  Horse Race Predictor - console launcher.
REM
REM  For everyday use double-click HorseRacePredictor.vbs (or the desktop
REM  shortcut): it opens the app in its own window with no console.
REM  This file is the visible version - use it when something goes wrong
REM  and you want to see the error.
REM
REM    run.bat          set up if needed, then start with the console shown
REM    run.bat setup    set up only, then exit
REM ---------------------------------------------------------------------

cd /d "%~dp0"

echo.
echo   Horse Race Predictor
echo   ====================
echo.

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo   Python is not installed, or is not on your PATH.
    echo   Install it from https://www.python.org/downloads/ and tick
    echo   "Add python.exe to PATH" on the first screen.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo   First run - setting up. This takes a few minutes.
    echo.
    %PY% -m venv .venv
    if errorlevel 1 (
        echo   Could not create the environment. Is Python fully installed?
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    echo   Installing the app's requirements...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo   Could not install the requirements. Check your internet
        echo   connection and run this file again.
        pause
        exit /b 1
    )
    echo   Setup finished.
    echo.
)

if /i "%~1"=="setup" (
    echo   Setup is complete. Close this window and use the desktop shortcut,
    echo   or HorseRacePredictor.vbs in this folder.
    echo.
    exit /b 0
)

echo   Starting. The app picks a free port automatically. Close the app
echo   window (or this console) to stop it.
echo.

".venv\Scripts\python.exe" launcher.py

echo.
echo   Horse Race Predictor has stopped.
pause
endlocal
