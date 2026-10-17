@echo off
cd /d "%~dp0"
title AI Fall Detection System
echo Starting AI Fall Detection System...

if exist "C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe" (
    "C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe" fall_detector.py
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        py -3 fall_detector.py
    ) else (
        python fall_detector.py
    )
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Program exited with error code %ERRORLEVEL%.
)
pause
