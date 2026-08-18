@echo off
REM ---------------------------------------------------------------
REM  XAUUSD signal engine - Windows launcher
REM  Runs the live PAPER SIGNAL engine. No orders are ever placed.
REM ---------------------------------------------------------------

cd /d "%~dp0"

if not exist ".env" (
    echo.
    echo ERROR: .env not found.
    echo Copy .env.example to .env and fill in your credentials first.
    echo.
    pause
    exit /b 1
)

echo Starting XAUUSD signal engine...
python main.py

echo.
echo Engine stopped. Press any key to close.
pause >nul
