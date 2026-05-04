@echo off
title Roman Urdu Sentiment API
cd /d "%~dp0"
echo Starting server...
start /B .\WPy64-3.13.12.0\python\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
timeout /t 5 >nul
start http://127.0.0.1:8000
echo Server running. Close this window to stop.
pause