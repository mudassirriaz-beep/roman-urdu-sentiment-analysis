@echo off
echo Installing packages...
.\WPy64-3.13.12.0\python\python.exe -m pip install --no-cache-dir fastapi uvicorn transformers torch pandas python-multipart
echo Done.
pause