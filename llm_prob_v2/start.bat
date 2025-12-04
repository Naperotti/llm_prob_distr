@echo off
REM Start the LLM Probability Distribution app

echo Starting LLM Probability Distribution server...
echo.

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

echo The app will open in your browser automatically.
echo Keep this window open while using the app.
echo Press Ctrl+C to stop the server.
echo.

REM Start the server
start http://localhost:8000
python -m uvicorn app:app --reload

echo.
echo Server stopped.
pause
