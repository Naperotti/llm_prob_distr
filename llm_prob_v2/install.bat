@echo off
REM Setup virtual environment and install dependencies

echo Setting up LLM Probability Distribution environment...
echo.

REM Check if venv already exists
if exist .venv (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    echo Trying Python 3.13...
    py -3.13 -m venv .venv 2>nul
    if errorlevel 1 (
        echo Python 3.13 not found, trying 3.11...
        py -3.11 -m venv .venv 2>nul
        if errorlevel 1 (
            echo Python 3.11/3.13 not found, using default Python...
            echo WARNING: UMAP may not work with Python 3.14+
            python -m venv .venv
        )
    )
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Installing dependencies (this may take a few minutes)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ==========================================
echo Installation complete!
echo You can now run start.bat
echo ==========================================
echo.
pause
