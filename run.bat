@echo off
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate
rem Normal launches stay offline: pip runs only on the first start or when requirements.txt changed.
fc /b requirements.txt .venv\requirements.installed >nul 2>&1
if errorlevel 1 (
    echo Installing requirements - first start or requirements.txt changed...
    pip install -q -r requirements.txt && copy /y requirements.txt .venv\requirements.installed >nul
)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
