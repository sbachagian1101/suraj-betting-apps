@echo off
cd /d "%~dp0"
if "%1"=="test" (
    python -m pytest -q
) else (
    python -m streamlit run app.py
)
