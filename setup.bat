@echo off
rem Downloads model weights and builds qwen.exe. Usage: setup.bat [--model 35b|122b|27b|all|none] [--no-build] [--cpu-only]
cd /d "%~dp0"
python setup.py %*
if errorlevel 1 pause
