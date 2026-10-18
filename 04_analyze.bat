@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python analyze.py --neg-mode clean
echo.
python analyze.py --neg-mode legacy
pause
