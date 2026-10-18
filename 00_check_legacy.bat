@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set types=reaction expression
set seeds=1 2
set modes=none spectral
set current=0

echo === Проверка: новый код по старому протоколу: 8 runs ===
for %%t in (%types%) do (
  for %%s in (%seeds%) do (
    for %%f in (%modes%) do (
      set /a current+=1
      echo [!current!/8] %%t %%f seed %%s
      python train.py --neg-mode legacy --feature-mode %%f --ppimode %%t --seed %%s --skip-existing --out results\check_legacy.csv
      if errorlevel 1 (
        echo ОШИБКА на запуске %%t %%f %%s - остановка
        pause
        exit /b 1
      )
    )
  )
)
echo.
echo === Готово: Проверка: новый код по старому протоколу ===
pause
