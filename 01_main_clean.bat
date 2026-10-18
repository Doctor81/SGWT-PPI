@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set types=reaction binding ptmod activation inhibition catalysis expression
set seeds=1 2 3 4 5 6 7 8 9 10
set modes=none diffusion spectral
set current=0

echo === Основная серия без утечки: 210 runs ===
for %%t in (%types%) do (
  for %%s in (%seeds%) do (
    for %%f in (%modes%) do (
      set /a current+=1
      echo [!current!/210] %%t %%f seed %%s
      python train.py --neg-mode clean --feature-mode %%f --ppimode %%t --seed %%s --skip-existing
      if errorlevel 1 (
        echo ОШИБКА на запуске %%t %%f %%s - остановка
        pause
        exit /b 1
      )
    )
  )
)
echo.
echo === Готово: Основная серия без утечки ===
pause
