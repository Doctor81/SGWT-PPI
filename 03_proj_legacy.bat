@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set types=reaction binding ptmod activation inhibition catalysis expression
set seeds=1 2 3 4 5 6 7 8 9 10
set modes=proj
set current=0

echo === Контроль LayerNorm+Linear по старому протоколу: 70 runs ===
for %%t in (%types%) do (
  for %%s in (%seeds%) do (
    for %%f in (%modes%) do (
      set /a current+=1
      echo [!current!/70] %%t %%f seed %%s
      python train.py --neg-mode legacy --feature-mode %%f --ppimode %%t --seed %%s --skip-existing
      if errorlevel 1 (
        echo ОШИБКА на запуске %%t %%f %%s - остановка
        pause
        exit /b 1
      )
    )
  )
)
echo.
echo === Готово: Контроль LayerNorm+Linear по старому протоколу ===
pause
