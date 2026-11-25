@echo off
setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" "src\main.py" %*
) else (
    echo [Aviso] No se encontro venv\Scripts\python.exe, intentando con python del sistema...
    python "src\main.py" %*
)

if errorlevel 1 (
    echo.
    echo [Error] El programa termino con codigo %errorlevel%.
    pause
)
endlocal
