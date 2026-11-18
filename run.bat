@echo off
setlocal
set "PY310=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if exist "%PY310%" (
    "%PY310%" "%~dp0main.py" %*
) else (
    python "%~dp0main.py" %*
)
