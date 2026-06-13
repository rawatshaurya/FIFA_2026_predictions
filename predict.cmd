@echo off
set "UV_CACHE_DIR=%~dp0.uv-cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0.uv-python"

pushd "%~dp0"
if /I "%~1"=="--offline" (
    uv run fifa-predict match --offline
) else (
    uv run fifa-predict match
)
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
