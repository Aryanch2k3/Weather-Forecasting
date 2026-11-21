@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo ERROR: Install Visual Studio Community or Build Tools with Desktop development with C++.
  exit /b 1
)
set "VSROOT="
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
if not defined VSROOT (
  echo ERROR: Visual Studio C++ tools not detected.
  exit /b 1
)
REM CUDA 13.2 may crash with newer MSVC 14.51. User's verified-good MSVC 14.44 is preferred.
REM Override with: set HPC_MSVC_VERSION=14.50
if defined HPC_MSVC_VERSION (
  set "PIN_VERSION=%HPC_MSVC_VERSION%"
) else (
  set "PIN_VERSION="
  for /d %%D in ("%VSROOT%\VC\Tools\MSVC\14.44.*") do set "PIN_VERSION=14.44"
)
if defined PIN_VERSION (
  call "%VSROOT%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 -vcvars_ver=%PIN_VERSION% >nul
) else (
  call "%VSROOT%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul
)
if errorlevel 1 exit /b 1
echo Selected MSVC: %VCToolsVersion%
if not exist build mkdir build
echo === Building serial CPU: eight real CSV weather parameters ===
cl /nologo /O2 /W3 /std:c11 /Fo:build\weather_serial.obj /Fe:build\weather_serial.exe src\weather_cpu.c
if errorlevel 1 exit /b 1
echo === Building OpenMP CPU ===
cl /nologo /O2 /W3 /std:c11 /openmp /Fo:build\weather_omp.obj /Fe:build\weather_omp.exe src\weather_cpu.c
if errorlevel 1 exit /b 1
where nvcc >nul 2>nul
if errorlevel 1 (
  echo WARNING: nvcc missing; CUDA skipped. CPU-only run remains available.
  exit /b 0
)
echo === Building CUDA atomic and shared-memory kernels [GTX 1650 sm_75] ===
nvcc -O3 -arch=sm_75 -o build\weather_cuda.exe src\weather_cuda.cu
if errorlevel 1 exit /b 1
echo === Build finished: .\build\ ===
exit /b 0
