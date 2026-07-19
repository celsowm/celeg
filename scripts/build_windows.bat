@echo off
setlocal enabledelayedexpansion
rem ===========================================================================
rem  Windows native build helper for lfm25-native-cpp (MSVC + nvcc + Ninja)
rem
rem  Usage:
rem    scripts\build_windows.bat            configure + build (Release, CUDA on)
rem    scripts\build_windows.bat configure  re-run cmake configure step only
rem    scripts\build_windows.bat build      build only (skip configure)
rem    scripts\build_windows.bat test       run ctest after a successful build
rem    scripts\build_windows.bat clean      remove the build directory
rem
rem  Environment overrides:
rem    VCVARSALL   path to vcvarsall.bat (auto-detected if unset)
rem    BUILD_DIR   output directory (default: build-win)
rem    BUILD_TYPE  Release|Debug|RelWithDebInfo (default: Release)
rem    CUDA_ARCH   CMAKE_CUDA_ARCHITECTURES (default: native)
rem    LFM_TESTS   ON|OFF (default: ON)
rem    LFM_CUDA    ON|OFF (default: ON)
rem ===========================================================================

set "ROOT=%~dp0.."
pushd "%ROOT%" >nul
set "ROOT=%CD%"
popd >nul

set "BUILD_DIR=%BUILD_DIR%"
if "%BUILD_DIR%"=="" set "BUILD_DIR=build-win"
set "BUILD_TYPE=%BUILD_TYPE%"
if "%BUILD_TYPE%"=="" set "BUILD_TYPE=Release"
set "CUDA_ARCH=%CUDA_ARCH%"
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=native"
set "LFM_TESTS=%LFM_TESTS%"
if "%LFM_TESTS%"=="" set "LFM_TESTS=ON"
set "LFM_CUDA=%LFM_CUDA%"
if "%LFM_CUDA%"=="" set "LFM_CUDA=ON"
rem nvcc 12.x/13.x only officially support MSVC 2019-2022; pass this flag so
rem newer BuildTools installs (e.g. VS 18/MSVC 19.50) still compile .cu files.
set "NVCC_ALLOW_UNSUPPORTED=%NVCC_ALLOW_UNSUPPORTED%"
if "%NVCC_ALLOW_UNSUPPORTED%"=="" set "NVCC_ALLOW_UNSUPPORTED=ON"

set "VCVARSALL=%VCVARSALL%"
if "%VCVARSALL%"=="" (
    set "VCVARSALL=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
)
if not exist "%VCVARSALL%" (
    for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VSINSTALL=%%i"
    if exist "!VSINSTALL!\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARSALL=!VSINSTALL!\VC\Auxiliary\Build\vcvarsall.bat"
)

if not exist "%VCVARSALL%" (
    echo [build_windows] error: vcvarsall.bat not found.
    echo [build_windows] set VCVARSALL to the full path of vcvarsall.bat or install
    echo [build_windows] the "MSVC v143 - VS 2022 C++ x64/x86 build tools" component.
    exit /b 1
)

echo [build_windows] using MSVC: %VCVARSALL%
call "%VCVARSALL%" x64
if errorlevel 1 (
    echo [build_windows] error: vcvarsall.bat failed.
    exit /b 1
)

where ninja >nul 2>nul
if errorlevel 1 (
    echo [build_windows] warning: ninja not found on PATH; falling back to NMake.
    set "GENERATOR=NMake Makefiles"
) else (
    set "GENERATOR=Ninja"
)

set "ACTION=%1"
if "%ACTION%"=="" set "ACTION=all"

if /i "%ACTION%"=="clean" (
    echo [build_windows] cleaning %BUILD_DIR%
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    exit /b 0
)

if /i "%ACTION%"=="build" goto :build
if /i "%ACTION%"=="test" goto :build

:configure
echo [build_windows] configuring into %BUILD_DIR% (%GENERATOR%, %BUILD_TYPE%)
set "CUDA_FLAGS_ARG="
if /i "%NVCC_ALLOW_UNSUPPORTED%"=="ON" set "CUDA_FLAGS_ARG=-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
if /i "%NVCC_ALLOW_UNSUPPORTED%"=="ON" set "CUDAFLAGS=-allow-unsupported-compiler"
cmake -S "%ROOT%" -B "%ROOT%\%BUILD_DIR%" -G "%GENERATOR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
    -DLFM_ENABLE_CUDA=%LFM_CUDA% ^
    -DLFM_BUILD_TESTS=%LFM_TESTS% ^
    -DLFM_RUN_CUDA_TESTS=ON ^
    %CUDA_FLAGS_ARG%
if errorlevel 1 (
    echo [build_windows] error: cmake configure failed.
    exit /b 1
)

if /i "%ACTION%"=="configure" exit /b 0

:build
if not exist "%ROOT%\%BUILD_DIR%\CMakeCache.txt" goto :configure
echo [build_windows] building %BUILD_TYPE%
cmake --build "%ROOT%\%BUILD_DIR%" --config %BUILD_TYPE% --parallel
if errorlevel 1 (
    echo [build_windows] error: build failed.
    exit /b 1
)

if /i "%ACTION%"=="test" (
    echo [build_windows] running ctest
    ctest --test-dir "%ROOT%\%BUILD_DIR%" --output-on-failure -C %BUILD_TYPE%
    exit /b %errorlevel%
)

echo [build_windows] done. binaries are in %ROOT%\%BUILD_DIR%
exit /b 0
