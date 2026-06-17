@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CMAKE="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
cd /d "e:\Master\2-2\tgmt\2080ti\tgmt\idea_multibranch"
rmdir /s /q build-msvc-vtk961 2>nul
%CMAKE% -S . -B build-msvc-vtk961 -G Ninja -DCMAKE_BUILD_TYPE=Release -DVTK_DIR="D:/Downloads/VTK-9.6.1-msvc2022-x64/lib/cmake/vtk-9.6"
if errorlevel 1 exit /b 1
%CMAKE% --build build-msvc-vtk961 --target treeTxtSaddleVolumeWeightedSimplifier
