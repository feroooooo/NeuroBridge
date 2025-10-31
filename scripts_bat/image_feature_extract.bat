@echo off
setlocal enabledelayedexpansion

REM
set "ERRORLEVEL=0"
call :main
if %ERRORLEVEL% neq 0 (
    echo Script Error
    exit /b 1
)
exit /b 0

:main
REM
python feature_extract.py --output_dir ".\data\image_feature\RN50" --aug_type "None"
if errorlevel 1 exit /b 1

REM
for %%A in (GaussianBlur GaussianNoise LowResolution Mosaic) do (
    python feature_extract.py --output_dir ".\data\image_feature\RN50\%%A" --aug_type "%%A"
    if errorlevel 1 exit /b 1
)

REM
python feature_fuse.py --image_feature_dir ".\data\image_feature\RN50" --aug_type "GaussianBlur" "GaussianNoise" "LowResolution" "Mosaic"
if errorlevel 1 exit /b 1

exit /b 0