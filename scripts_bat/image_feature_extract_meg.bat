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
python feature_extract.py --output_dir ".\data\image_feature_meg\RN50" --image_set_dir "./data/image_set_meg" --aug_type "None" --num_images_per_object 12
if errorlevel 1 exit /b 1

REM
for %%A in (GaussianBlur GaussianNoise LowResolution Mosaic) do (
    python feature_extract.py --output_dir ".\data\image_feature_meg\RN50\%%A" --image_set_dir "./data/image_set_meg" --aug_type "%%A" --num_images_per_object 12
    if errorlevel 1 exit /b 1
)

REM
python feature_fuse.py --image_feature_dir ".\data\image_feature_meg\RN50" --aug_type "GaussianBlur" "GaussianNoise" "LowResolution" "Mosaic"
if errorlevel 1 exit /b 1

exit /b 0