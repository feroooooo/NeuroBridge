@echo off
setlocal enabledelayedexpansion

REM
set "IMAGE_FEATURE_BASE_DIR=.\data\image_feature"
set "IMAGE_ENCODER_TYPE=RN50"
set "IMAGE_FEATURE_DIR=%IMAGE_FEATURE_BASE_DIR%\%IMAGE_ENCODER_TYPE%"
set "TEXT_FEATURE_DIR="
set "EEG_DATA_DIR=.\data\preprocessed_eeg"
set "DEVICE=cuda:0"
set "EEG_ENCODER_TYPE=EEGProject"
set "BATCH_SIZE=1024"
set "LEARNING_RATE=1e-4"
set "NUM_EPOCHS=50"
set "BRAIN_AREA=o+p"
set "PROJECTOR=linear"
set "FEATURE_DIM=512"
set "OUTPUT_DIR=.\results\intra_subjects_symmetry_loss"

REM
for /l %%S in (2,1,10) do (
    set "SUB_ID=%%S"

    REM
    set "OUTPUT_NAME=sub-0!SUB_ID!"
    if !SUB_ID! GEQ 10 set "OUTPUT_NAME=sub-!SUB_ID!"

    echo Training subject !SUB_ID!...

    REM
    python train.py ^
        --batch_size "%BATCH_SIZE%" ^
        --learning_rate "%LEARNING_RATE%" ^
        --output_name "!OUTPUT_NAME!" ^
        --eeg_encoder_type "%EEG_ENCODER_TYPE%" ^
        --train_subject_ids !SUB_ID! ^
        --test_subject_ids !SUB_ID! ^
        --num_epochs "%NUM_EPOCHS%" ^
        --image_feature_dir "%IMAGE_FEATURE_DIR%" ^
        --text_feature_dir "%TEXT_FEATURE_DIR%" ^
        --eeg_data_dir "%EEG_DATA_DIR%" ^
        --device "%DEVICE%" ^
        --output_dir "%OUTPUT_DIR%" ^
        --brain_area "%BRAIN_AREA%" ^
        --image_aug ^
        --aug_image_feature_dirs ".\data\image_feature\RN50\GaussianBlur-GaussianNoise-LowResolution-Mosaic" ^
        --eeg_aug ^
        --eeg_aug_type "smooth" ^
        --image_test_aug ^
        --img_l2norm ^
        --projector "%PROJECTOR%" ^
        --feature_dim "%FEATURE_DIM%" ^
        --data_average ^
        --save_weights ^
        --seed 2025

    if errorlevel 1 (
        echo Script Error
        exit /b 1
    )
)

REM
python average.py --result_dir "%OUTPUT_DIR%"
if errorlevel 1 (
    echo Script Error
    exit /b 1
)

exit /b 0
