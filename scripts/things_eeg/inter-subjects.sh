#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_BASE_DIR="./data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="RN50"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="./data/things_eeg/preprocessed_eeg"
DEVICE="cuda:0"
EEG_ENCODER_TYPE="TSConv"
BATCH_SIZE=1024
LEARNING_RATE=1e-4
NUM_EPOCHS=50
BRAIN_AREA="all"
PROJECTOR="linear"
FEATURE_DIM=512
OUTPUT_DIR="./results/things_eeg/inter_subjects"

for SUB_ID in {1..10}
do
    OUTPUT_NAME=$(printf "sub-%02d" $SUB_ID)
    echo "Training subject ${SUB_ID}..."

    TRAIN_IDS=""
    for i in {1..10}
    do
        if [ "$i" -ne "$SUB_ID" ]; then
            TRAIN_IDS+="$i "
        fi
    done

    python train.py \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --output_name "$OUTPUT_NAME" \
        --eeg_encoder_type "$EEG_ENCODER_TYPE" \
        --train_subject_ids $TRAIN_IDS \
        --test_subject_ids $SUB_ID \
        --num_epochs "$NUM_EPOCHS" \
        --image_feature_dir "$IMAGE_FEATURE_DIR" \
        --text_feature_dir "$TEXT_FEATURE_DIR" \
        --eeg_data_dir "$EEG_DATA_DIR" \
        --device "$DEVICE"  \
        --output_dir "$OUTPUT_DIR" \
        --brain_area "$BRAIN_AREA" \
        --image_aug \
        --aug_image_feature_dirs "./data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic" \
        --eeg_aug \
        --eeg_aug_type "smooth" \
        --image_test_aug \
        --img_l2norm \
        --projector "$PROJECTOR" \
        --feature_dim "$FEATURE_DIM" \
        --data_average \
        --save_weights \
        --seed 2025;
done

python average.py --result_dir "$OUTPUT_DIR";