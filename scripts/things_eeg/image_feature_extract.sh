#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

python extract_feature.py --output_dir "./data/things_eeg/image_feature/RN50" --aug_type "None" --num_images_per_object_train 10 --num_images_per_object_test 1 --device "cuda:0"

for AUG_TYPE in GaussianBlur GaussianNoise LowResolution Mosaic
do
    python extract_feature.py --output_dir "./data/things_eeg/image_feature/RN50/${AUG_TYPE}" --aug_type "$AUG_TYPE" --num_images_per_object_train 10 --num_images_per_object_test 1 --device "cuda:0"
done

python fuse_feature.py --image_feature_dir "./data/things_eeg/image_feature/RN50" --aug_type "GaussianBlur" "GaussianNoise" "LowResolution" "Mosaic"