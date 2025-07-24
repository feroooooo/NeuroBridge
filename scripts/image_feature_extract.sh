#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

python feature_extract.py --output_dir "./data/image_feature/RN50" --aug_type "None"

for AUG_TYPE in GaussianBlur GaussianNoise LowResolution Mosaic
do
    python feature_extract.py --output_dir "./data/image_feature/RN50/${AUG_TYPE}" --aug_type "$AUG_TYPE"
done

python feature_fuse.py --image_feature_dir "./data/image_feature/RN50" --aug_type "GaussianBlur" "GaussianNoise" "LowResolution" "Mosaic"