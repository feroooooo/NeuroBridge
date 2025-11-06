#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

python feature_extract.py --image_set_dir "./data/things_meg/image_set" --output_dir "./data/things_meg/image_feature/RN50" --aug_type "None" --num_images_per_object 12 --device "cuda:0"

for AUG_TYPE in GaussianBlur GaussianNoise LowResolution Mosaic
do
    python feature_extract.py --image_set_dir "./data/things_meg/image_set" --output_dir "./data/things_meg/image_feature/RN50/${AUG_TYPE}" --aug_type "$AUG_TYPE" --num_images_per_object 12 --device "cuda:0"
done

python feature_fuse.py --image_feature_dir "./data/things_meg/image_feature/RN50" --aug_type "GaussianBlur" "GaussianNoise" "LowResolution" "Mosaic"