import os
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_feature_dir", type=str, default="./data/things_eeg/image_feature/RN50")
parser.add_argument("--aug_type", type=str, nargs='+', default=["GaussianBlur", "GaussianNoise", "LowResolution", "Mosaic"], choices=["GaussianBlur", "GaussianNoise", "Mosaic", "RandomCrop", "LowResolution", "ColorJitter", "GrayScale", "None"])
args = parser.parse_args()

image_feature_dir = args.image_feature_dir
output_dir = "-".join(args.aug_type)
aug_list = args.aug_type
train_feature = []
test_feature = []

for aug in aug_list:
    aug_feature_dir = os.path.join(image_feature_dir, aug)
    train_feature_path = os.path.join(aug_feature_dir, "train.npy")
    test_feature_path = os.path.join(aug_feature_dir, "test.npy")
    train_feature.append(np.load(train_feature_path))
    test_feature.append(np.load(test_feature_path))

train_feature = np.concatenate(train_feature, axis=0)
test_feature = np.concatenate(test_feature, axis=0)
print(train_feature.shape, test_feature.shape)

train_feature = np.mean(train_feature, axis=0, keepdims=True)
test_feature = np.mean(test_feature, axis=0, keepdims=True)

print(train_feature.shape, test_feature.shape)

os.makedirs(os.path.join(image_feature_dir, output_dir), exist_ok=True)
np.save(os.path.join(image_feature_dir, output_dir, "train.npy"), train_feature)
np.save(os.path.join(image_feature_dir, output_dir, "test.npy"), test_feature)