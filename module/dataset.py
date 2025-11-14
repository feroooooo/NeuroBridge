import os
import json
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class EEGPreImageDataset(Dataset):
    def __init__(
        self, 
        subject_ids: list[int], 
        eeg_data_dir: str,
        brain_area:str,
        time_window: list[int],
        image_feature_dir: str,
        text_feature_dir: str,
        image_aug: bool, 
        aug_image_feature_dirs: list[str], 
        average: bool = True, 
        _random: bool = False,
        eeg_transform=None,
        train=True,
        image_test_aug=False,
        eeg_test_aug=False,
        frozen_eeg_prior=False,
    ):
        super().__init__()
        self.subject_ids = subject_ids
        self.average = average
        self.random = _random
        self.eeg_transform = eeg_transform
        self.augment_indices = []
        self.image_feature_dir = image_feature_dir
        self.text_feature_dir = text_feature_dir
        self.train = train
        self.image_aug = image_aug
        self.image_test_aug = image_test_aug
        self.eeg_test_aug = eeg_test_aug
        self.frozen_eeg_prior = frozen_eeg_prior
        
        self.info = json.load(open(os.path.join(eeg_data_dir, "info.json"), 'r'))
        self.all_channels = self.info['ch_names']
        
        # self.all_channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
		# 		  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
		# 		  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
		# 		  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
		# 		  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
		# 		  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
		# 		  'O1', 'Oz', 'O2']

        if brain_area == 'all':
            self.selected_channels = self.all_channels
        # Occipital + Parietal
        elif brain_area == 'o+p':
            self.selected_channels = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
        elif brain_area == 'o+p+t':
            self.selected_channels = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2', 'FT9', 'FT7', 'FT8', 'FT10', 'T7', 'T8', 'TP7', 'TP9', 'TP10', 'TP8']
        # Frontal
        elif brain_area == 'f':
            self.selected_channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8']
        # Central
        elif brain_area == 'c':
            self.selected_channels = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']
        # Temporal
        elif brain_area == 't':
            self.selected_channels = ['FT9', 'FT7', 'FT8', 'FT10', 'T7', 'T8']
        # Occipital
        elif brain_area == 'o':
            self.selected_channels = ['PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
        # Parietal
        elif brain_area == 'p':
            self.selected_channels = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8']
        
        self.eeg_data_list = []
        for subject_id in tqdm(subject_ids):
            subject_dir = os.path.join(eeg_data_dir, f"sub-{subject_id:02}")
            if train:
                eeg_data_path = os.path.join(subject_dir, "train.npy")
            else:
                eeg_data_path = os.path.join(subject_dir, "test.npy")

            eeg_data = np.load(eeg_data_path)
            # shape: (1654, 10, 4, 63, 250)
            if self.average:
                # shape: (1654, 10, 63, 250)
                eeg_data = np.mean(eeg_data, axis=2)
            # select channels
            if brain_area != 'all':
                selected_idx = [self.all_channels.index(ch) for ch in self.selected_channels]
                eeg_data = np.take(eeg_data, selected_idx, axis=-2)
            self.channels_num = eeg_data.shape[-2]
            
            # Apply time window
            start, end = time_window
            end = min(end, eeg_data.shape[-1])
            eeg_data = eeg_data[..., start:end]
            
            # If it's the training set and a transform is specified, apply the EEG data transformation
            if self.frozen_eeg_prior:
                if self.eeg_transform is not None and (self.train or self.eeg_test_aug):
                    for object_idx in range(eeg_data.shape[0]):
                        for image_idx in range(eeg_data.shape[1]):
                            if not self.average:
                                for repetition_idx in range(eeg_data.shape[2]):
                                    eeg_data[object_idx, image_idx, repetition_idx] = self.eeg_transform(eeg_data[object_idx, image_idx, repetition_idx])
                            else:
                                eeg_data[object_idx, image_idx] = self.eeg_transform(eeg_data[object_idx, image_idx])

            self.eeg_data_list.append(eeg_data)
        
        self.num_subjects = len(self.eeg_data_list)
        self.num_objects = eeg_data.shape[0]
        self.num_images_per_object = eeg_data.shape[1]
        if not self.average:
            self.num_repetitions = eeg_data.shape[2]
        self.num_channels = eeg_data.shape[-2]
        self.num_sample_points = eeg_data.shape[-1]
        
        if self.image_aug:
            self.aug_image_features = []
            for aug_image_feature_dir in aug_image_feature_dirs:
                if train:
                    aug_image_feature_path = os.path.join(aug_image_feature_dir, "train.npy")
                else:
                    aug_image_feature_path = os.path.join(aug_image_feature_dir, "test.npy")
                aug_image_feature = np.load(aug_image_feature_path)
                self.aug_image_features.append(aug_image_feature)
        
        if train:
            self.image_feature_path = os.path.join(self.image_feature_dir, "image_train.npy")
        else:
            self.image_feature_path = os.path.join(self.image_feature_dir, "image_test.npy")
        if self.text_feature_dir is not None and self.text_feature_dir != '':
            if train:
                self.text_feature_path = os.path.join(self.text_feature_dir, "train.npy")
            else:
                self.text_feature_path = os.path.join(self.text_feature_dir, "test.npy")
        
        self.image_features = np.load(self.image_feature_path)
        self.feature_dim = self.image_features.shape[-1]
        if self.text_feature_dir is not None and self.text_feature_dir != '':
            self.text_features = np.load(self.text_feature_path)
    
    def __len__(self):
        if self.average and self.random:
            length = self.num_objects * self.num_images_per_object
        elif self.average and not self.random:
            length = self.num_objects * self.num_images_per_object * self.num_subjects
        elif not self.average and self.random:
            length = self.num_objects * self.num_images_per_object
        else:  # not self.average and not self.random
            length = self.num_objects * self.num_images_per_object * self.num_repetitions * self.num_subjects
        return length
        
    def __getitem__(self, index):
        # When averaging, use default 0
        repetition_idx = 0
        
        # Average & Random: Loop through objects and images, random subject
        if self.average and self.random:
            subject_idx = random.randint(0, len(self.subject_ids) - 1)
            object_idx = index // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            eeg_data = self.eeg_data_list[subject_idx][object_idx][image_idx]
        
        # Average & Not Random: Loop through all subjects and images
        elif self.average and not self.random:
            subject_idx = index // (self.num_objects * self.num_images_per_object)
            object_idx = (index % (self.num_objects * self.num_images_per_object)) // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            eeg_data = self.eeg_data_list[subject_idx][object_idx][image_idx]
        
        # Not Average & Random: Loop through objects and images, random subject and repetition
        elif not self.average and self.random:
            subject_idx = random.randint(0, self.num_subjects - 1)
            repetition_idx = random.randint(0, self.num_repetitions - 1)
            object_idx = index // self.num_images_per_object
            image_idx = index % self.num_images_per_object
            eeg_data = self.eeg_data_list[subject_idx][object_idx][image_idx][repetition_idx]
        
        # Not Average & Not Random: Complete loop through all EEG data
        else:
            subject_idx = index // (self.num_objects * self.num_images_per_object * self.num_repetitions)
            object_idx = (index % (self.num_objects * self.num_images_per_object * self.num_repetitions)) // (self.num_images_per_object * self.num_repetitions)
            image_idx = (index % (self.num_images_per_object * self.num_repetitions)) // self.num_repetitions
            repetition_idx = index % self.num_repetitions
            eeg_data = self.eeg_data_list[subject_idx][object_idx][image_idx][repetition_idx]
        
        # If it's the training set and a transform is specified, apply the EEG data transformation
        if not self.frozen_eeg_prior:
            if self.eeg_transform is not None and (self.train or self.eeg_test_aug):
                eeg_data = self.eeg_transform(eeg_data)
        
        if self.image_aug:
            if self.train or self.image_test_aug:
                aug_idx = random.randint(0, len(self.aug_image_features) - 1)
                rep_idx = random.randint(0, self.aug_image_features[0].shape[0] - 1)
                image_feature = self.aug_image_features[aug_idx][rep_idx][object_idx][image_idx]
            else:
                image_feature = self.image_features[object_idx][image_idx]
        else:
            image_feature = self.image_features[object_idx][image_idx]

        if self.text_feature_dir is not None and self.text_feature_dir != '':
            text_feature = self.text_features[object_idx][image_idx]
        else:
            text_feature = np.zeros((self.feature_dim,))
        
        return (
            torch.tensor(eeg_data, dtype=torch.float32), 
            torch.tensor(image_feature, dtype=torch.float32), 
            torch.tensor(text_feature, dtype=torch.float32),
            self.subject_ids[subject_idx], 
            object_idx, 
            image_idx, 
            repetition_idx
        )


if __name__ == '__main__':
    pass