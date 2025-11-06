import os
import pickle
import shutil
import argparse

import mne
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default='./data/things_meg', type=str, help="raw data directory")
    parser.add_argument('--output_dir', default='./data/things_meg', type=str, help="output directory")
    parser.add_argument('--zscore', action="store_true")
    args = parser.parse_args()

    # Print input arguments
    print('\nInput arguments:')
    for key, val in vars(args).items():
        print('{:20} {}'.format(key, val))

    meg_dir = os.path.join(args.raw_data_dir, 'raw_meg')
    image_dir = os.path.join(args.raw_data_dir, 'image_set', 'object_images')

    image_concept_list = []
    total_images = 0
    concept_idx = 1
    file_list = []

    for concept_name in sorted(os.listdir(image_dir)):
        concept_path = os.path.join(image_dir, concept_name)
        if os.path.isdir(concept_path):
            files = sorted(os.listdir(concept_path))
            total_images += len(files)
            file_list.extend([os.path.join(concept_path, f) for f in files])
            image_concept_list.extend([concept_idx] * len(files))
            concept_idx += 1

    print(f"Number of concepts: {concept_idx - 1}")
    print(f"There are a total of {total_images} files in the image directory '{image_dir}'")

    for sub_id in range(1, 5):
        fif_file = os.path.join(meg_dir,f"preprocessed_P{sub_id}-epo.fif")

        def read_and_crop_epochs(fif_file):
            epochs = mne.read_epochs(fif_file, preload=True)
            cropped_epochs = epochs.crop(tmin=0, tmax=1.0)
            return cropped_epochs

        epochs = read_and_crop_epochs(fif_file)    

        sorted_indices = np.argsort(epochs.events[:, 2])
        epochs = epochs[sorted_indices]

        print(len(epochs.events))
        # Verify the shape of the epochs data
        ch_names = epochs.ch_names
        print(len(ch_names))
        print(ch_names)

        def filter_valid_epochs(epochs, exclude_event_id=999999):
            return epochs[epochs.events[:, 2] != exclude_event_id]
        valid_epochs = filter_valid_epochs(epochs)

        def identify_zs_event_ids(epochs, num_repetitions=12):
            event_ids = epochs.events[:, 2]
            unique_event_ids, counts = np.unique(event_ids, return_counts=True)
            zs_event_ids = unique_event_ids[counts == num_repetitions]
            return zs_event_ids

        zs_event_ids = identify_zs_event_ids(valid_epochs)
        # Verify the zero-shot event IDs
        print("Zero-shot Event IDs:", zs_event_ids)

        # Separate and process datasets
        training_epochs = valid_epochs[~np.isin(valid_epochs.events[:, 2], zs_event_ids)]
        # Verify the number of events in the training set
        print("Number of events in the training set:", len(training_epochs.events))
        print(len(training_epochs.events))

        # Extract event IDs from the filtered training epochs
        training_event_ids = np.unique(training_epochs.events[:, 2])

        # # Check for any overlap between zero-shot and training event IDs
        # overlap_ids = np.intersect1d(zs_event_ids, training_event_ids)

        # # Print the overlap, if any
        # print("Overlapping Event IDs:", overlap_ids)

        zs_test_epochs = valid_epochs[np.isin(valid_epochs.events[:, 2], zs_event_ids)]
        print(len(zs_test_epochs.events))

        zs_event_to_category_map = {}
        for i, event_id in enumerate(zs_event_ids):
            # Using the row index (i) to map to the image category index
            # Assuming the first event_id corresponds to the first row, second event_id to the second row, and so on
            image_category_index = image_concept_list[event_id-1]
            
            zs_event_to_category_map[event_id] = image_category_index
            
        test_set_categories = []
        # Iterate over the event IDs in the test set
        for event_id in zs_event_ids:
            if event_id in zs_event_to_category_map:
                # Get the category index from the mapping
                category_index = zs_event_to_category_map[event_id]
                test_set_categories.append(category_index)
                
        event_to_category_map = {}
        for i, event_id in enumerate(training_event_ids):
            # Using the row index (i) to map to the image category index
            # Assuming the first event_id corresponds to the first row, second event_id to the second row, and so on
            image_category_index = image_concept_list[event_id-1]
            
            event_to_category_map[event_id] = image_category_index
            
        train_set_categories = []
        # Extract event IDs from the training set
        training_event_ids = training_epochs.events[:, 2]
        # Iterate over the event IDs in the training set
        for event_id in training_event_ids:
            if event_id in event_to_category_map:
                # Get the category index from the mapping
                category_index = event_to_category_map[event_id]        
                train_set_categories.append(category_index)

        train_set_categories_filtered = [item for item in train_set_categories if item not in test_set_categories]
        len(train_set_categories_filtered)

        # Create a mask for epochs to keep in the training set
        keep_epochs_mask = [category not in test_set_categories for category in train_set_categories]
        # Apply the mask to filter out epochs from training_epochs
        training_epochs_filtered = training_epochs[keep_epochs_mask]

        def reshape_meg_data(epochs, train):
            if train:
                final_data = epochs.get_data()[:, :, :-1].reshape(1654, 12, 271, 200)[:, :, np.newaxis, :, :]
            else:
                final_data = epochs.get_data()[:, :, :-1].reshape(200, 12, 271, 200)[:, np.newaxis, :, :]
            return final_data

        train_data = reshape_meg_data(training_epochs_filtered, train=True)
        print(train_data.shape)
        test_data = reshape_meg_data(zs_test_epochs, train=False)
        print(test_data.shape)
        
        if args.zscore:
            # zscore on train (channel wise)
            mean_train = np.mean(train_data, axis=(0,1,2,4), keepdims=True)
            std_train = np.std(train_data, axis=(0,1,2,4), keepdims=True)
            train_data = (train_data - mean_train) / std_train
            test_data = (test_data - mean_train) / std_train
            print("Z-score normalization applied.")

        times = np.linspace(0, 0.995, 200)

        train_dict = {
            'data': train_data.astype(np.float32),
            'ch_names': [],
            'times': times
        }

        test_dict = {
            'data': test_data.astype(np.float32),
            'ch_names': [],
            'times': times
        }

        save_dir = os.path.join(args.output_dir, 'preprocessed_meg', 'sub-'+format(sub_id,'02'))
        os.makedirs(save_dir, exist_ok=True)
        file_name_test = 'test.npy'
        file_name_train = 'train.npy'

        save_pic = open(os.path.join(save_dir, file_name_train), 'wb')
        pickle.dump(train_dict, save_pic, protocol=4)
        save_pic.close()

        save_pic = open(os.path.join(save_dir, file_name_test), 'wb')
        pickle.dump(test_dict, save_pic, protocol=4)
        save_pic.close()

    ################################################################################
    # processing image files

    # load csv file
    path = os.path.join(args.raw_data_dir, "sourcedata/sample_attributes_P1.csv")
    df = pd.read_csv(path)

    print(f"CSV file '{path}' has been loaded, containing {len(df)} rows.")

    # filter train images
    train_df = df[(df['category_nr'] <= 1854) & (df['test_image_nr'].isna())]

    # filter test images
    test_df = df[(df['category_nr'] <= 1854) & (df['trial_type'] == 'test')]

    # ensure no overlap in categories between train and test sets
    train_df = train_df[~train_df['category_nr'].isin(test_df['category_nr'])]

    # sort by category_nr
    train_df = train_df.sort_values(by=['category_nr'])
    test_df = test_df.sort_values(by=['category_nr'])

    base_dir = os.path.join(args.raw_data_dir, 'image_set/object_images')

    print(f"Training set has {len(train_df)} images.")
    dst_dir = os.path.join(args.output_dir, 'image_set/train_images')
    for index, row in train_df.iterrows():
        category_nr = str(row['category_nr']).zfill(5)
        src_path = row['image_path'].replace('images_meg/', base_dir + '/')
        concept = os.path.basename(os.path.dirname(src_path))
        dst_path = f"{dst_dir}/{category_nr}_{concept}/{os.path.basename(src_path)}"
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

    print(f"Test set has {len(test_df)} images.")
    dst_dir = os.path.join(args.output_dir, 'image_set/test_images')
    for index, row in test_df.iterrows():
        category_nr = str(row['category_nr']).zfill(5)
        src_path = row['image_path'].replace('images_test_meg/', base_dir + '/')
        concept = os.path.basename(src_path).rsplit('_', 1)[0]
        src_path = os.path.join(os.path.dirname(src_path), concept, os.path.basename(src_path))
        dst_path = f"{dst_dir}/{category_nr}_{concept}/{os.path.basename(src_path)}"
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)