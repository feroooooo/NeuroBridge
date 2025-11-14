import argparse
import os
import sys
import pickle
import random
import json

import mne
import numpy as np
from sklearn.utils import shuffle

from module.util import dump_pretty

# Calculate loop range of subjects and sessions
def get_loop_range(_type, num, _id):
    if _id == 0:
        loop_range = range(1, num + 1)
    elif 0 < _id <= num:
        loop_range = range(_id, _id + 1)
    else:
        print(f"invalid {_type} id")
        sys.exit()
    return loop_range

def mvnn(epoched_train, epoched_test):
    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy
    session_data = [epoched_test, epoched_train]

    ### Compute the covariance matrices ###
    # Data partitions covariance matrix of shape:
    # Data partitions × EEG channels × EEG channels
    sigma_part = np.empty((len(session_data),session_data[0].shape[2],
        session_data[0].shape[2]))
    for p in range(sigma_part.shape[0]):
        # Image conditions covariance matrix of shape:
        # Image conditions × EEG channels × EEG channels
        sigma_cond = np.empty((session_data[p].shape[0],
            session_data[0].shape[2],session_data[0].shape[2]))
        for i in tqdm(range(session_data[p].shape[0])):
            cond_data = session_data[p][i]
            # Compute covariace matrices at each time point, and then
            # average across time points
            flag = "epochs"
            if flag == "time":
                sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
                    shrinkage='auto') for t in range(cond_data.shape[2])],
                    axis=0)
            # Compute covariace matrices at each epoch (EEG repetition),
            # and then average across epochs/repetitions
            elif flag == "epochs":
                sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
                    shrinkage='auto') for e in range(cond_data.shape[0])],
                    axis=0)
        # Average the covariance matrices across image conditions
        sigma_part[p] = sigma_cond.mean(axis=0)
    # Average the covariance matrices across image partitions
    sigma_tot = sigma_part[1]
    # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

    ### Whiten the data ###
    whitened_test = np.reshape((np.reshape(session_data[0], (-1,
        session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
        @ sigma_inv).swapaxes(1, 2), session_data[0].shape)
    whitened_train = np.reshape((np.reshape(session_data[1], (-1,
        session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
            @ sigma_inv).swapaxes(1, 2), session_data[1].shape)

	## Output ##
    return whitened_train, whitened_test

# Preprocess EEG Data
def preprocess(data_path:str, data_part:str, channels_order, args, seed:int):
    eeg_data = np.load(data_path, allow_pickle=True).item()
    ch_names = eeg_data['ch_names']
    sfreq = eeg_data['sfreq']
    ch_types = eeg_data['ch_types']
    eeg_data = eeg_data['raw_eeg_data']
    
    # Convert to MNE raw format
    print("Convert to MNE raw format:")
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    del eeg_data
    
    # Get events and drop stimulation channel
    events = mne.find_events(raw, stim_channel='stim')
    channels = [ch for ch in raw.info['ch_names'] if ch != 'stim']
    print("\nAll channels:", channels)
    raw.pick(channels)
    raw.reorder_channels(channels_order)
    
    # Reject the target trials (event 99999)
    idx_target = np.where(events[:,2] == 99999)[0]
    events = np.delete(events, idx_target, 0)
    
    # Epoching and baseline correction
    print("\nEpoching and baseline correction:")
    epochs = mne.Epochs(raw, events, tmin=-args.baseline_duration, tmax=args.after_duration, baseline=(None,0), preload=True) # -.1 to 0 just for baseline correction
    del raw
    
    # Resampling
    if args.rfreq > sfreq or args.rfreq < 0:
        print("Invalid resampling frequence")
    elif args.rfreq == sfreq or args.rfreq == 0:
        print("No resampling")
    else:
        epochs.resample(args.rfreq)
            
    # Get channels and times relevant to sample points
    ch_names = epochs.info['ch_names'] # Same as channels
    times = epochs.times
    freq = epochs.info['sfreq']
    
    print("Ordered channels:", ch_names)

    # Get data, events and image condition
    data = epochs.get_data()
    events = epochs.events[:,2]
    # For training data, each 10 conditions represent an object
    img_conditions = np.unique(events) # Ordered from smallest to largest
    del epochs
    
    # Select only a maximum number of EEG repetitions
    if data_part == "test":
        max_rep = 20
    elif data_part == "train":
        max_rep = 2 # Each session only uses half of the training image

    # Sorted data matrix of shape:
    # Image conditions × EEG repetitions × EEG channels × EEG time points
    sorted_data = np.zeros((len(img_conditions), max_rep, data.shape[1], data.shape[2] - int(args.baseline_duration * freq)))
    for i in range(len(img_conditions)):
        idx = np.where(events == img_conditions[i])[0] # Find the indices of the selected image condition
        idx = shuffle(idx, random_state=seed, n_samples=max_rep) # Randomly select only the max number of EEG repetitions
        sorted_data[i] = data[idx][:, :, int(args.baseline_duration * freq):] # discard sample points before stimulus
    del data
    
    return sorted_data, img_conditions, ch_names, times, freq


# Save EEG data in npy file, all data for a subject were saved into one file.
def save_eeg_subject(epoched_data, precision, output_dir):
    # save_dir = os.path.join(output_dir, 'sub-'+format(sub,'02'))
	### Merge and save the test data ###
    for s in range(len(epoched_data["test"])):
        if s == 0:
            merged_test = epoched_data["test"][0]["data"]
        else:
            merged_test = np.append(merged_test, epoched_data["test"][s]["data"], 1)
    if precision == "fp16":
        merged_test = merged_test.astype(np.float16)
    elif precision == "fp32":
        merged_test = merged_test.astype(np.float32)
    elif precision == "fp64":
        merged_test = merged_test.astype(np.float64)
    # 200 * 80 * 63 * 250
    # reshape
    merged_test = merged_test.reshape(200, 1, 80, 63, 250)
    print("test data shape:", merged_test.shape)

    np.save(os.path.join(output_dir, "test.npy"), merged_test)

    ### Merge and save the training data ###
    for s in range(len(epoched_data["train"])):
        if s == 0:
            white_data = epoched_data["train"][0]["data"]
            img_cond = epoched_data["train"][0]["img_conditions"]
        else:
            white_data = np.append(white_data, epoched_data["train"][s]["data"], 0)
            img_cond = np.append(img_cond, epoched_data["train"][s]["img_conditions"], 0)
    # Data matrix of shape:
    # Image conditions × EGG repetitions × EEG channels × EEG time points
    merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2,
        white_data.shape[2],white_data.shape[3]))
    for i in range(len(np.unique(img_cond))):
        # Find the indices of the selected category
        idx = np.where(img_cond == i+1)[0]
        for r in range(len(idx)):
            if r == 0:
                ordered_data = white_data[idx[r]]
            else:
                ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
        merged_train[i] = ordered_data
    
    if precision == "fp16":
        merged_train = merged_train.astype(np.float16)
    elif precision == "fp32":
        merged_train = merged_train.astype(np.float32)
    elif precision == "fp64":
        merged_train = merged_train.astype(np.float64)
    # 16540 * 4 * 63 * 250
    # reshape
    merged_train = merged_train.reshape(1654, 10, 4, 63, 250)
    print("train data shape:", merged_train.shape)
    
    np.save(os.path.join(output_dir, "train.npy"), merged_train)

# Z-score normalization channel-wise
def zscore_channelwise(train_data, test_data):
    # train_data: [Cnd, Rep, Ch, T]
    axes = (0, 1, 3)  # over all trials & time, keep Ch
    train_mean = train_data.mean(axis=axes, keepdims=True)
    train_std  = train_data.std(axis=axes, keepdims=True)
    eps = 1e-8
    train_std = np.maximum(train_std, eps)
    return (train_data - train_mean)/train_std, (test_data - train_mean)/train_std


if __name__ == "__main__":
    # Get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_num', default=10, type=int, help="number of subjects (only work when sub_id is 0)")
    parser.add_argument('--sub_id', default=0, type=int, help="subject_id for preprocess, 0 means all")
    parser.add_argument('--ses_num', default=4, type=int, help="number of sessions (only work when ses_id is 0)")
    parser.add_argument('--ses_id', default=0, type=int, help="ses_id for preprocess, 0 means all")
    parser.add_argument('--rfreq', default=250, type=int, help="resampling frequency, 0 means no resample")
    parser.add_argument('--baseline_duration', default=.2, type=float, help="duration for baseline correlation")
    parser.add_argument('--after_duration', default=1.0, type=float, help="duration after stimulus")
    parser.add_argument('--raw_data_dir', default='./data/things_eeg/raw_eeg', type=str, help="raw data directory")
    parser.add_argument('--output_dir', default='./data/things_eeg/preprocessed_eeg', type=str, help="output directory")
    parser.add_argument('--mvnn', action="store_true")
    parser.add_argument('--zscore', action="store_true")
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp64", "fp32", "fp16"], help="precision: float32 or float16")
    parser.add_argument("--seed", type=int, default=20200220, help="random seed for reproducible results")
    args = parser.parse_args()

    # Print input arguments
    print('\nInput arguments:')
    for key, val in vars(args).items():
        print('{:20} {}'.format(key, val))

    # Set random seed for reproducible results
    seed = args.seed
    random.seed(seed)
    
    channels_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                    'O1', 'Oz', 'O2']
    
    sub_range = get_loop_range("subject", args.sub_num, args.sub_id)
    ses_range = get_loop_range("session", args.ses_num, args.ses_id)

    # Preprocess data of each subjects seperately
    for sub in sub_range:
        # Check if the subject has been processed
        if os.path.isdir(os.path.join(args.output_dir, 'sub-'+format(sub,'02'))) and sub != 10:
            print(f"Subject {sub} already processed, skipping...")
            continue

        epoched_data = {"train": [], "test": []}
        for ses in ses_range:
            # Load EEG data from file
            print(f"\nSubject {sub}, Session {ses}\n")
            
            eeg_dir = os.path.join(args.raw_data_dir, 'sub-' + format(sub,'02'), 'ses-' + format(ses,'02'))
            print("---Processing Data---")
            train_eeg_path = os.path.join(eeg_dir, 'raw_eeg_training.npy')
            train_data, train_img_conditions, train_ch_names, train_times, train_freq = preprocess(train_eeg_path, "train", channels_order, args, seed)
            
            test_eeg_path = os.path.join(eeg_dir, 'raw_eeg_test.npy')
            test_data, test_img_conditions, test_ch_names, test_times, test_freq = preprocess(test_eeg_path, "test", channels_order, args, seed)
            
            ch_names = train_ch_names
            times = train_times
            assert args.mvnn != args.zscore
            print("---Normalizing---")
            if args.mvnn:
                train_data, test_data = mvnn(train_data, test_data)
            if args.zscore:
                train_data, test_data = zscore_channelwise(train_data, test_data)
            epoched_data['train'].append({"data": train_data, "img_conditions": train_img_conditions, "sub_id": sub})
            epoched_data['test'].append({"data": test_data, "img_conditions": test_img_conditions, "sub_id": sub})
        print("Saving...")
        save_dir = os.path.join(args.output_dir, 'sub-'+format(sub,'02'))
         # Create the directory if not existing and save the data
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
        save_eeg_subject(epoched_data, args.precision, save_dir)

    # save ch_names and times as json
    info_dict = {
        "ch_names": ch_names,
        "times": times.tolist(),
        "baseline_duration": args.baseline_duration,
        "after_duration": args.after_duration,
        "normalization": "mvnn" if args.mvnn else "zscore",
        "sfreq": train_freq,
        "precision": args.precision,
        "seed": args.seed,
        "train_data_shape": epoched_data['train'][0]['data'].shape,
        "test_data_shape": epoched_data['test'][0]['data'].shape
    }
    with open(os.path.join(args.output_dir, "info.json"), "w", encoding="utf-8") as f:
        dump_pretty(info_dict, f, indent=4, ensure_ascii=False)