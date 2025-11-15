import os
import argparse
import logging
from datetime import datetime
import json
import random
import time
import sys
import shutil

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from module.dataset import EEGPreImageDataset
from module.eeg_encoder.atm.atm import ATMS
from module.eeg_encoder.model import EEGNet, EEGProject, TSConv, EEGTransformer
from module.loss import ContrastiveLoss
from module.util import retrieve_all
from module.projector import *
from module.eeg_augmentation import RandomTimeShift, RandomGaussianNoise, RandomChannelDropout, RandomSmooth

# Set the random seed. If not provided, generate a new one based on the current time.
def seed_everything(seed: int = None):
    if seed is None:
        # If no seed is provided, generate a new one based on the current time
        seed = int(time.time()) % (2**32 - 1)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[seed_everything] All seeds set to: {seed}")
    return seed

if __name__ == '__main__':
    # Get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help='training device')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--output_dir', default='./result', type=str)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--train_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--test_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--data_average', action='store_true')
    parser.add_argument('--data_random', action='store_true')
    parser.add_argument('--init_temperature', default=0.07, type=float)
    parser.add_argument('--t_learnable', action='store_true')
    parser.add_argument('--softplus', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--img_l2norm', action='store_true')
    parser.add_argument('--text_l2norm', action='store_true')
    parser.add_argument('--eeg_l2norm', action='store_true')
    parser.add_argument('--eeg_data_dir', default='./things_eeg/data/preprocessed_eeg', type=str, help='where your EEG data are')
    parser.add_argument("--selected_channels", default=[], nargs='*', type=str, help="selected EEG channels, empty means all channels")
    parser.add_argument('--time_window', type=int, default=[0, 250], nargs=2, help='time window for EEG data, in sample points')
    parser.add_argument('--eeg_aug', action='store_true')
    parser.add_argument('--eeg_aug_type', type=str, choices=['noise', 'time_shift', 'channel_dropout', 'smooth'], default='noise', help='eeg augmentation type')
    parser.add_argument('--eeg_encoder_type', type=str, choices=['ATM', "EEGNet", "EEGProject", "TSConv", "EEGTransformer"], default='EEGProject')
    parser.add_argument('--image_aug', action='store_true')
    parser.add_argument('--image_test_aug', action='store_true')
    parser.add_argument('--eeg_test_aug', action='store_true')
    parser.add_argument('--frozen_eeg_prior', action='store_true', help='whether to use frozen eeg prior')
    
    parser.add_argument('--image_data_dir', default='./data/things_eeg/image_set/train_images', type=str, help='where your image data are')
    
    parser.add_argument('--projector', type=str, choices=['direct', 'linear', 'mlp'], default='direct')
    parser.add_argument('--feature_dim', type=int, default=512, help='dont work when direct')

    parser.add_argument('--image_feature_dir', default='./data/things_eeg/image_feature/RN50', type=str, help='where your image feature are')
    parser.add_argument('--aug_image_feature_dirs', default=[], nargs='+', type=str, help='where your augmentation image feature are')
    parser.add_argument('--text_feature_dir', default='./data/things_eeg/text_feature/BLIP2', type=str, help='where your text feature are')

    parser.add_argument('--save_weights', action='store_true', help='whether to save model weights')
    
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

    args = parser.parse_args()
    
    seed = seed_everything(seed=args.seed)
    
    if args.output_name is not None:
        log_dir = os.path.join(args.output_dir, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}-{args.output_name}")
    else:
        log_dir = os.path.join(args.output_dir, datetime.now().strftime(r'%Y%m%d-%H%M%S'))
        
    log_dir_suffix = '-'.join(log_dir.split('-')[2:])
    if os.path.exists(args.output_dir):
        for existing_dir in os.listdir(args.output_dir):
            if existing_dir.endswith(log_dir_suffix):
                if os.path.exists(os.path.join(args.output_dir, existing_dir, "result.csv")):
                    print(f"Experiment with the same name '{log_dir_suffix}' already exists. Exiting to avoid overwriting.")
                    sys.exit(0)
                else:
                    shutil.rmtree(os.path.join(args.output_dir, existing_dir))
                    print(f"Removed incomplete experiment directory '{existing_dir}' to avoid conflicts.")

    writer = SummaryWriter(log_dir=log_dir)
    
    # Save the configuration to a JSON file
    args_dict = vars(args)
    with open(os.path.join(writer.log_dir, "train_config.json"), 'w') as f:
        json.dump(args_dict, f, indent=4)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8',
        filename=f'{writer.log_dir}/train.log',
        filemode='w'
    )

    def log(str):
        logging.info(str)
        print(str)

    log('Input arguments:')
    for key, val in vars(args).items():
        log(f'{key:22} {val}')
        
    with open(os.path.join(args.output_dir, "last_run.txt"), 'w') as f:
        f.write(writer.log_dir)
        
    print("")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')

    print('\n>>> Loading Train Data <<<')
    if args.eeg_aug:
        if args.eeg_aug_type == 'noise':
            eeg_transform = RandomGaussianNoise(std=0.001)
        elif args.eeg_aug_type == 'time_shift':
            eeg_transform = RandomTimeShift(max_shift=5)
        elif args.eeg_aug_type == 'channel_dropout':
            eeg_transform = RandomChannelDropout(drop_prob=0.1)
        elif args.eeg_aug_type == 'smooth':
            eeg_transform = RandomSmooth(kernel_size=5, smooth_prob=0.3)
    else:
        eeg_transform = None
    
    train_dataset = EEGPreImageDataset(args.train_subject_ids, args.eeg_data_dir, args.selected_channels, args.time_window, args.image_feature_dir, args.text_feature_dir, args.image_aug, args.aug_image_feature_dirs, args.data_average, args.data_random, eeg_transform, True, args.image_test_aug, args.eeg_test_aug, args.frozen_eeg_prior)
    
    eeg_sample_points = train_dataset.num_sample_points
    log(f'EEG sample points: {eeg_sample_points}')
    feature_dim = train_dataset.image_features.shape[-1]
    log(f'feature dimension: {feature_dim}')

    log(f'data length: {len(train_dataset)}')
    channels_num = train_dataset.channels_num
    log(f'number of channels: {channels_num}')

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    print('\n>>> Loading Test Data <<<')
    test_dataset = EEGPreImageDataset(args.test_subject_ids, args.eeg_data_dir, args.selected_channels, args.time_window, args.image_feature_dir, args.text_feature_dir, args.image_aug, args.aug_image_feature_dirs, True, False, eeg_transform, False, args.image_test_aug, args.eeg_test_aug, args.frozen_eeg_prior)
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    
    args_dict = vars(args)
    keys_needed = ['eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir']
    inference_config = {k: args_dict[k] for k in keys_needed}
    inference_config['eeg_sample_points'] = eeg_sample_points
    inference_config['feature_dim'] = feature_dim
    with open(os.path.join(writer.log_dir, "evaluate_config.json"), 'w') as f:
        json.dump(inference_config, f, indent=4)
    
    if args.eeg_encoder_type == 'ATM':
        model = ATMS(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGNet':
        model = EEGNet(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGProject':
        model = EEGProject(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'TSConv':
        model = TSConv(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGTransformer':
        model = EEGTransformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    model = model.to(device)

    # Output the number of trainable parameters in the model (formatted in millions)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_m = num_params / 1e6
    log(f'EEG Encoder trainable parameters: {num_params_m:.2f}M')
    
    log(str(model))
    
    if args.projector == 'direct':
        eeg_projector = ProjectorDirect().to(device)
        img_projector = ProjectorDirect().to(device)
        text_projector = ProjectorDirect().to(device)
    elif args.projector == 'linear':
        eeg_projector = ProjectorLinear(feature_dim, args.feature_dim).to(device)
        img_projector = ProjectorLinear(feature_dim, args.feature_dim).to(device)
        text_projector = ProjectorLinear(feature_dim, args.feature_dim).to(device)
    elif args.projector == 'mlp':
        eeg_projector = ProjectorMLP(feature_dim, args.feature_dim).to(device)
        img_projector = ProjectorMLP(feature_dim, args.feature_dim).to(device)
        text_projector = ProjectorMLP(feature_dim, args.feature_dim).to(device)
        
    num_params = sum(p.numel() for p in eeg_projector.parameters() if p.requires_grad) + sum(p.numel() for p in img_projector.parameters() if p.requires_grad) + sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
    num_params_m = num_params / 1e6
    log(f'Projector trainable parameters: {num_params_m:.2f}M')
    
    criterion = ContrastiveLoss(args.init_temperature, args.alpha, args.beta, args.eeg_l2norm, args.img_l2norm, args.text_l2norm, args.t_learnable, args.softplus).to(device)
    log(str(criterion))
    
    # Collect all trainable parameters from both model and criterion
    trainable_parameters = list(model.parameters()) + list(eeg_projector.parameters()) + list(img_projector.parameters()) + list(text_projector.parameters())
    if args.t_learnable:  # Only add criterion parameters if temperature is learnable
        trainable_parameters.extend([p for p in criterion.parameters() if p.requires_grad])
    
    optimizer = optim.AdamW(trainable_parameters, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

    # Training process
    model.train()
    eeg_projector.train()
    img_projector.train()
    text_projector.train()
    best_loss = float('inf')
    best_epoch = 0
    best_test_loss = float('inf')
    best_test_epoch = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs} [Train]"):
            eeg_batch = batch[0].to(device)
            subject_id_batch = batch[3].to(device)
            image_feature_batch = batch[1].to(device)
            text_feature_batch = batch[2].to(device)

            optimizer.zero_grad()
            if args.eeg_encoder_type in ['ATM', 'eegproject', 'nice']:
                eeg_feature_batch = model(eeg_batch, subject_id_batch)
            else:
                eeg_feature_batch = model(eeg_batch)
            
            eeg_feature_batch = eeg_projector(eeg_feature_batch)
            image_feature_batch = img_projector(image_feature_batch)
            text_feature_batch = text_projector(text_feature_batch)
            
            loss = criterion(eeg_feature_batch, image_feature_batch, text_feature_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Release GPU memory
            # del eeg_batch, image_feature_batch
            # torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        log(f"Epoch [{epoch}/{args.num_epochs}] Train Loss: {avg_loss:.4f}")
        if args.save_weights:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'eeg_projector_state_dict': eeg_projector.state_dict(),
                'img_projector_state_dict': img_projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, f"{writer.log_dir}/checkpoint_last.pth")

        model.eval()
        eeg_projector.eval()
        img_projector.eval()
        text_projector.eval()
        total_test_loss = 0.0
        eeg_feature_list = []
        image_feature_list = []

        with torch.no_grad():
            for batch in test_dataloader:
                eeg_batch = batch[0].to(device)
                subject_id_batch = batch[3].to(device)
                image_feature_batch = batch[1].to(device)
                text_feature_batch = batch[2].to(device)

                if args.eeg_encoder_type in ['ATM']:
                    eeg_feature_batch = model(eeg_batch, subject_id_batch)
                else:
                    eeg_feature_batch = model(eeg_batch)
                
                eeg_feature_batch = eeg_projector(eeg_feature_batch)
                image_feature_batch = img_projector(image_feature_batch)
                text_feature_batch = text_projector(text_feature_batch)
                
                loss = criterion(eeg_feature_batch, image_feature_batch, text_feature_batch)
                total_test_loss += loss.item()

                eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
                image_feature_list.append(image_feature_batch.cpu().numpy())

                # del eeg_batch, image_feature_batch
                # torch.cuda.empty_cache()

        avg_test_loss = total_test_loss / len(test_dataloader)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)

        # Concatenate all EEG and image features for retrieval
        eeg_feature_all = np.concatenate(eeg_feature_list, axis=0)
        image_feature_all = np.concatenate(image_feature_list, axis=0)
        top5_count, top1_count, total = retrieve_all(eeg_feature_all, image_feature_all, args.data_average)
        top5_acc = top5_count / total * 100
        top1_acc = top1_count / total * 100
        log(f"top5 acc {top5_acc:.2f}%\ttop1 acc {top1_acc:.2f}%\tTest Loss: {avg_test_loss:.4f}")

        # Save the best model
        is_better = False
        if avg_test_loss < best_test_loss:
            is_better = True
        elif avg_test_loss == best_test_loss and top1_acc > best_top1_acc:
            # Use top1 accuracy as a secondary criterion when the loss is the same
            is_better = True

        if is_better:
            best_test_loss = avg_test_loss
            best_top5_acc = top5_acc
            best_top1_acc = top1_acc
            best_test_epoch = epoch
            if args.save_weights:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'eeg_projector_state_dict': eeg_projector.state_dict(),
                    'img_projector_state_dict': img_projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_test_loss
                }, f"{writer.log_dir}/checkpoint_test_best.pth")
    
    result_dict = {}
    result_dict['top1 acc'] = f'{top1_acc:.2f}'
    result_dict['top5 acc'] = f'{top5_acc:.2f}'
    result_dict['best top1 acc'] = f'{best_top1_acc:.2f}'
    result_dict['best top5 acc'] = f'{best_top5_acc:.2f}'
    result_dict['best test loss'] = f'{best_test_loss:.4f}'
    result_dict['best epoch'] = best_test_epoch
    df = pd.DataFrame(result_dict, index=[0])
    df.to_csv(os.path.join(log_dir, 'result.csv'), index=False)
    
    log(f'best test loss: {best_test_loss:.4f} top5 acc: {best_top5_acc:.2f} top1 acc: {best_top1_acc:.2f} at epoch {best_test_epoch}')