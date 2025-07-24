import os
from PIL import Image
import argparse

import numpy as np
from torchvision import transforms
import open_clip
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
import torch
from tqdm import tqdm

from module.image_augmentation import ColorJitter, HorizontalFlip, Mosaic, GrayScale, GaussianBlur, GaussianNoise, RandomCrop, LowResolution

def extract_clip(image:str, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    feature = image_features.detach().cpu().numpy()
    # shape: (1, D)
    return feature


def extract_open_clip(image, processor, model, augmentation, device):
    if augmentation is not None:
        image = augmentation(image)
    image = processor(image).unsqueeze(0).to(device)  # shape: (1, 3, H, W)
    with torch.no_grad():
        image_features = model.encode_image(image)
    feature = image_features.detach().cpu().numpy()  # shape: (1, D)
    return feature


def extract_dinov2(image:str, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_dim]
    feature = cls_embedding.detach().cpu().numpy()
    return feature


def extract_image_features(image_dir, num_images_per_object, processor, model, model_type, augmentation, device):
    image_classes = sorted(os.listdir(image_dir))
    image_list = []
    for image_class in image_classes:
        image_class_path = os.path.join(image_dir, image_class)
        image_files = sorted(os.listdir(image_class_path))
        for image_file in image_files:
            image_file_path = os.path.join(image_class_path, image_file)
            image_list.append(image_file_path)

    all_features = []
    for image in tqdm(image_list):
        image = Image.open(image).convert('RGB')
        image = image.resize((224, 224))
        if model_type == 'clip':
            feature = extract_clip(image, processor, model, device)
        elif model_type == 'open_clip':
            feature = extract_open_clip(image, processor, model, augmentation, device)
        elif model_type == 'dinov2':
            feature = extract_dinov2(image, processor, model, device)
        all_features.append(feature)
    
    all_features = np.concatenate(all_features, axis=0)
    print(all_features.shape)
    
    assert all_features.shape[0] % num_images_per_object == 0
    
    reshaped_features = all_features.reshape(-1, num_images_per_object, all_features.shape[-1])
    return reshaped_features


def extract_text_features(image_dir, processor, model, device):
    pass


def preprocess(image:Image.Image, augmentation=None):
    transform_resize = transforms.Resize((224, 224))
    image = transform_resize(image)
    if augmentation is not None:
        image = augmentation(image)
    # Convert the image to a tensor and normalize
    transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    image = transform_norm(image)
    return image


# ViT-B-16: laion2b_s34b_b88k
# ViT-B-32: laion2b_s34b_b79k
# ViT-L-14: laion2b_s32b_b82k
# ViT-H-14: laion2b_s32b_b79k
# ViT-g-14: laion2b_s34b_b88k
# ViT-bigG-14: laion2b_s39b_b160k
# RN50: openai
# RN101: openai
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, help="training device")
    parser.add_argument("--model_type", type=str, choices=["clip", "open_clip", "dinov2"], default="open_clip")
    parser.add_argument("--backbone", type=str, choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-H-14", "ViT-g-14", "ViT-bigG-14", "RN50", "RN101"], default="RN50")
    parser.add_argument("--pretrained", type=str, choices=["laion2b_s32b_b79k", "laion2b_s32b_b82k", "laion2b_s34b_b79k", "laion2b_s34b_b88k", "laion2b_s39b_b160k", "openai"], default="openai")
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--image_set_dir", type=str, default="./data/image_set")
    parser.add_argument("--output_dir", type=str, default="./data/image_feature/RN50")
    parser.add_argument("--aug_type", type=str, default="None", choices=["GaussianBlur", "GaussianNoise", "Mosaic", "RandomCrop", "LowResolution", "ColorJitter", "GrayScale", "None"])
    args = parser.parse_args()
    
    print('Input arguments:')
    for key, val in vars(args).items():
        print(f'{key:22} {val}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.model_type == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif args.model_type == "open_clip":
        model, _, processor = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pretrained, precision='fp32', device=device)
        # model, _, processor = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k', precision='fp32', device=device)
    elif args.model_type == "dinov2":
        # small base large giant
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
        model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)
    model.eval()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.aug_type == "GaussianBlur":
        augmentation = GaussianBlur(blur_kernel_size=15, fluctuation_range=0)
    elif args.aug_type == "GaussianNoise":
        augmentation = GaussianNoise(mean=0.0, std=25.0, fluctuation_range=0)
    elif args.aug_type == "Mosaic":
        augmentation = Mosaic(mosaic_level=5)
    elif args.aug_type == "RandomCrop":
        augmentation = RandomCrop(size=(224, 224))
    elif args.aug_type == "LowResolution":
        augmentation = LowResolution(scale=0.1)
    elif args.aug_type == "ColorJitter":
        augmentation = ColorJitter(s=0.5, p=1.0)
    elif args.aug_type == "GrayScale":
        augmentation = GrayScale(p=1.0)
    else:
        augmentation = None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if augmentation is None:
        train_image_features = extract_image_features(os.path.join(args.image_set_dir, "train_images"), 10, processor, model, args.model_type, augmentation, device)
        print(f"Train image feature shape: {train_image_features.shape}")
        np.save(os.path.join(args.output_dir, "image_train.npy"), train_image_features)
    else:
        train_image_features_list = []
        for i in range(args.repeat_times):
            train_image_features = extract_image_features(os.path.join(args.image_set_dir, "train_images"), 10, processor, model, args.model_type, augmentation, device)
            train_image_features_list.append(train_image_features)
        train_image_features = np.stack(train_image_features_list, axis=0)
        print(f"Train image feature shape: {train_image_features.shape}")
        np.save(os.path.join(args.output_dir, "train.npy"), train_image_features)
    
    if augmentation is None:
        test_image_features = extract_image_features(os.path.join(args.image_set_dir, "test_images"), 1, processor, model, args.model_type, augmentation, device)
        print(f"Test image feature shape: {test_image_features.shape}")
        np.save(os.path.join(args.output_dir, "image_test.npy"), test_image_features)
    else:
        test_image_features_list = []
        for i in range(args.repeat_times):
            test_image_features = extract_image_features(os.path.join(args.image_set_dir, "test_images"), 1, processor, model, args.model_type, augmentation, device)
            test_image_features_list.append(test_image_features)
        test_image_features = np.stack(test_image_features_list, axis=0)
        print(f"Test image feature shape: {test_image_features.shape}")
        np.save(os.path.join(args.output_dir, "test.npy"), test_image_features)