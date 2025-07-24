# NeuroBridge
This is the official implementation for NeuroBridge.
![framework](./figure/framework.jpg)

## Environment Setup

- Python 3.12.2
- CUDA 12.6
- PyTorch 2.6.0

Create conda environment:
```bash
conda create -n neurobridge python==3.12.2
conda activate neurobridge
```
Install required depencencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
Download the Things-Image from the [OSF repository](https://osf.io/y63gw/files/osfstorage) and Things-EEG from the [OSF repository](https://osf.io/anp5v/files/osfstorage). Organize the data according to the following directory structure:
```
data
├── image_set
│   ├── train_images
│   └── test_images
└── raw_eeg
│   ├── sub-01
│   ├── ...
│   └── sub-10
```

## Data Preprocessing
Execute the following code to perform preprocessing on the raw EEG data:
```Bash
python preprocess.py --mvnn
```

## Extract Image Feature and Fuse
Run the script below to extract image features using OpenCLIP:
```Bash
/bin/bash scripts/image_feature_extract.sh
```

## Run
To run the experiments using the provided configurations, execute the following scripts.

Intra-subject: train and test on one subject
```Bash
/bin/bash scripts/intra-subjects.sh
```
Inter-subject: leave one subject out for test
```Bash
/bin/bash scripts/inter-subjects.sh
```

## Acknowledge
- [A large and rich EEG dataset for modeling human visual object recognition](https://www.alegifford.com/projects/eeg_dataset/) [THINGS-EEG]
- [Decoding Natural Images from EEG for Object Recognition](https://github.com/eeyhsong/NICE-EEG) [NICE, ICLR 2024]
- [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://github.com/dongyangli-del/EEG_Image_decode) [ATM, NeurIPS 2024]
- [CognitionCapturer: Decoding Visual Stimuli From Human EEG Signal With Multimodal Information](https://github.com/XiaoZhangYES/CognitionCapturer) [CognitionCapturer, AAAI 2025]
- [Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://github.com/HaitaoWuTJU/Uncertainty-aware-Blur-Prior) [UBP, CVPR 2025]