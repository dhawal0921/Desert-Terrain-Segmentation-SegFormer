# Offroad Semantic Segmentation

**Team Name:** The Iterators  
**Track:** Computer Vision / Semantic Segmentation  

## Project Overview
This project implements a robust Semantic Segmentation model designed for **Offroad Autonomy** using Duality AI's Falcon simulator. The goal is to accurately classify terrain types (Sky, Ground, Vegetation, Obstacles) in a synthetic desert environment to aid Unmanned Ground Vehicles (UGVs) in path planning and obstacle avoidance.

We utilized **SegFormer**, a transformer-based efficient segmentation architecture, to balance high accuracy (mIoU) with inference speed suitable for edge deployment.

## Architecture & Approach

* **Model:** SegFormer-B1 (Pre-trained on ADE20K)
* **Framework:** PyTorch & Hugging Face Transformers
* **Dataset:** Synthetic Desert Environment (FalconCloud)
* **Optimization:** Mixed Precision Training (FP16), Data Augmentation (Albumentations), and Gradient Accumulation.

## File Information

> ### train.py
>  This file [train.py](train.py), is used to train the model on the Training_Dataset.

>### test.py
>  This file [test.py](test.py), is used to train the model on the Training_Dataset.

>### trained_model
>  This folder contains the actual trained model.
  
> ### Offroad_Segmentation_Training_Dataset
>  This is the training dataset for the model.

> ### Offroad_Segmentation_testImages
>   This is the testing data for the model.
  
## Dataset & Class Mapping
The dataset consists of synthetic off-road images. Although the problem statement listed 10 classes, the dataset contained only 6 unique classes. To stabilize training, we mapped the sparse IDs to a continuous range 0-5.

| Original ID | Class Name        | New Mapped ID |
| :---        | :---              | :---          |
| 0           | Background        | 0             |
| 1           | Sky               | 1             |
| 2           | Ground            | 2             |
| 3           | Vegetation        | 3             |
| 27          | Obstacles (Rocks) | 4             |
| 39          | Landscape         | 5             |

> Note: The **"Ground" class (ID 2)** was **excluded** from **final IoU calculations** because it was **absent** in the test set.


## Configuration Details
* **Epochs**: 50 (with Early Stopping patience=7)
*	**Batch Size**: 2 (Gradient Accumulation = 2, Effective Batch = 4)
*	**Learning Rate**: 6e-5
*	**Augmentations**: Horizontal Flip (p=0.5), Random Brightness/Contrast (p=0.2)
*	**Loss Function**: Standard Cross-Entropy (handled by Hugging Face Trainer)


## Installation & Requirements

### 1. Prerequisites
* Python 3.10+
* CUDA-enabled GPU (Recommended)

### 2. Install the model (Anaconda Environment)

  Run the following command to setup an Anaconda Environment:
  ```bash
  conda create -n desert_hack python=3.9 -y
  ```
  ```bash
  conda activate desert_hack
  ```

  Run this to install PyTorch in the Environment:
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

  Run the following command to install all necessary libraries:
  ```bash
  pip install torch torchvision transformers evaluate opencv-python albumentations tqdm scipy
  ```

  Locate to your main Folder:
  ```bash
  cd [FOLDER PATH CONTAINING train.py AND THE DATASET]
  ```

### TRAINING MODEL
  **Update [train.py](train.py)**

  In train.py file, line 29,
  Do the necessary changes,
  ```bash
  BASE_PATH = r"[FOLDER PATH WITH DATASET & train.py]\Offroad_Segmentation_Training_Dataset"
  ```

  **Training:**
  Run the code below to start training.
  ```bash
  python train.py
  ```

### TESTING MODEL
  Run the code below to test the model:
  ```bash
  python test.py --model_path "[MODEL PATH]" --test_img_dir "[FOLDER PATH]\Offroad_Segmentation_testImages\Color_Images" --test_msk_dir "[FOLDER PATH]\Offroad_Segmentation_testImages\Segmentation"
  ```


## Key Results & Optimizations

### **1. Performance Metrics**
*  	**Adjusted Mean IoU:** 0.5604 (Calculated over 5 active classes).
*  	**Inference Speed:** ~39.40 ms/image (Passed <50ms limit).
  
### **2. Per-Class IoU**
*  **Obstacles:** 0.6976 (Reliable hazard detection).
*  **Sky:** 0.5012 (Robust horizon line).
*  **Vegetation:** 0.1525 (Significantly improved via logit tuning).
*  **Landscapes:** 0.9866 (Near perfect terrain segmentation)
  
### **3. Critical Optimizations**

*  **Ground Exclusion:** We explicitly handled the missing "Ground" class in the metric calculation to prevent artificial score deflation.
*  **Spatial Bias:** We utilized the structured nature of desert scenes (Sky at top, Road in middle) to inject spatial priors directly into the inference loop.
*  **Zero-Copy Inference:** All post-processing was moved to the GPU tensor level, avoiding slow CPU transfers and ensuring the speed check was passed.

