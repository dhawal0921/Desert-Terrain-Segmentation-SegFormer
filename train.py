import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import albumentations as A

# ==========================================
# 1. CONFIGURATION
# ==========================================
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 2           
GRAD_ACCUMULATION = 2    
EPOCHS = 50             
LEARNING_RATE = 6e-5

## ==========================================
# 1. FIXED FOLDER PATHS
# ==========================================
# Using the absolute path to bypass any directory confusion
BASE_PATH = r"[FOLDER PATH WITH DATASET & train.py]\Offroad_Segmentation_Training_Dataset"

TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train", "Color_Images")
TRAIN_MSK_DIR = os.path.join(BASE_PATH, "train", "Segmentation")
VAL_IMG_DIR   = os.path.join(BASE_PATH, "val", "Color_Images")
VAL_MSK_DIR   = os.path.join(BASE_PATH, "val", "Segmentation")

# This will save the model inside your daddy_model folder
OUTPUT_DIR = r"C:\Users\kulka\Downloads\real_daddy"

# Verification print (To be 100% sure before training starts)
if not os.path.exists(TRAIN_IMG_DIR):
    print(f"❌ ERROR: Cannot find {TRAIN_IMG_DIR}. Please check the folder name!")
else:
    print(f"✅ Paths verified! Found training folder.")

# THE MAPPING FIX: [0, 1, 2, 3, 27, 39] -> [0, 1, 2, 3, 4, 5]
LABEL_MAP = {0:0, 1:1, 2:2, 3:3, 27:4, 39:5}
id2label = {0: "Background", 1: "Sky", 2: "Ground", 3: "Vegetation", 4: "Obstacles", 5: "Landscape"}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# ==========================================
# 2. DATASET CLASS
# ==========================================
class OffroadDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # REMAP pixels
        mask = np.zeros_like(raw_mask)
        for old_val, new_val in LABEL_MAP.items():
            mask[raw_mask == old_val] = new_val

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        return {k: v.squeeze() for k, v in inputs.items()}

# ==========================================
# 3. PREPARE DATA & PATHS
# ==========================================
def get_paths(img_dir, msk_dir):
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    msks = sorted([os.path.join(msk_dir, f) for f in os.listdir(msk_dir) if f.endswith(('.jpg', '.png'))])
    return imgs, msks

train_imgs, train_msks = get_paths(TRAIN_IMG_DIR, TRAIN_MSK_DIR)
val_imgs, val_msks = get_paths(VAL_IMG_DIR, VAL_MSK_DIR)

print(f"✅ Found {len(train_imgs)} Training images.")
print(f"✅ Found {len(val_imgs)} Validation images.")

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512", size=IMAGE_SIZE)

train_dataset = OffroadDataset(train_imgs, train_msks, processor, 
                               transform=A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)]))
val_dataset = OffroadDataset(val_imgs, val_msks, processor)

# ==========================================
# 4. METRICS, MODEL & TRAINER
# ==========================================
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    logits = torch.nn.functional.interpolate(logits, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
    predictions = logits.argmax(dim=1)
    
    results = metric.compute(
        predictions=predictions, 
        references=labels, 
        num_labels=num_labels, 
        ignore_index=255, 
        reduce_labels=False
    )
    
    # NEW SECURE CONVERSION LOGIC
    processed_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            # If it's a list of values (like per-category IoU), convert each element
            processed_results[k] = v.tolist() 
        elif isinstance(v, (np.float32, np.float64)):
            # If it's a single numpy number, make it a float
            processed_results[k] = float(v)
        else:
            processed_results[k] = v
            
    return processed_results

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=6e-5,
    num_train_epochs=50,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    fp16=True,
    gradient_checkpointing=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    logging_steps=25,
    dataloader_num_workers=0
)

# 1. First, create the model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b1-finetuned-ade-512-512",
    num_labels=num_labels, 
    id2label=id2label, 
    label2id=label2id, 
    ignore_mismatched_sizes=True
)

# 2. Then, initialize the trainer
trainer = Trainer(
    model=model,  # Now 'model' is defined!
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7)]
)

trainer.train()