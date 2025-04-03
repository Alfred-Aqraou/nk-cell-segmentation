#from datasets import load_dataset

# from dataset import NKCellDataset
# import torch
# from torch.utils.data import DataLoader
# import torchvision.transforms as T

# from transformers import AutoModelForImageClassification
# from transformers import Trainer, TrainingArguments
# from sklearn.metrics import accuracy_score

# # Load dataset
# train_transform = T.ToTensor()
# dataset = NKCellDataset(
#     dataset_dir="dataset",
#     transform= train_transform)

# BATCH_SIZE = 16
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for images, masks, label_flags in dataloader:
#     images = images.to(device)
#     masks = masks.to(device)
#     print("Images shape:", images.shape)
#     print("Labels shape:", masks.shape)
#     print("Labels[0] shape:", len(masks[0]))
#     break


# # Load the pre-trained DINOv2 model
# model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base", num_labels=YOUR_NUM_CLASSES)

# model.classifier = torch.nn.Linear(model.config.hidden_size, YOUR_MUMS_CLASSES)

# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     save_steps=500,
#     save_total_limit=2,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["val"],
# )

# trainer.train()

# # Perform inference on validation set
# predictions = trainer.predict(dataset["val"])


# accuracy = accuracy_score(predictions.label_ids, predictions.predictions.argmax(-1))
# print(f"Accuracy: {accuracy:.4f}")



## NOTE: The above code (from the article) is made for classification. We need to freeze the classification head in order to modify the code for segmentation

## NOTE: This code below is from ChatGPT

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, TrainingArguments, Trainer
from dataset import NKCellDataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading & Transforms ---
# Using CenterCrop to extract a 960x960 region from the center of each image.
train_transform = T.Compose([
    T.CenterCrop((960, 960)),
    T.ToTensor(),
    # Optionally add normalization matching DINOv2 pre-training
])

dataset_dir = "dataset"

# Ensure your NKCellDataset returns (image, mask) pairs.
dataset = NKCellDataset(dataset_dir=dataset_dir, transform=train_transform)

full_dataset = NKCellDataset(
        dataset_dir=dataset_dir, 
        transform=train_transform
    )

# Filter labeled subset only (where mask_path is not None)
labeled_indices = [
    i for i, mask_path in enumerate(full_dataset.mask_paths) if mask_path is not None
]
labeled_dataset = torch.utils.data.Subset(full_dataset, labeled_indices)

# Train/val split
val_size = int(0.2 * len(labeled_dataset))
train_size = len(labeled_dataset) - val_size
train_dataset, val_dataset = random_split(labeled_dataset, [train_size, val_size])

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# BATCH_SIZE = 16
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inspect one batch to verify shapes.
for images, masks, label_flags in train_dataloader:
    images = images.to(device)
    masks = masks.to(device)
    print("Images shape:", images.shape)  # Expected: [B, C, 960, 960]
    print("Masks shape:", masks.shape)      # Expected: [B, 960, 960] or [B, 1, 960, 960]
    break

# --- Model Setup: DINOv2 Backbone + Segmentation Head ---
# Load the DINOv2 backbone without the classification head.
backbone = AutoModel.from_pretrained("facebook/dinov2-base")
hidden_size = backbone.config.hidden_size  # Hidden dimension from the backbone

# Define a simple segmentation head that upsamples patch tokens to the original image size.
class DinoV2SegmentationHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size // 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size // 2, hidden_size // 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_size // 4, num_classes, kernel_size=1)
        )
    
    def forward(self, features, target_height, target_width):
        # features shape: [B, N, hidden_size] where N is the number of patches.
        B, N, hidden = features.shape
        grid_size = int(N ** 0.5)  # Assumes a square grid (e.g., 16x16 patches)
        # Reshape to [B, hidden, grid_size, grid_size]
        feature_map = features.transpose(1, 2).reshape(B, hidden, grid_size, grid_size)
        x = self.decoder(feature_map)
        # Upsample to match the original (cropped) image dimensions.
        x = F.interpolate(x, size=(target_height, target_width), mode="bilinear", align_corners=False)
        return x

# Combine the backbone and the segmentation head.
class DinoV2SegmentationModel(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes):
        super().__init__()
        self.backbone = backbone
        self.segmentation_head = DinoV2SegmentationHead(hidden_size, num_classes)
    
    def forward(self, x):
        # Save the original image spatial dimensions (960x960 after cropping).
        target_height, target_width = x.shape[2], x.shape[3]
        outputs = self.backbone(x)
        features = outputs.last_hidden_state  # [B, N, hidden_size]
        segmentation_logits = self.segmentation_head(features, target_height, target_width)
        return segmentation_logits

# Define the number of segmentation classes.
YOUR_NUM_CLASSES = 3  # Update with your actual number of classes

# Create the full segmentation model.
model = DinoV2SegmentationModel(backbone, hidden_size, YOUR_NUM_CLASSES)
model.to(device)

# --- Training Setup ---
training_args = TrainingArguments(
    output_dir="./results_dino_segmentation",
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

# Define a placeholder compute_metrics function.
# Replace with an actual segmentation metric (e.g., mean IoU, Dice score).
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(dim=1)
    return {"mean_iou": 0.0}

# Note: Ensure your dataset contains "train" and "val" splits.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
)

# --- Training & Inference ---
trainer.train()
predictions = trainer.predict(dataset["val"])
print("Inference complete.")

