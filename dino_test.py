import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from dataset import NKCellDataset

import torch.nn as nn
import torch.optim as optim

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Toggle flag:
USE_DINO = False    # Set True to run DINO training, False to run CV with supervised segmentation

# =============================================================================
# 1. Time-based Stratified K-Fold Splitting Using Metadata from CSV
# =============================================================================
def load_metadata(csv_file):
    # Read the CSV containing metadata.
    metadata = pd.read_csv(csv_file)
    # Convert the timestamp column to pandas datetime.
    metadata['timestamp'] = pd.to_datetime(metadata['timestamp'].str.replace('s','', regex=False), unit='s').astype(int)
    return metadata

def assign_time_bins(metadata, n_bins=3):
    # Split the timestamp into discrete bins.
    # Here n_bins can be adjusted based on the temporal resolution you need.
    metadata = metadata.sort_values("timestamp")
    metadata["time_bin"] = pd.qcut(metadata["timestamp"], q=n_bins, labels=False)
    return metadata

def get_stratified_indices(metadata):
    # Use the binned timestamp as the stratification label
    strat_labels = metadata["time_bin"].values
    indices = np.arange(len(metadata))
    return indices, strat_labels

# =============================================================================
# 2. DINO (Self-Supervised) Training Skeleton
# =============================================================================
class DinoLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1):
        super(DinoLoss, self).__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center = torch.zeros(1, out_dim)  # center is updated during training

    def forward(self, student_output, teacher_output):
        # Scale outputs
        student_out = student_output / self.student_temp
        teacher_out = torch.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        
        # Cross-entropy between teacher and student probability distributions
        loss = -torch.mean(torch.sum(teacher_out * torch.log_softmax(student_out, dim=-1), dim=-1))
        return loss

def update_teacher(student_model, teacher_model, momentum=0.996):
    # Update teacher parameters as exponential moving average of student parameters.
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data = teacher_param.data * momentum + student_param.data * (1 - momentum)

# =============================================================================
# Main Script
# =============================================================================
if __name__ == '__main__':
    # ======= Data Transforms =======
    train_transform = T.Compose([
        T.CenterCrop((960, 960)),
        T.RandomHorizontalFlip(p=0.5),       
        T.RandomVerticalFlip(p=0.5),        
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.CenterCrop((960, 960)),
        T.ToTensor(),
    ])

    dataset_dir = "dataset"
    metadata_csv = f"{dataset_dir}/metadata.csv"  # Assume your CSV file is named metadata.csv

    # Load full dataset (this dataset loader should internally use the CSV metadata or
    # be synchronized with its ordering)
    full_dataset = NKCellDataset(dataset_dir=dataset_dir, transform=train_transform)

    # =============================================================================
    # If you want to run CV:
    # =============================================================================
    if not USE_DINO:
        # Load metadata and prepare time bins
        metadata = load_metadata(metadata_csv)
        metadata = assign_time_bins(metadata, n_bins=3)

        # Identify indices of labeled samples in the dataset
        labeled_indices = [i for i, mask_path in enumerate(full_dataset.mask_paths) if mask_path is not None]
        
        # Extract metadata only for labeled samples
        labeled_metadata = metadata.iloc[labeled_indices].copy()
        labeled_metadata.reset_index(drop=True, inplace=True)

        # Prepare stratification labels
        strat_labels = labeled_metadata["time_bin"].values
        indices = np.arange(len(labeled_metadata))

        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        overall_cv_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, strat_labels)):
            print(f"Starting fold {fold+1}/{n_splits}")
            
            # Map stratified fold indices back to original dataset
            train_indices = [labeled_indices[i] for i in train_idx]
            val_indices   = [labeled_indices[i] for i in val_idx]

            # Create subsets
            train_subset = Subset(full_dataset, train_indices)
            val_subset = Subset(full_dataset, val_indices)

            # DataLoaders
            BATCH_SIZE = 16
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

            # Model, loss, optimizer
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            criterion = DiceLoss(mode='binary', from_logits=True)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            num_epochs = 1
            patience = 10
            best_val_loss = float('inf')
            patience_counter = 0

            train_losses, val_losses = [], []

            for epoch in tqdm.tqdm(range(num_epochs), desc=f"Fold {fold+1} Epochs"):
                model.train()
                train_loss = 0.0

                for images, masks, label_flags in train_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    outputs = model(images)

                    labeled_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                    if len(labeled_idx) > 0:
                        labeled_outputs = outputs[labeled_idx]
                        labeled_masks = masks[labeled_idx]

                        loss = criterion(labeled_outputs, labeled_masks)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item() * len(labeled_idx)

                avg_train_loss = train_loss / len(train_subset)
                train_losses.append(avg_train_loss)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks, label_flags in val_loader:
                        images = images.to(device)
                        masks = masks.to(device)

                        outputs = model(images)

                        labeled_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                        if len(labeled_idx) > 0:
                            labeled_outputs = outputs[labeled_idx]
                            labeled_masks = masks[labeled_idx]
                            loss = criterion(labeled_outputs, labeled_masks)
                            val_loss += loss.item() * len(labeled_idx)

                avg_val_loss = val_loss / len(val_subset)
                val_losses.append(avg_val_loss)

                print(f"Fold {fold+1} Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Early stopping
                tolerance = 0.015
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict()
                    patience_counter = 0
                elif avg_val_loss < best_val_loss + tolerance:
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered for this fold.")
                    break

            overall_cv_results.append(best_val_loss)
            print(f"Fold {fold+1} best validation loss: {best_val_loss:.4f}\n")

        print("Cross Validation results:", overall_cv_results)
        print("Mean CV Loss:", np.mean(overall_cv_results))

    # =============================================================================
    # Else if you want to run DINO (Self-Supervised Learning)
    # =============================================================================
    else:
        # In this branch, we ignore the labels (unsupervised) and train a DINO model.
        # For simplicity, we reuse the NKCellDataset as our unlabeled dataset.
        # Note: In practice, DINO training involves a teacher and student network.
        # Below is a simplified skeleton.

        # Create DataLoader on the full dataset
        BATCH_SIZE = 16
        loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Define the network architecture for student and teacher.
        # You can use the same architecture (for example, a ResNet backbone or even a U-Net encoder part).
        student_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",   # Optionally, you may start from random weights as well.
            in_channels=3,
            classes=1,
        )
        teacher_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        student_model.to(device)
        teacher_model.to(device)
        
        # Freeze teacher model parameters at the start.
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Define the self-supervised loss.
        # Here `out_dim` should match the output dimensions of your feature head.
        # In a practical DINO setup, you would add projection heads to your network.
        loss_fn = DinoLoss(out_dim=1).to(device)
        optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
        
        num_epochs = 100
        for epoch in tqdm.tqdm(range(num_epochs), desc="DINO Training"):
            student_model.train()
            total_loss = 0.0
            for images, _, _ in loader:  # Ignore masks and label flags
                images = images.to(device)
                # Forward pass for both student and teacher networks.
                student_output = student_model(images)
                with torch.no_grad():
                    teacher_output = teacher_model(images)
                
                loss = loss_fn(student_output, teacher_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update teacher parameters using exponential moving average
                update_teacher(student_model, teacher_model, momentum=0.996)
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"DINO Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Optionally, you can save your student model (or teacher model) for feature extraction
        torch.save(student_model.state_dict(), "student_model_dino.pth")
        print("DINO training complete and model saved.")

    # =============================================================================
    # Common Evaluation / Plotting (if applicable)
    # =============================================================================
    # For CV training branch, you can plot training curves and evaluate Dice score as before.
    # For the DINO branch, since training is unsupervised, evaluation may involve clustering
    # features extracted from the network or qualitative assessment.
