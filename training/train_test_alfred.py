import os
import sys
from datetime import datetime

# Add the project root (one directory up) to Python's module search path.
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(project_root)

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.model_selection import KFold

from dataset import NKCellDataset, JointTransform
from models.unet import create_unet
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn

dice_loss = DiceLoss(mode='binary', from_logits=True)
bce_loss = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

def run_cross_validation(num_epochs=100, batch_size=16, patience=10, n_splits=5, figure_prefix=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformation pipelines for training and validation.
    train_base_transform = T.Compose([
    T.RandomCrop((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.ToTensor(),
])
    train_transform = JointTransform(train_base_transform)

    val_transform = T.Compose([
        T.CenterCrop((256, 256)),
        T.ToTensor(),
    ])
    
    # Construct dataset_dir relative to this file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = "dataset" if not os.path.exists(dataset_dir:="/work3/s234843/02466-Project/dataset") else dataset_dir
    
    # Create two dataset instances:
    full_dataset_train = NKCellDataset(dataset_dir=dataset_dir, transform=train_transform)
    full_dataset_val   = NKCellDataset(dataset_dir=dataset_dir, transform=val_transform)
    
    # Filter labeled samples (where mask_path exists on disk).
    labeled_indices = [
        i for i, mask_path in enumerate(full_dataset_train.mask_paths)
        if mask_path is not None and os.path.exists(mask_path)
    ]
    print(f"Total labeled samples: {len(labeled_indices)}")
    
    # Set up KFold cross-validation.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []  # to save the best validation loss from each fold.
    
    # Begin cross-validation.
    for fold, (train_idx, val_idx) in enumerate(kf.split(labeled_indices)):
        # Map from KFold indices to absolute dataset indices.
        train_indices = np.array(labeled_indices)[train_idx].tolist()
        val_indices   = np.array(labeled_indices)[val_idx].tolist()
        
        # Create subsets using the appropriate transformation instance.
        train_dataset = Subset(full_dataset_train, train_indices)
        val_dataset   = Subset(full_dataset_val, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Create a new model instance for this fold.
        model = create_unet(
            encoder_name="resnet50", 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=1, 
            freeze_encoder=False
        )
        model.to(device)

                # Freeze selected encoder layers (conv1, bn1, layer1)
        for name, param in model.encoder.named_parameters():
            if name.startswith("conv1") or name.startswith("bn1") or name.startswith("layer1") or name.startswith("layer2"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Check if encoder parameters are frozen or not
        frozen_params = [name for name, param in model.encoder.named_parameters() if not param.requires_grad]
        print(f"Fold {fold+1}: Frozen encoder parameters count: {len(frozen_params)} / {len(list(model.encoder.parameters()))}")
        print(f"Using pretrained weights: {any('encoder' in key and '0.weight' in key for key in model.state_dict().keys())}")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        epoch_iou_scores = []
        
        # Training loop for the fold.
        for epoch in tqdm.tqdm(range(num_epochs), desc=f"Fold {fold+1} Training"):
            model.train()
            epoch_train_loss = 0.0
            
            for images, masks, label_flags in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                # Compute loss only for labeled samples.
                labeled_batch_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                if len(labeled_batch_idx) > 0:
                    labeled_outputs = outputs[labeled_batch_idx]
                    labeled_masks = masks[labeled_batch_idx]
                    loss = dice_loss(labeled_outputs, labeled_masks)#combined_loss(labeled_outputs, labeled_masks)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_train_loss += loss.item() * len(labeled_batch_idx)
            
            num_labeled = sum(label_flags.sum().item() for _, _, label_flags in train_loader)
            avg_train_loss = epoch_train_loss / num_labeled
            train_losses.append(avg_train_loss)
            
            # Validation phase.
            model.eval()
            epoch_val_loss = 0.0
            iou_scores = []
            with torch.no_grad():
                for images, masks, label_flags in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    labeled_batch_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                    if len(labeled_batch_idx) > 0:
                        labeled_outputs = outputs[labeled_batch_idx]
                        labeled_masks = masks[labeled_batch_idx]
                        loss = combined_loss(labeled_outputs, labeled_masks)
                        epoch_val_loss += loss.item() * len(labeled_batch_idx)

                        # IOU score
                        # IOU score
                        preds = (torch.sigmoid(labeled_outputs) > 0.5).float()
                        if labeled_masks.dim() == 3:
                            labeled_masks = labeled_masks.unsqueeze(1)
                        intersection = (preds * labeled_masks).sum(dim=(1, 2, 3))
                        union = preds.sum(dim=(1, 2, 3)) + labeled_masks.sum(dim=(1, 2, 3))
                        iou = intersection / (union - intersection + 1e-8)
                        iou_scores.extend(iou.cpu().numpy())

            avg_val_loss = epoch_val_loss / len(val_indices)  # Only divide by labeled validation samples
            val_losses.append(avg_val_loss)
            mean_iou = np.mean(iou_scores)
            epoch_iou_scores.append(mean_iou)
            print(f"Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f},IoU: {mean_iou:.4f}")
            
            tolerance = 0.015
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            elif avg_val_loss < best_val_loss + tolerance:
                # Within tolerance; reset patience.
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Fold {fold+1}: Early stopping triggered at epoch {epoch+1}.")
                break
        
        fold_metrics.append({
            'fold': fold+1,
            'best_val_loss': best_val_loss,
            'best_iou': np.max(epoch_iou_scores),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'iou_scores': epoch_iou_scores
        })
        
        # Plot and save the loss curves for this fold.
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs_range, val_losses, label='Val Loss', color='orange')
        plt.title(f"Fold {fold+1} Loss Curve")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(figure_prefix+f"loss_curve_fold{fold+1}.png")
        plt.close()

        # Plot IoU vs Epochs
        epochs_range = range(1, len(val_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, epoch_iou_scores, label='Mean IoU', color='green')
        plt.title(f"Fold {fold+1} IoU Curve")
        plt.xlabel('Epoch')
        plt.ylabel('IoU Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(figure_prefix+f"iou_curve_fold{fold+1}.png")
        plt.close()
        
        print(f"Fold {fold+1} completed with best validation loss: {best_val_loss:.4f}\n")
    
    # Summary of cross-validation.
    print("Cross-validation complete. Summary of best validation losses per fold:")
    for metrics in fold_metrics:
        print(f"  Fold {metrics['fold']}: Best Validation Loss = {metrics['best_val_loss']:.4f}," f"Best IoU = {metrics['best_iou']:.4f}")

if __name__ == '__main__':
    current_time = datetime.now().strftime("%m-%d/%H-%M-%S")  # Month-Day Hour-Minute
    try:
        os.makedirs(current_time)  # Create directory to store files
    except FileExistsError:
        pass
    run_cross_validation(num_epochs=100, batch_size=8, patience=10, n_splits=5, figure_prefix=current_time+"/")
