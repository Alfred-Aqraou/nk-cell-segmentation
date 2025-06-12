import os, sys, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.unet import create_unet_with_dino_resnet18
from datetime import datetime
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.model_selection import KFold

from dataset import NKCellDataset, JointTransform, extract_hour_from_filename
from models.unet import create_unet
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
import torch.nn as nn

from solo.methods.dino import DINO
import torchvision.models as models

dice_loss = DiceLoss(mode='binary', from_logits=True)
bce_loss = nn.BCEWithLogitsLoss()
iou_loss = JaccardLoss(mode='binary', from_logits=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help = 'Specify epochs. I.e: 100', required=True, type=int)
parser.add_argument("--batchsize", help = 'Specify batch size. I.e: 16', required=True, type=int)
parser.add_argument("--loss", help = 'Specify loss type. I.e: "dice"', required=True, type=str)
parser.add_argument("--encoder", help = 'Specify encoder, you can also add a ckpt path. I.e: "resnet50"', required=True, type=str)
parser.add_argument("--freeze", help = 'Specify whether to use freeze_opts. I.e: True', required=True, type=bool)
parser.add_argument("--optimizer", help = 'Specify optimizer. I.e: "adam"', required=True, type=str)
parser.add_argument("--lr", help = 'Specify learning rate. I.e: 0.0005', required=True, type=float)
parser.add_argument("--resize", help = 'Specify dimensions to resize to. I.e: 256 256', nargs='+', required=True, type=int)
parser.add_argument("--nsplits", help = 'Specify outer n splits. I.e: 5', required=True, type=int)

args = parser.parse_args()
print("Given arguments:")
print(args)

args.freeze = str(args.freeze).strip().lower() in ("true", "1", "yes", "y", "t")

args.resize = tuple(args.resize)
for i in args.resize:
    assert isinstance(i, int)

if ".ckpt" in args.encoder:
    assert os.path.isfile(args.encoder)

PARAMETERS = {
    "EPOCHS": args.epochs,
    "batch_size": args.batchsize,
    "loss_type": args.loss, # 'dice', 'combined', 'iou
    "encoder": args.encoder,
    "freeze_opt": args.freeze,
    "optimizer": args.optimizer,
    "learning_rate": args.lr,
    "resize": args.resize,
    "outer_n_splits": args.nsplits
    }

def combined_loss(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

def get_loss_function(loss_type):
    if loss_type == "dice":
        return lambda pred, target: dice_loss(pred, target)
    elif loss_type == "bce":
        return lambda pred, target: bce_loss(pred, target)
    elif loss_type == 'iou':
        return lambda pred, target: iou_loss(pred, target)
    else:
        return combined_loss
    
def extract_encoder_name(encoder):
    if not isinstance(encoder, str):
        return str(encoder)
    if "dino" in encoder and ".ckpt" in encoder:
        match = re.search(r'dino/([^/]+)/dino-\1-ep=(\d+)\.ckpt', encoder)
        if match:
            return f"dino_{match.group(1)}_e{match.group(2)}"
    return encoder  # Return original if not a dino path or no match

def run_cross_validation(num_epochs, batch_size, n_splits, patience=10, figure_prefix=""):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    global_iou_per_image = {}  # img_path → iou
    # Define transformation pipelines for training and validation.
    train_base_transform = T.Compose([
    T.RandomCrop(PARAMETERS["resize"]),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.ToTensor(),
])
    train_transform = JointTransform(train_base_transform)

    val_transform = T.Compose([
        T.CenterCrop(PARAMETERS["resize"]),
        T.ToTensor(),
    ])
    
    # Construct dataset_dir relative to this file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
    "/Users/aliafif/Desktop/Code/Fagprojekt/02466-Project/dataset",
    "/work3/s234843/02466-Project/dataset",
    os.path.join(os.path.dirname(__file__), "dataset")
    ]
    dataset_dir = next((p for p in possible_paths if os.path.exists(p)), None)
    if dataset_dir is None:
        raise FileNotFoundError("Dataset directory not found in any of the expected locations.")
    
    # Create two dataset instances:
    full_dataset_train = NKCellDataset(dataset_dir=dataset_dir, transform=train_transform)
    full_dataset_val   = NKCellDataset(dataset_dir=dataset_dir, transform=val_transform)
    
    # Filter labeled samples (where mask_path exists on disk).
    labeled_indices = [
        i for i, mask_path in enumerate(full_dataset_train.mask_paths)
        if mask_path is not None and os.path.exists(mask_path)
    ]
    print(f"Total labeled samples: {len(labeled_indices)}")
    
    global_hours = []
    global_ious = []
    # Set up KFold cross-validation.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []  # to save the best validation loss from each fold.
    
    all_args = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(labeled_indices)):
        args = (fold, train_idx, val_idx, labeled_indices, dataset_dir, device, batch_size, num_epochs, patience, figure_prefix)
        all_args.append(args)

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

        if ".ckpt" in PARAMETERS['encoder']:
            model = create_unet_with_dino_resnet18(
                ckpt_path=os.path.abspath(PARAMETERS['encoder']),
                in_channels=3,
                classes=1,
                freeze_encoder=PARAMETERS['freeze_opt']
            ).to(device)
        else:
            model = create_unet(
                encoder_name=PARAMETERS['encoder'],
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                freeze_encoder=PARAMETERS['freeze_opt']
            ).to(device)

        model.to(device)

        # Check if encoder parameters are frozen or not
        frozen_params = [name for name, param in model.encoder.named_parameters() if not param.requires_grad]
        print(f"Fold {fold+1}: Frozen encoder parameters count: {len(frozen_params)} / {len(list(model.encoder.parameters()))}")
        print(f"Using pretrained weights: {any('encoder' in key and '0.weight' in key for key in model.state_dict().keys())}")

        if PARAMETERS['optimizer'] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=PARAMETERS['learning_rate'])
        elif PARAMETERS['optimizer'] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=PARAMETERS['learning_rate'], momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {PARAMETERS['optimizer']}. Available optimizers: 'adam', 'sgd'")
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        epoch_iou_scores = []

        loss_fn = get_loss_function(PARAMETERS['loss_type'])
        
        # Training loop for the fold.
        for epoch in tqdm.tqdm(range(num_epochs), desc=f"Fold {fold+1} Training"):
            model.train()
            epoch_train_loss = 0.0
            
            for images, masks, label_flags, img_paths in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                # Compute loss only for labeled samples.
                labeled_batch_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                if len(labeled_batch_idx) > 0:
                    labeled_outputs = outputs[labeled_batch_idx]
                    labeled_masks = masks[labeled_batch_idx]
                    loss = loss_fn(labeled_outputs, labeled_masks) #dice_loss(labeled_outputs, labeled_masks)#combined_loss(labeled_outputs, labeled_masks)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_train_loss += loss.item() * len(labeled_batch_idx)
            
            num_labeled = sum(label_flags.sum().item() for _, _, label_flags, _ in train_loader)
            avg_train_loss = epoch_train_loss / num_labeled
            train_losses.append(avg_train_loss)
            
            # Validation phase.
            model.eval()
            epoch_val_loss = 0.0
            iou_scores = []
            fold_iou_list = []

            per_image_running_sum = {}
            per_image_count = {}

            with torch.no_grad():
                for images, masks, label_flags, img_paths in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    labeled_batch_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                    if len(labeled_batch_idx) > 0:
                        labeled_outputs = outputs[labeled_batch_idx]
                        labeled_masks = masks[labeled_batch_idx]
                        loss = loss_fn(labeled_outputs, labeled_masks)
                        epoch_val_loss += loss.item() * len(labeled_batch_idx)

                        # IOU score
                        preds = (torch.sigmoid(labeled_outputs) > 0.5).float()
                        if labeled_masks.dim() == 3:
                            labeled_masks = labeled_masks.unsqueeze(1)
                        intersection = (preds * labeled_masks).sum(dim=(1, 2, 3))
                        union = preds.sum(dim=(1, 2, 3)) + labeled_masks.sum(dim=(1, 2, 3))
                        iou = intersection / (union - intersection + 1e-8)
                        ious = iou.cpu().numpy()
                        iou_scores.extend(ious)

                        # For each labeled sample, map IOU to its hour:
                        for idx_in_batch, sample_iou in zip(labeled_batch_idx.cpu().numpy(), ious):
                            img_path = img_paths[idx_in_batch]
                            hour = extract_hour_from_filename(img_path)
                            if hour is not None:
                                if img_path not in per_image_running_sum:
                                    per_image_running_sum[img_path] = 0.0
                                    per_image_count[img_path] = 0
                                per_image_running_sum[img_path] += sample_iou
                                per_image_count[img_path] += 1

            avg_val_loss = epoch_val_loss / len(val_indices)  # Only divide by labeled validation samples
            val_losses.append(avg_val_loss)
            mean_iou = np.mean(iou_scores)
            epoch_iou_scores.append(mean_iou)
            print(f"Fold {fold+1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f},IoU: {mean_iou:.4f}")
            
            for img_path in per_image_running_sum:
                hour = extract_hour_from_filename(img_path)
                if hour is not None:
                    mean_iou = per_image_running_sum[img_path] / per_image_count[img_path]
                    global_hours.append(hour)
                    global_ious.append(mean_iou)

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
        plt.ylabel(f'Loss ({PARAMETERS['loss_type']})')
        plt.legend()
        plt.grid(True)
        plt.savefig(figure_prefix+f"{extract_encoder_name(PARAMETERS["encoder"])}_loss_curve_fold{fold+1}.png")
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
        plt.savefig(figure_prefix+f"{extract_encoder_name(PARAMETERS["encoder"])}_iou_curve_fold{fold+1}.png")
        plt.close()
        
        print(f"Fold {fold+1} completed with best validation loss: {best_val_loss:.4f}\n")
        # Save per-image IoU for this fold to disk
        np.save(figure_prefix + f"{extract_encoder_name(PARAMETERS["encoder"])}_per_image_iou_fold{fold+1}.npy", np.array(fold_iou_list))
    
    # Summary of cross-validation.
    print("Cross-validation complete. Summary of best validation losses per fold:")
    for metrics in fold_metrics:
        print(f"  Fold {metrics['fold']}: Best Validation Loss = {metrics['best_val_loss']:.4f}," f"Best IoU = {metrics['best_iou']:.4f}")

    # --- Global IoU vs Hour plot ---
    plt.figure(figsize=(8,5))
    plt.scatter(global_hours, global_ious, color='green')
    plt.title('Global IoU vs Hour (One point per image)')
    plt.xlabel('Hour')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.savefig(f'{time_folder}/{extract_encoder_name(PARAMETERS["encoder"])}_global_iou_vs_hour.png')
    plt.close()

    print(f"Saved global IoU vs Hour plot to {time_folder + '/global_iou_vs_hour.png'}")
if __name__ == '__main__':
    current_time = datetime.now().strftime("%m-%d/%H-%M-%S")  # Month-Day Hour-Minute
    time_folder = "exports/unet/"+current_time
    os.makedirs(time_folder, exist_ok=True)

    run_cross_validation(num_epochs=PARAMETERS['EPOCHS'], batch_size=PARAMETERS['batch_size'], n_splits=PARAMETERS['outer_n_splits'], patience=10, figure_prefix=time_folder+"/")
