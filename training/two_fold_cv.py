import os
import sys
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(project_root)
import csv
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from itertools import product
from sklearn.model_selection import KFold

from dataset import NKCellDataset, JointTransform
from models.unet import create_unet
from segmentation_models_pytorch.losses import DiceLoss

# Setup device (for M1/M2 chips with MPS support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Running with:", device)

# Loss functions
dice_loss = DiceLoss(mode='binary', from_logits=True)
bce_loss = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

def get_loss_function(loss_type):
    if loss_type == "dice":
        return lambda pred, target: dice_loss(pred, target)
    elif loss_type == "bce":
        return lambda pred, target: bce_loss(pred, target)
    else:
        return combined_loss

def compute_iou(pred, target):
    pred = (pred > 0).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / union).item() if union > 0 else 0.0

def train_and_evaluate(train_indices, val_indices, batch_size, encoder_name, freeze_encoder, optimizer_type, loss_type, fold, lr, num_epochs=50, patience=20, figure_prefix=""):
    train_transforms = T.Compose([
        T.RandomCrop((256, 256)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
    ])
    val_transforms = T.Compose([
        T.CenterCrop((256, 256)),
        T.ToTensor(),
    ])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    global dataset_dir
    full_train = NKCellDataset(dataset_dir=dataset_dir, transform=JointTransform(train_transforms))
    full_val = NKCellDataset(dataset_dir=dataset_dir, transform=JointTransform(val_transforms))

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = create_unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        freeze_encoder=freeze_encoder
    ).to(device)

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    loss_fn = get_loss_function(loss_type)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm.tqdm(range(num_epochs), desc=f"Fold {fold+1} Training"):
        model.train()
        train_loss = 0.0
        for images, masks, label_flags in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            labeled_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
            if len(labeled_idx) > 0:
                loss = loss_fn(outputs[labeled_idx], masks[labeled_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(labeled_idx)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks, label_flags in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                labeled_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                if len(labeled_idx) > 0:
                    loss = loss_fn(outputs[labeled_idx], masks[labeled_idx])
                    val_loss += loss.item() * len(labeled_idx)
                    val_iou += compute_iou(outputs[labeled_idx], masks[labeled_idx]) * len(labeled_idx)

        avg_val_loss = val_loss / len(val_indices)
        avg_val_iou = val_iou / len(val_indices)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        elif avg_val_loss <= best_val_loss * 1.01:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return avg_val_loss, avg_val_iou

def outer_cross_validation():
    global dataset_dir
    batch_sizes = [8,16]
    loss_types = ['dice','bce','combined']
    encoders = ['resnet50']
    freeze_opts = [True, False]
    optimizers = ['adam']
    learning_rates = [5e-4,1e-3, 5e-3]
    outer_n_splits = 3

    results = []
    dataset = NKCellDataset(dataset_dir=dataset_dir, transform=None)
    labeled_indices = [
        i for i, mask_path in enumerate(dataset.mask_paths)
        if mask_path is not None and os.path.exists(mask_path)
    ]
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=42)

    for combo in product(batch_sizes, loss_types, encoders, freeze_opts, optimizers, learning_rates):
        batch_size, loss_type, encoder, freeze_encoder, opt_type, lr = combo
        print(f"\n Config: BS={batch_size}, Loss={loss_type}, Encoder={encoder}, Freeze={freeze_encoder}, Optimizer={opt_type}, LR={lr}")
        val_losses = []
        val_ious = []

        for fold, (train_idx, val_idx) in enumerate(outer_kf.split(labeled_indices)):
            train_indices = np.array(labeled_indices)[train_idx].tolist()
            val_indices = np.array(labeled_indices)[val_idx].tolist()

            val_loss, val_iou = train_and_evaluate(
                train_indices, val_indices,
                batch_size, encoder, freeze_encoder,
                opt_type, loss_type, fold, lr,
                figure_prefix=f"results_{fold}_{batch_size}_{loss_type}_{encoder}/"
            )
            val_losses.append(val_loss)
            val_ious.append(val_iou)

        mean_loss = np.mean(val_losses)
        mean_iou = np.mean(val_ious)
        print(f"Mean Val Loss: {mean_loss:.4f}, Mean Val IoU: {mean_iou:.4f}")

        results.append({
            "batch_size": batch_size,
            "loss_type": loss_type,
            "encoder": encoder,
            "freeze_encoder": freeze_encoder,
            "optimizer": opt_type,
            "lr": lr,
            "mean_val_loss": mean_loss,
            "mean_val_iou": mean_iou
        })

    # Save results to CSV
    with open("results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Plot top 10 configurations
    top_results = sorted(results, key=lambda r: r["mean_val_loss"])[:10]
    labels = [f'{r["batch_size"]}-{r["loss_type"]}-{r["encoder"]}-{"T" if r["freeze_encoder"] else "F"}-{r["optimizer"]}-LR{r["lr"]}' for r in top_results]
    losses = [r["mean_val_loss"] for r in top_results]

    plt.figure(figsize=(12, 6))
    plt.barh(labels, losses, color="skyblue")
    plt.xlabel("Mean Validation Loss")
    plt.title("Top 10 Hyperparameter Configurations")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("top_10_configs.png")
    plt.close()

if __name__ == "__main__":
    global dataset_dir
    dataset_dir = "/work3/s234843/02466-Project/dataset"
    if not os.path.exists(dataset_dir):
        dataset_dir = "dataset"

    outer_cross_validation()
