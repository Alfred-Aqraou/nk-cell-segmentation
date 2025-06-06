import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from multiprocessing import Pool, set_start_method
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
from collections import defaultdict
from datetime import datetime

import argparse


from dataset import NKCellDataset, JointTransform
from models.unet import create_unet
from segmentation_models_pytorch.losses import DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--workers", help = "Amount of CPU workers to run in parallel", type=int)
args = parser.parse_args()
print("Given arguments:")
print(args)

global PARAMETERS
PARAMETERS = {
    "cpu_workers": 8 if not args.workers else args.workers,
    "EPOCHS": 100,
    "batch_sizes": [8, 16, 32],
    "loss_types": ['dice', 'combined'],
    "encoders": ['resnet18','resnet34','resnet50'],
    "freeze_opts": [True, False],
    "optimizers": ['adam', 'sgd'],
    "learning_rates": [1e-4,5e-4,1e-3],
    "outer_n_splits": 3,
    "resize": [(128, 128), (512, 512)]
    }

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

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

def train_and_evaluate(train_indices, val_indices, batch_size, encoder_name, freeze_encoder, optimizer_type, loss_type, fold, lr, resize, dataset_dir, num_epochs=100, patience=10):
    train_transforms = T.Compose([
        T.RandomCrop(resize),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
    ])
    val_transforms = T.Compose([
        T.CenterCrop(resize),
        T.ToTensor(),
    ])
    current_dir = os.path.dirname(os.path.abspath(__file__))
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

def run_single_fold(args):
    batch_size, loss_type, encoder, freeze_encoder, opt_type, lr, resize, fold, train_idx, val_idx, dataset_dir = args
    val_loss, val_iou = train_and_evaluate(
        train_idx, val_idx,
        batch_size, encoder, freeze_encoder,
        opt_type, loss_type, fold, lr, resize, dataset_dir
    )
    return {
        "batch_size": batch_size,
        "loss_type": loss_type,
        "encoder": encoder,
        "freeze_encoder": freeze_encoder,
        "optimizer": opt_type,
        "lr": lr,
        "resize": resize,
        "fold": fold,
        "val_loss": val_loss,
        "val_iou": val_iou
    }

def outer_cross_validation(figure_prefix=""):
    global dataset_dir
    global PARAMETERS
    batch_sizes = PARAMETERS["batch_sizes"]
    loss_types = PARAMETERS["loss_types"]
    encoders = PARAMETERS["encoders"]
    freeze_opts = PARAMETERS["freeze_opts"]
    optimizers = PARAMETERS["optimizers"]
    learning_rates = PARAMETERS["learning_rates"]
    resize = PARAMETERS["resize"]

    outer_n_splits = PARAMETERS["outer_n_splits"]
    num_workers = PARAMETERS["cpu_workers"]

    dataset = NKCellDataset(dataset_dir=dataset_dir, transform=None)
    labeled_indices = [
        i for i, mask_path in enumerate(dataset.mask_paths)
        if mask_path is not None and os.path.exists(mask_path)
    ]
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=42)

    all_tasks = []
    for combo in product(batch_sizes, loss_types, encoders, freeze_opts, optimizers, learning_rates, resize):
        batch_size, loss_type, encoder, freeze_encoder, opt_type, lr, resize = combo
        for fold, (train_idx, val_idx) in enumerate(outer_kf.split(labeled_indices)):
            train_indices = np.array(labeled_indices)[train_idx].tolist()
            val_indices = np.array(labeled_indices)[val_idx].tolist()
            task = (batch_size, loss_type, encoder, freeze_encoder, opt_type, lr, resize, fold, train_indices, val_indices, dataset_dir)
            all_tasks.append(task)

    with Pool(processes=num_workers) as pool:
        all_results = pool.map(run_single_fold, all_tasks)

    # Process results
    combo_results = defaultdict(list)
    for result in all_results:
        combo_key = (result["batch_size"], result["loss_type"], result["encoder"], result["freeze_encoder"], result["optimizer"], result["lr"], result["resize"])
        combo_results[combo_key].append((result["val_loss"], result["val_iou"]))

    final_results = []
    for combo_key, fold_results in combo_results.items():
        val_losses, val_ious = zip(*fold_results)
        std_val_loss = np.std(val_losses)
        std_val_iou = np.std(val_ious)
        mean_val_loss = np.mean(val_losses)
        mean_val_iou = np.mean(val_ious)
        final_results.append({
            "batch_size": combo_key[0],
            "loss_type": combo_key[1],
            "encoder": combo_key[2],
            "freeze_encoder": combo_key[3],
            "optimizer": combo_key[4],
            "lr": combo_key[5],
            "resize": combo_key[6],
            "mean_val_loss": mean_val_loss,
            "std_val_loss": std_val_loss,
            "mean_val_iou": mean_val_iou,
            "std_val_iou": std_val_iou
        })
        print(f"Config: BS={combo_key[0]}, Loss={combo_key[1]}, Encoder={combo_key[2]}, Freeze={combo_key[3]}, Opt={combo_key[4]}, LR={combo_key[5]}")
        print(f"Mean Val Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}, Mean Val IoU: {mean_val_iou:.4f} ± {std_val_iou:.4f}")

    # Save results to CSV
    print("Writing results.csv")
    try:
        with open(figure_prefix + "results.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_results[0].keys())
            writer.writeheader()
            writer.writerows(final_results)
    except:
        print("Couldn't write results.csv")
    else:
        print("Finished writing results.csv")

    # Plot top 10 configurations
    print("Plotting top 10 configs")
    top_results = sorted(final_results, key=lambda r: r["mean_val_loss"])[:10]
    labels = [f'{r["batch_size"]}-{r["loss_type"]}-{r["encoder"]}-{"T" if r["freeze_encoder"] else "F"}-{r["optimizer"]}-LR{r["lr"]}' for r in top_results]
    losses = [r["mean_val_loss"] for r in top_results]
    plt.figure(figsize=(12, 6))
    plt.barh(labels, losses, color="skyblue")
    plt.xlabel("Mean Validation Loss")
    plt.title("Top 10 Hyperparameter Configurations")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(figure_prefix + "top_10_configs.png")
    plt.close()
    print("Finished plotting")

    global starttime
    endtime = datetime.now().strftime("%m-%d/%H-%M-%S")
    print("Started at:", starttime)
    print("Finished at:", endtime)
    print("Time elapsed:", endtime-current_time)

if __name__ == "__main__":
    set_start_method('spawn', force=True)

    global dataset_dir
    dataset_dir = "dataset" if not os.path.exists(dataset_dir:="/work3/s234843/02466-Project/dataset") else dataset_dir

    global starttime
    starttime = datetime.now()
    current_time = starttime.strftime("%m-%d/%H-%M-%S")  # Month-Day Hour-Minute
    export_folder = "exports/unet/"+current_time
    os.makedirs(export_folder, exist_ok=True)

    outer_cross_validation(figure_prefix=export_folder+"/")
