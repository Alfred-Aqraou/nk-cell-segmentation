import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from multiprocessing import Pool, set_start_method
from multiprocessing import Manager, Process
import queue as queue_module  # to avoid name clash
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

from functools import reduce
import operator

import re
from typing import List
# import pandas as pd


# import wandb

from dataset import NKCellDataset, JointTransform
from models.unet import create_unet
from models.unet import create_unet_with_dino_resnet18
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss

import argparse
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Print worker for safe multiprocessing printing
def print_worker(print_queue):
    while True:
        try:
            msg = print_queue.get(timeout=10)
            if msg == "STOP":
                break
            print(msg)
        except queue_module.Empty:
            continue

parser = argparse.ArgumentParser()
parser.add_argument("--workers", help = "Amount of CPU workers to run in parallel", type=int, default=8)
parser.add_argument("--epochs", help = 'Specify epochs. I.e: 100', required=True, type=int)
parser.add_argument("--encoders", help = "Specify encoders, you can also add a ckpt path", nargs='+', required=True, type=str) # ['resnet18', 'resnet34', 'resnet50']
parser.add_argument("--loss", help = "Specify loss types", nargs='+', required=True) # ['dice', 'combined']
parser.add_argument("--batchsizes", help = 'Specify batch sizes. I.e: 8 16', nargs='+', required=True, type=int)
parser.add_argument("--freeze", help = 'Specify whether to use freeze_opts. I.e: True False', nargs='+', required=True, type=str)
parser.add_argument("--optimizers", help = 'Specify optimizer. I.e: "adam"', nargs='+', required=True, type=str)
parser.add_argument("--lr", help = 'Specify learning rate. I.e: 0.0005 0.0001', nargs='+', required=True, type=float)
parser.add_argument("--resize", help = 'Specify dimensions to resize to. I.e: 256 512', nargs='+', required=True, type=int)
parser.add_argument("--outer", help = 'Specify outer n splits. I.e: 5', required=True, type=int)
parser.add_argument("--inner", help = 'Specify inner n splits. I.e: 5', required=True, type=int)

args = parser.parse_args()

args.freeze = [str(val).strip().lower() in ("true", "1", "yes", "y", "t") for val in args.freeze]

# args.resize = tuple(args.resize)
# for i in args.resize:
#     assert isinstance(i, int)
# assert len(args.resize) == 2

global PARAMETERS
PARAMETERS = {
    "cpu_workers": args.workers,
    "EPOCHS": args.epochs,
    "batch_sizes": [*args.batchsizes],
    "loss_types": [*args.loss],
    "encoders": [*args.encoders],
    "freeze_opts": [*args.freeze],
    "optimizers": [*args.optimizers],
    "learning_rates": [*args.lr],
    "resize": [*args.resize],
    "outer_n_splits": args.outer,
    "inner_n_splits": args.inner
    }


# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Loss functions
dice_loss = DiceLoss(mode='binary', from_logits=True)
bce_loss = nn.BCEWithLogitsLoss()
iou_loss = JaccardLoss(mode='binary', from_logits=True)

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

def compute_iou(pred, target):
    pred = (pred > 0).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / union).item() if union > 0 else 0.0

def extract_encoder_name(encoder):
    if not isinstance(encoder, str):
        return str(encoder)
    if "dino" in encoder and ".ckpt" in encoder:
        match = re.search(r'dino/([^/]+)/dino-\1-ep=(\d+)\.ckpt', encoder)
        if match:
            return f"dino_{match.group(1)}_e{match.group(2)}"
    return encoder  # Return original if not a dino path or no match

def train_and_evaluate(train_indices, val_indices, batch_size, encoder_name, freeze_encoder, optimizer_type, loss_type, fold, lr, resize, dataset_dir: str, num_epochs=100, patience=10, is_outer_test=False, print_queue=None):
    common_geometric = T.Compose([
    T.RandomCrop(size=(resize, resize)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    ])

    image_only = T.Compose([
    common_geometric,
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.ToTensor(),
    ])

    mask_only = T.Compose([
    common_geometric,
    T.ToTensor(),
    ])
    train_transforms = JointTransform(image_transform=image_only, mask_transform=mask_only)

    val_transform = T.Compose([
        T.CenterCrop((resize, resize)),
        T.ToTensor(),
    ])
    val_transforms = JointTransform(image_transform=val_transform, mask_transform=val_transform)

    full_train = NKCellDataset(dataset_dir=dataset_dir, transform=train_transforms)
    full_val = NKCellDataset(dataset_dir=dataset_dir, transform=val_transforms)

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    def seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                          worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        worker_init_fn=seed_worker, generator=g)

    if ".ckpt" in encoder_name:
        model = create_unet_with_dino_resnet18(
            ckpt_path=os.path.abspath(encoder_name),
            in_channels=3,
            classes=1,
            freeze_encoder=freeze_encoder
        ).to(device)
    else:
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

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks, label_flags, _ in train_loader:
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
            for images, masks, label_flags, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                labeled_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                if len(labeled_idx) > 0:
                    loss = loss_fn(outputs[labeled_idx], masks[labeled_idx])
                    val_loss += loss.item() * len(labeled_idx)
                    val_iou += compute_iou(outputs[labeled_idx], masks[labeled_idx]) * len(labeled_idx)

        avg_val_loss = val_loss / len(val_indices)
        avg_val_iou = val_iou / len(val_indices)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_indices):.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}", end="\r")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        elif avg_val_loss <= best_val_loss * 1.01:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # pbar.update(1)
    if is_outer_test:
        if print_queue:
            print_queue.put(f"✅ Outer test fold completed: Test Loss = {avg_val_loss:.4f}, Test IoU = {avg_val_iou:.4f}")
        else:
            print(f"✅ Outer test fold completed: Test Loss = {avg_val_loss:.4f}, Test IoU = {avg_val_iou:.4f}")
    else:
        if print_queue:
            print_queue.put(f"🔁 Inner CV - Completed config: Val Loss = {avg_val_loss:.4f}, Val IoU = {avg_val_iou:.4f}")
        else:
            print(f"🔁 Inner CV - Completed config: Val Loss = {avg_val_loss:.4f}, Val IoU = {avg_val_iou:.4f}")
    return avg_val_loss, avg_val_iou

def run_single_fold(args, print_queue=None):
    # Now expects dataset as final argument
    batch_size, loss_type, encoder, freeze_encoder, opt_type, lr, resize, fold, train_idx, val_idx, dataset_dir, num_epochs, patience, is_outer_test = args
    val_loss, val_iou = train_and_evaluate(
        train_indices=train_idx, val_indices=val_idx,
        batch_size=batch_size, encoder_name=encoder, freeze_encoder=freeze_encoder,
        optimizer_type=opt_type, loss_type=loss_type, fold=fold, lr=lr, resize=resize, dataset_dir=dataset_dir, num_epochs=num_epochs, patience=patience,
        is_outer_test=is_outer_test, print_queue=print_queue
    )
    result = {
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
    return result

def run_inner_cv(train_indices, batch_sizes, loss_types, encoders, freeze_opts, optimizers, learning_rates, resize, dataset_dir: str, num_epochs: int, patience: int, inner_folds=3, print_queue=None):
    inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=SEED)
    all_configs = list(product(batch_sizes, loss_types, encoders, freeze_opts, optimizers, learning_rates, resize))
    # Collect all tasks for parallel execution
    tasks = []
    for inner_train_idx, inner_val_idx in inner_kf.split(train_indices):
        inner_train = np.array(train_indices)[inner_train_idx].tolist()
        inner_val = np.array(train_indices)[inner_val_idx].tolist()
        for combo in all_configs:
            task = (*combo, 0, inner_train, inner_val, dataset_dir, num_epochs, patience, False)  # is_outer_test=False for inner
            tasks.append(task)

    # Run in parallel with tqdm progress bar
    from functools import partial
    from tqdm import tqdm
    with Pool(processes=PARAMETERS["cpu_workers"]) as pool:
        wrapped_fn = partial(run_single_fold, print_queue=print_queue)
        with tqdm(total=len(tasks), desc="Inner CV progress", position=1, leave=False) as inner_pbar:
            results = []
            for r in pool.imap_unordered(wrapped_fn, tasks):
                results.append(r)
                inner_pbar.update(1)

    # Aggregate results
    config_scores = defaultdict(list)
    for result in results:
        config = (
            result["batch_size"],
            result["loss_type"],
            result["encoder"],
            result["freeze_encoder"],
            result["optimizer"],
            result["lr"],
            result["resize"]
        )
        config_scores[config].append(result["val_loss"])

    # Compute mean validation loss across folds
    avg_scores = {config: np.mean(losses) for config, losses in config_scores.items()}
    best_config = min(avg_scores, key=avg_scores.get)
    return best_config, avg_scores[best_config] # best config, loss

def outer_cross_validation(figure_prefix="", dataset_dir: str = "dataset", patience: int = 20):
    print("Started at:", starttime)

    # Hyperparameters
    num_epochs = PARAMETERS["EPOCHS"]
    batch_sizes = PARAMETERS["batch_sizes"]
    loss_types = PARAMETERS["loss_types"]
    encoders = PARAMETERS["encoders"]
    freeze_opts = PARAMETERS["freeze_opts"]
    optimizers = PARAMETERS["optimizers"]
    learning_rates = PARAMETERS["learning_rates"]
    resize = PARAMETERS["resize"]
    inner_folds = PARAMETERS["inner_n_splits"]

    outer_n_splits = PARAMETERS["outer_n_splits"]
    num_workers = PARAMETERS["cpu_workers"]

    # Instantiate dataset ONCE and pass to all functions
    dataset = NKCellDataset(dataset_dir=dataset_dir, transform=None)
    labeled_indices = [
        i for i, mask_path in enumerate(dataset.mask_paths)
        if mask_path is not None and os.path.exists(mask_path)
    ]
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=SEED)

    all_results = []

    # Setup print queue and worker
    manager = Manager()
    print_queue = manager.Queue()
    printer = Process(target=print_worker, args=(print_queue,))
    printer.start()

    for fold, (train_idx, val_idx) in tqdm.tqdm(enumerate(outer_kf.split(labeled_indices)), desc="Outer CV folds", total=outer_n_splits):
        train_indices = np.array(labeled_indices)[train_idx].tolist()
        val_indices = np.array(labeled_indices)[val_idx].tolist()

        # Run inner CV on training data to select best hyperparams
        best_config, best_loss = run_inner_cv(
            train_indices=train_indices,
            batch_sizes=batch_sizes,
            loss_types=loss_types,
            encoders=encoders,
            freeze_opts=freeze_opts,
            optimizers=optimizers,
            learning_rates=learning_rates,
            resize=resize,
            dataset_dir=dataset_dir,
            num_epochs=num_epochs,
            patience=patience,
            inner_folds=inner_folds,
            print_queue=print_queue
        )

        output_names = [
            "batch_size",
            "loss_type",
            "encoder",
            "freeze_encoder",
            "optimizer",
            "lr",
            "resize",
            "val_loss"
        ]

        # Interpret zip as string:
        best_config_str = ", ".join(f"{name}={value}" for name, value in zip(output_names[:-1], best_config))
        if print_queue:
            print_queue.put(f"Best config with val loss {best_loss}: {best_config_str}")
        else:
            print(f"Best config with val loss {best_loss}:", best_config_str)

        # Train final model with best config on full training set, evaluate on val set
        final_args = (*best_config, fold, train_indices, val_indices, dataset_dir, num_epochs, patience, True)  # is_outer_test=True for final outer test
        result = run_single_fold(final_args, print_queue=print_queue)

        all_results.append(result)

    # Stop the print worker
    print_queue.put("STOP")
    printer.join()

    # Process results
    combo_results = defaultdict(list)
    for result in all_results:
        combo_key = (result["batch_size"], result["loss_type"], result["encoder"], result["freeze_encoder"], result["optimizer"], result["lr"], result["resize"])
        combo_results[combo_key].append((result["val_loss"], result["val_iou"]))

    final_results = []
    for combo_key, fold_results in combo_results.items():
        val_losses, val_ious = zip(*fold_results)
        std_test_loss = np.std(val_losses)
        std_test_iou = np.std(val_ious)
        mean_test_loss = np.mean(val_losses)
        mean_test_iou = np.mean(val_ious)
        final_results.append({
            "batch_size": combo_key[0],
            "loss_type": combo_key[1],
            "encoder": combo_key[2],
            "freeze_encoder": combo_key[3],
            "optimizer": combo_key[4],
            "lr": combo_key[5],
            "resize": combo_key[6],
            "mean_test_loss": mean_test_loss,
            "std_test_loss": std_test_loss,
            "mean_test_iou": mean_test_iou,
            "std_test_iou": std_test_iou
        })
        print(f"Config: BS={combo_key[0]}, Loss={combo_key[1]}, Encoder={combo_key[2]}, Freeze={combo_key[3]}, Opt={combo_key[4]}, LR={combo_key[5]}")
        print(f"Mean Val Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}, Mean Val IoU: {mean_test_iou:.4f} ± {std_test_iou:.4f}")

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
    top_results = sorted(final_results, key=lambda r: r["mean_test_loss"])[:10]
    labels = [f'{r["batch_size"]}-{r["loss_type"]}-{extract_encoder_name(r["encoder"])}-{"T" if r["freeze_encoder"] else "F"}-{r["optimizer"]}-LR{r["lr"]}' for r in top_results]
    losses = [r["mean_test_loss"] for r in top_results]
    plt.figure(figsize=(12, 6))
    plt.barh(labels, losses, color="skyblue")
    plt.xlabel("Mean Validation Loss")
    plt.title("Top 10 Hyperparameter Configurations")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(figure_prefix + "top_10_configs.png")
    plt.close()
    print("Finished plotting")

    endtime = datetime.now()
    print("Started at:", starttime)
    print("Finished at:", endtime)
    print("Time elapsed:", endtime-starttime)

if __name__ == "__main__":
    set_start_method('spawn', force=True)

    print("Given arguments:")
    print(args)
    

    global starttime
    starttime = datetime.now()
    current_time = starttime.strftime("%m-%d/%H-%M-%S")  # Month-Day Hour-Minute
    export_folder = "exports/unet/"+current_time
    os.makedirs(export_folder, exist_ok=True)

    keys_to_multiply = [
    "batch_sizes",
    "loss_types",
    "encoders",
    "freeze_opts",
    "optimizers",
    "learning_rates",
    "resize"
]

    # Compute total combinations
    lengths = [len(PARAMETERS[k]) for k in keys_to_multiply]
    total_combinations = reduce(operator.mul, lengths)*PARAMETERS["outer_n_splits"]*PARAMETERS["inner_n_splits"]
    #global pbar
    #pbar = tqdm.tqdm(total=total_combinations, leave=True, desc="Combinations finished")

    outer_cross_validation(figure_prefix=export_folder+"/")
    #pbar.close()