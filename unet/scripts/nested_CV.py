import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from multiprocessing import Pool, set_start_method
from multiprocessing import Manager, Process
from functools import partial
from tqdm import tqdm
import queue as queue_module  # to avoid name clash
import csv
import torch
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
from dataset import NKCellDataset, JointTransform, extract_hour_from_filename
from models.unet import create_unet
from models.unet import create_unet_with_dino_resnet18
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss

import argparse

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
# Loss functions
dice_loss = DiceLoss(mode='binary', from_logits=True)
bce_loss = nn.BCEWithLogitsLoss()
iou_loss = JaccardLoss(mode='binary', from_logits=True)

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

def combined_loss(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

def get_loss_function(loss_type:str):
    if loss_type.lower() == "dice":
        return lambda pred, target: dice_loss(pred, target)
    elif loss_type.lower() == "bce":
        return lambda pred, target: bce_loss(pred, target)
    elif loss_type.lower() == 'iou':
        return lambda pred, target: iou_loss(pred, target)
    else:
        return combined_loss

def compute_iou(preds, targets):
    preds = (torch.sigmoid(preds) > 0.5).float()
    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = intersection / (union + 1e-8)
    return iou

def extract_encoder_name(encoder):
    if not isinstance(encoder, str):
        return str(encoder)
    if "dino" in encoder and ".ckpt" in encoder:
        match = re.search(r'dino/([^/]+)/dino-\1-ep=(\d+)\.ckpt', encoder)
        if match:
            return f"dino_{match.group(1)}_e{match.group(2)}"
    return encoder  # Return original if not a dino path or no match

def train_and_evaluate(train_indices, val_indices, batch_size, encoder_name, freeze_encoder, optimizer_type, loss_type, fold, lr, resize, dataset_dir: str, seed, num_epochs=100, patience=10, is_outer_test=False, print_queue=None, export_dir=""):
    """
    Train and evaluate the model on the given indices.
    """
    common_geometric = T.Compose([
    T.Resize(size=(resize,resize)),
    #T.CenterCrop(size=(resize,resize)),
    #T.RandomCrop(size=(resize, resize)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    ])

    image_only = T.Compose([
    common_geometric,
    #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.ToTensor(),
    ])

    mask_only = T.Compose([
    common_geometric,
    T.ToTensor(),
    ])
    train_transforms = JointTransform(image_transform=image_only, mask_transform=mask_only, seed=seed)

    val_transform = T.Compose([
        #T.CenterCrop((resize, resize)),
        T.Resize((resize,resize)),
        T.ToTensor(),
    ])
    val_transforms = JointTransform(image_transform=val_transform, mask_transform=val_transform, seed=seed)

    full_train = NKCellDataset(dataset_dir=dataset_dir, transform=train_transforms)
    full_val = NKCellDataset(dataset_dir=dataset_dir, transform=val_transforms)

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

        # --- Per-worker reproducibility for DataLoader ---
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
    best_val_iou = 0
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
        iou_scores = []
        fold_iou_list = []
        with torch.no_grad():
            for images, masks, label_flags, img_paths in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                labeled_idx = (label_flags == 1).nonzero(as_tuple=True)[0]
                if len(labeled_idx) > 0:
                    labeled_outputs = outputs[labeled_idx]
                    labeled_masks = masks[labeled_idx]
                    loss = loss_fn(labeled_outputs, labeled_masks)
                    val_loss += loss.item() * len(labeled_idx)
                    ious = compute_iou(labeled_outputs, labeled_masks)
                    ious = ious.cpu().numpy()
                    iou_scores.extend(ious)

                    if is_outer_test:
                        # For each labeled sample, map IOU to its hour:
                        for idx_in_batch, sample_iou in zip(labeled_idx.cpu().numpy(), ious):
                            img_path = img_paths[idx_in_batch]
                            hour = extract_hour_from_filename(img_path)
                            if hour is not None:
                                global_hours.append(hour)
                                global_ious.append(sample_iou)
                            
                            img_id = os.path.basename(img_path)  # Unique identifier
                            fold_iou_list.append((img_id, sample_iou))

        # Save per-image IoU with IDs for this fold
        if is_outer_test:
            np.save(os.path.join(export_dir, f"{extract_encoder_name(encoder_name)}_{batch_size}_{freeze_encoder}_{optimizer_type}_{loss_type}_{lr}_{resize}_seed{seed}_epoch{num_epochs}_pat{patience}_per_image_iou_fold{fold+1}.npy"), 
            np.array(fold_iou_list, dtype=object))

        avg_val_loss = val_loss / len(val_indices)
        avg_val_iou = np.mean(iou_scores)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_indices):.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}", end="\r")

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            patience_counter = 0
            #best_model_path = os.path.join(export_dir, f"best_model_fold{fold}.pth")
            #torch.save(model.state_dict(), best_model_path)
            
        elif avg_val_iou >= best_val_iou * 1.01:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load best checkpoint before final return
    #best_model_path = os.path.join(export_dir, f"best_model_fold{fold}.pth")
    #if os.path.exists(best_model_path):
    #    model.load_state_dict(torch.load(best_model_path))

        # Visual comparison of predicted vs actual masks using best model
        model.eval()
        with torch.no_grad():
            for i, (images, masks, _, paths) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                num_to_save = min(1, images.size(0))  # save 1 example
                for j in range(num_to_save):
                    img = images[j].cpu().permute(1, 2, 0)
                    gt_mask = masks[j].cpu().squeeze()
                    pred_mask = preds[j].cpu().squeeze()

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(img)
                    axes[0].set_title("Input Image")
                    axes[1].imshow(gt_mask, cmap='gray')
                    axes[1].set_title("Ground Truth Mask")
                    axes[2].imshow(pred_mask, cmap='gray')
                    axes[2].set_title("Predicted Mask")
                    for ax in axes:
                        ax.axis("off")
                    plt.tight_layout()

                    img_id = os.path.basename(paths[j]) if paths[j] else f"sample{j}"
                    fig_path = os.path.join(export_dir, f"fold{fold}_pred_vs_gt_{img_id}.png")
                    plt.savefig(fig_path)
                    plt.close()
                break  # Only save the first batch

    if is_outer_test:
        if print_queue:
            print_queue.put(f"✅ Outer test fold completed: Test Loss = {avg_val_loss:.4f}, Test IoU = {avg_val_iou:.4f}")
        else:
            print(f"✅ Outer test fold completed: Test Loss = {avg_val_loss:.4f}, Test IoU = {avg_val_iou:.4f}")
    else:
        if print_queue:
            print_queue.put(f"🔁 Inner CV - Completed config: Val Loss = {avg_val_loss:.4f}, Val IoU = {avg_val_iou:.4f}.     ")
        else:
            print(f"🔁 Inner CV - Completed config: Val Loss = {avg_val_loss:.4f}, Val IoU = {avg_val_iou:.4f}       ")
    return avg_val_loss, avg_val_iou

def run_single_inner_fold(args, print_queue=None, export_dir=""):
    """
    Run a single inner fold with the given hyperparameters.
    This function is designed to be run in parallel.
    """
    # Now expects dataset as final argument
    batch_size, loss_type, encoder, freeze_encoder, opt_type, lr, resize, fold, train_idx, val_idx, dataset_dir, seed, num_epochs, patience, is_outer_test = args
    val_loss, val_iou = train_and_evaluate(
        train_indices=train_idx, val_indices=val_idx,
        batch_size=batch_size, encoder_name=encoder, freeze_encoder=freeze_encoder,
        optimizer_type=opt_type, loss_type=loss_type, fold=fold, lr=lr, resize=resize, dataset_dir=dataset_dir, seed=seed, num_epochs=num_epochs, patience=patience,
        is_outer_test=is_outer_test, print_queue=print_queue, export_dir=export_dir
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

def run_inner_cv(train_indices, batch_sizes, loss_types, encoders, freeze_opts, optimizers, learning_rates, resize, dataset_dir: str, seed, num_epochs: int, patience: int, inner_folds=3, print_queue=None, export_dir=""):
    """
    Run inner cross-validation to find the best hyperparameters on the training set.
    """
    inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    all_configs = list(product(batch_sizes, loss_types, encoders, freeze_opts, optimizers, learning_rates, resize))
    # Collect all tasks for parallel execution
    tasks = []
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(train_indices)):
        inner_train = np.array(train_indices)[inner_train_idx].tolist()
        inner_val = np.array(train_indices)[inner_val_idx].tolist()
        
        with open(os.path.join(export_dir, "runtime.log"), "a") as f:
            f.write(f"    I{inner_fold+1} - Train: {len(inner_train)}, Val: {len(inner_val)}\n")
            f.write(f"    I{inner_fold+1} - Train indices: {inner_train}\n")
            f.write(f"    I{inner_fold+1} - Val indices: {inner_val}\n")

        for combo in all_configs:
            task = (*combo, 0, inner_train, inner_val, dataset_dir, seed, num_epochs, patience, False)  # is_outer_test=False for inner
            tasks.append(task)

    # Run in parallel with tqdm progress bar
    with Pool(processes=PARAMETERS["cpu_workers"]) as pool:
        wrapped_fn = partial(run_single_inner_fold, print_queue=print_queue, export_dir=export_dir)
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
        config_scores[config].append(result["val_iou"])
        #config_scores[config].append(result["val_loss"])


    # Compute mean validation IoU across folds
    avg_scores = {config: np.mean(losses) for config, losses in config_scores.items()}
    if not avg_scores:
        raise ValueError("No valid configurations found during inner CV. Check for NaN IoU values or training issues.")
    #best_config = min(avg_scores, key=avg_scores.get)
    best_config = max(avg_scores, key=avg_scores.get)
    return best_config, avg_scores[best_config] # best config, IoU

def outer_cross_validation(dataset_dir: str, seed, export_dir="", patience: int = 20):
    """
    Run outer cross-validation to evaluate the model on the test set.
    """
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

    dataset = NKCellDataset(dataset_dir=dataset_dir, transform=None)
    labeled_indices = [
        i for i, mask_path in enumerate(dataset.mask_paths)
        if mask_path is not None and os.path.exists(mask_path)
    ]
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=seed)

    with open(os.path.join(export_dir, "runtime.log"), "a") as f:
        f.write(f"Labeled indices length: {len(labeled_indices)}\n")
        f.write(f"Labeled indices: {labeled_indices}\n")

    all_results = []

    # Setup print queue and worker
    manager = Manager()
    print_queue = manager.Queue()
    printer = Process(target=print_worker, args=(print_queue,))
    printer.start()

    global global_hours
    global global_ious
    global_hours = []
    global_ious = []

    for fold, (train_idx, test_idx) in tqdm(enumerate(outer_kf.split(labeled_indices)), desc="Outer CV folds", total=outer_n_splits):
        train_indices = np.array(labeled_indices)[train_idx].tolist()
        test_indices = np.array(labeled_indices)[test_idx].tolist()

        with open(os.path.join(export_dir, "runtime.log"), "a") as f:
            f.write(f"O{fold+1} - Train: {len(train_indices)}, Test: {len(test_indices)}\n")
            f.write(f"O{fold+1} - Train indices: {train_indices}\n")
            f.write(f"O{fold+1} - Test indices: {test_indices}\n")

        # Run inner CV on training data to select best hyperparams
        best_config, best_iou = run_inner_cv(
            train_indices=train_indices,
            batch_sizes=batch_sizes,
            loss_types=loss_types,
            encoders=encoders,
            freeze_opts=freeze_opts,
            optimizers=optimizers,
            learning_rates=learning_rates,
            resize=resize,
            dataset_dir=dataset_dir,
            seed=seed,
            num_epochs=num_epochs,
            patience=patience,
            inner_folds=inner_folds,
            print_queue=print_queue,
            export_dir=export_dir
        )

        output_names = [
            "batch_size",
            "loss_type",
            "encoder",
            "freeze_encoder",
            "optimizer",
            "lr",
            "resize",
            "test_loss"
        ]

        # Interpret zip as string:
        best_config_str = ", ".join(f"{name}={value}" for name, value in zip(output_names[:-1], best_config))
        if print_queue:
            print_queue.put(f"Best config with iou score {best_iou}: {best_config_str}")
        else:
            print(f"Best config with iou score {best_iou}:", best_config_str)

        # Train final model with best config on full training set, evaluate on test set
        final_args = (*best_config, fold, train_indices, test_indices, dataset_dir, seed, num_epochs, patience, True)  # is_outer_test=True for final outer test
        result = run_single_inner_fold(final_args, print_queue=print_queue, export_dir=export_dir)



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
        print(f"Mean Test Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}, Mean Test IoU: {mean_test_iou:.4f} ± {std_test_iou:.4f}")

    # Save results to CSV
    # print("Writing results.csv")
    try:
        with open(export_dir + "results.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_results[0].keys())
            writer.writeheader()
            writer.writerows(final_results)
    except:
        print("Couldn't write results.csv")
    else:
        print("Wrote results.csv")

    # print("Plotting top 3 hyperparameter configurations")
    top_results = sorted(final_results, key=lambda r: r["mean_test_loss"])[:10]
    labels = [f'{r["batch_size"]}-{r["loss_type"]}-{extract_encoder_name(r["encoder"])}-{"T" if r["freeze_encoder"] else "F"}-{r["optimizer"]}-LR{r["lr"]}' for r in top_results]
    losses = [r["mean_test_loss"] for r in top_results]
    plt.figure(figsize=(12, 6))
    plt.barh(labels, losses, color="skyblue")
    plt.xlabel("Mean Validation Loss")
    plt.title("Top 3 Hyperparameter Configurations")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "top_3_configs.png"))
    plt.close()
    # print("Finished plotting")

    endtime = datetime.now()
    print("Started at:", starttime)
    print("Finished at:", endtime)
    print("Time elapsed:", endtime-starttime)

if __name__ == "__main__":
    set_start_method('spawn', force=True)

    # print("Given arguments:")
    print(args)

    print(f"Using device: {device}")
    

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
    total_combinations = reduce(operator.mul, lengths)*PARAMETERS["outer_n_splits"]*PARAMETERS["inner_n_splits"]+PARAMETERS["outer_n_splits"]
    print(f"Total combinations: {total_combinations}")

    dataset_dir = "dataset" if not os.path.exists(dataset_dir:="/work3/s234843/02466-Project/dataset") else dataset_dir

    SEED = 42
    # random.seed(SEED)
    # np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    outer_cross_validation(export_dir=export_folder, seed=SEED, dataset_dir=dataset_dir)
    #pbar.close()