# training/train.py

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from dataset import NKCellDataset
from models.unet import create_unet
from segmentation_models_pytorch.losses import DiceLoss

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transforms for training and validation
    train_transform = T.Compose([
        T.CenterCrop((960, 960)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.CenterCrop((960, 960)),
        T.ToTensor(),
    ])

    dataset_dir = "dataset"

    # Create two dataset instances:
    
    full_dataset_train = NKCellDataset(dataset_dir=dataset_dir, transform=train_transform)
    full_dataset_val   = NKCellDataset(dataset_dir=dataset_dir, transform=val_transform)
    
    # Filter labeled samples (where mask_path is not None)
    # (Assumes that the order of file names is identical between the two dataset instances.)
    labeled_indices = [i for i, mask_path in enumerate(full_dataset_train.mask_paths) if mask_path is not None]
    
    # Shuffle and split the labeled indices into training and validation sets
    np.random.seed(42)
    labeled_indices = np.array(labeled_indices)
    np.random.shuffle(labeled_indices)
    total_count = len(labeled_indices)
    val_count = int(0.2 * total_count)
    train_indices = labeled_indices[:-val_count]
    val_indices   = labeled_indices[-val_count:]
    
    # Create Subset objects using the respective dataset instances
    train_dataset = Subset(full_dataset_train, train_indices.tolist())
    val_dataset   = Subset(full_dataset_val, val_indices.tolist())
    
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create the U-Net model (with encoder freezing as configured)
    model = create_unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1, 
        freeze_encoder=True
    )
    model.to(device)
    
    # Set up loss function and optimizer
    criterion = DiceLoss(mode='binary', from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 100  # number of training epochs
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists for storing losses for plotting
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        train_loss = 0.0

        for images, masks, label_flags in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Compute loss only for labeled samples in this batch
            labeled_indices_batch = (label_flags == 1).nonzero(as_tuple=True)[0]
            if len(labeled_indices_batch) > 0:
                labeled_outputs = outputs[labeled_indices_batch]
                labeled_masks   = masks[labeled_indices_batch]
                loss = criterion(labeled_outputs, labeled_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(labeled_indices_batch)
        
        avg_train_loss = train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, label_flags in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                labeled_indices_batch = (label_flags == 1).nonzero(as_tuple=True)[0]
                if len(labeled_indices_batch) > 0:
                    labeled_outputs = outputs[labeled_indices_batch]
                    labeled_masks   = masks[labeled_indices_batch]
                    loss = criterion(labeled_outputs, labeled_masks)
                    val_loss += loss.item() * len(labeled_indices_batch)
        
        avg_val_loss = val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)

        tolerance = 0.015
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("Validation loss improved - model updated.")
        elif avg_val_loss < best_val_loss + tolerance:
            patience_counter = 0
            print("No significant improvement on validation loss, but within tolerance - patience reset.")
        else:
            patience_counter += 1
            print(f"No improvement on validation loss. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Plot the training and validation loss curves
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curve_{len(epochs_range)}epochs.png')
    plt.show()

    # Evaluate model performance on the validation set (mean Dice score)
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks, label_flags in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            labeled_indices_batch = (label_flags == 1).nonzero(as_tuple=True)[0]
            if len(labeled_indices_batch) == 0:
                continue

            labeled_preds = preds[labeled_indices_batch]
            labeled_masks = masks[labeled_indices_batch]

            intersection = (labeled_preds * labeled_masks).sum(dim=(1,2,3))
            union = labeled_preds.sum(dim=(1,2,3)) + labeled_masks.sum(dim=(1,2,3))
            dice = 2.0 * intersection / (union + 1e-8)
            dice_scores.extend(dice.cpu().numpy())

    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice Score on Validation Set: {mean_dice:.4f}")


if __name__ == '__main__':
    main()