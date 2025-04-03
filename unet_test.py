import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from dataset import NKCellDataset

import torch
import torch.nn as nn
import torch.optim as optim

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import numpy as np
import matplotlib.pyplot as plt
import tqdm

if __name__ == '__main__':
    # Example transform
    train_transform = T.Compose([
        T.CenterCrop((960, 960)),
        T.RandomHorizontalFlip(p=0.5),       # Flip horizontally with 50% probability
        T.RandomVerticalFlip(p=0.5),         # Flip vertically with 50% probability
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

    # Define model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    # Load custom weights
    # custom_weights_path = "weights.pt"
    # custom_weights = torch.load(custom_weights_path, map_location="cpu")  # Load weights to CPU first

    # # Option 1: Load weights into the encoder only (if weights.pt contains encoder weights)
    # try:
    #     model.encoder.load_state_dict(custom_weights)
    #     print("Loaded custom weights into the encoder successfully.")
    # except RuntimeError as e:
    #     print(f"Error loading weights into encoder: {e}")
    #     # Option 2: Load weights into the full model (if weights.pt contains the entire UNet)
    #     try:
    #         model.load_state_dict(custom_weights)
    #         print("Loaded custom weights into the full model successfully.")
    #     except RuntimeError as e:
    #         print(f"Error loading weights into full model: {e}")
    #         raise ValueError("The weights in 'weights.pt' do not match the model architecture.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = DiceLoss(model='binary',from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 100  # short run for demonstration
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        for images, masks, label_flags in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            
            # Only compute loss where label_flag == 1
            labeled_indices = (label_flags == 1).nonzero(as_tuple=True)[0]
            if len(labeled_indices) > 0:
                labeled_outputs = outputs[labeled_indices]
                labeled_masks = masks[labeled_indices]

                loss = criterion(labeled_outputs, labeled_masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(labeled_indices)
        
        avg_train_loss = train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, label_flags in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                # Only compute loss for labeled
                labeled_indices = (label_flags == 1).nonzero(as_tuple=True)[0]
                if len(labeled_indices) > 0:
                    labeled_outputs = outputs[labeled_indices]
                    labeled_masks = masks[labeled_indices]
                    
                    loss = criterion(labeled_outputs, labeled_masks)
                    val_loss += loss.item() * len(labeled_indices)
        
        avg_val_loss = val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)

        tolerance = 0.015

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Save best weights
            patience_counter = 0
            print("Validation loss improved - model saved.")
        elif avg_val_loss < best_val_loss + tolerance:  
            # Within tolerance, do not update best_val_loss but reset patience  
            patience_counter = 0  
            print("No improvement on validation loss, but within tolerance - patience reset.") 
        else:
            patience_counter += 1
            print(f"No improvement on validation loss. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # -- Plot the training and validation loss --
    epochs_range = range(1, len(train_losses) + 1) # Set range to actual number of epochs run
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='orange')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_{len(epochs_range)}')
    plt.show()

    # Evaluate Dice score
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks, label_flags in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # Only compute metric for labeled examples
            labeled_indices = (label_flags == 1).nonzero(as_tuple=True)[0]
            if len(labeled_indices) == 0:
                continue

            labeled_preds = preds[labeled_indices]
            labeled_masks = masks[labeled_indices]

            intersection = (labeled_preds * labeled_masks).sum(dim=(1,2,3))
            union = labeled_preds.sum(dim=(1,2,3)) + labeled_masks.sum(dim=(1,2,3))
            dice = 2.0 * intersection / (union + 1e-8)
            dice_scores.extend(dice.cpu().numpy())

    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice on Validation Set: {mean_dice:.4f}")
