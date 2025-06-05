import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# =====================
# Dataset
# =====================
class NKSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# =====================
# Decoder
# =====================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(768, 256, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.decode(x)

# =====================
# Segmenter Model
# =====================
class Segmenter(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.decoder = Decoder()

    def forward(self, x):
        x = self.vit.get_intermediate_layers(x, n=1, reshape=True)[0]
        return self.decoder(x)

# =====================
# Training utils
# =====================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device).float()

        preds = model(images)
        preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear')

        loss = criterion(preds.squeeze(1), masks.squeeze(1) / 255.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# =====================
# Visualization
# =====================
def show_prediction(image_tensor, pred_mask, true_mask):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    pred = pred_mask.squeeze().detach().cpu().numpy()
    true = true_mask.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(image); plt.title("Input Image")
    plt.subplot(1, 3, 2); plt.imshow(pred > 0.5); plt.title("Predicted Mask")
    plt.subplot(1, 3, 3); plt.imshow(true > 0.5); plt.title("Ground Truth")
    plt.show()

# =====================
# Main (example usage)
# =====================
if __name__ == "__main__":
    from torchvision.models import vit_b_16
    from torchvision.models.vision_transformer import ViT_B_16_Weights

    # Modify this to your paths
    image_dir = "/absolute/path/to/with_masks/images"
    mask_dir = "/absolute/path/to/with_masks/masks"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = NKSegmentationDataset(image_dir, mask_dir, transform, mask_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load backbone
    vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = Segmenter(vit).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train one epoch
    loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
    print("Training loss:", loss)

    # Show a prediction
    sample_img, sample_mask = dataset[0]
    with torch.no_grad():
        pred_mask = model(sample_img.unsqueeze(0).to(device))
    show_prediction(sample_img, pred_mask, sample_mask)