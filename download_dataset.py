
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import NKCellDataset

if __name__ == '__main__':
    # Load with PyTorch
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = NKCellDataset(root_dir="dataset", transform=transform)
    if dataset.download():
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for images, labels in loader:
            print(f"Batch: {images.shape}, Labels: {labels}")
            break