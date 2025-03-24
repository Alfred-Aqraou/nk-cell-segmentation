from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import NKCellDataset

if __name__ == '__main__':
    # Load with PyTorch
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = NKCellDataset(dataset_dir="dataset", transform=transform)
    # Use verify parameter to check dataset validity, avoid using for every run as it takes longer
    if dataset.download(verify=False):
        with open("download.log", 'x') as file:
            file.write('Download successful!')
        try:
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            for images, labels in loader:
                print(f"Batch: {images.shape}, Labels: {labels}")
                break
        except ValueError:
            print("Unable to import data in PyTorch DataLoader, file structure isn't implemented correctly, yet...")