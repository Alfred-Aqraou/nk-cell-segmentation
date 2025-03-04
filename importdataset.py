import os, requests, tarfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

user = os.getenv('USER02466')
pswd = os.getenv('PASS02466')

if not (user and pswd):
    raise ValueError("Credentials not found, set the username and password environment variables. Instructions in README")

extract_dir = "dataset"

data_filename = "dataset.tar"
url = f"https://mathias.eksworld.dk/files/{data_filename}"

if not os.path.exists(data_filename):
    print("Downloading dataset...")
    response = requests.get(url, auth=(user, pswd))
    if response.status_code == 200:
        with open(data_filename, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Download failed: {response.status_code}")

    # Extract tar
    print("Extracting dataset...")
    with tarfile.open(data_filename, 'r') as tar:
        tar.extractall(path=".", filter="fully_trusted")
    
    os.remove("dataset.tar")

# Load with PyTorch
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset_path = os.path.join(extract_dir, "labeled", "train")
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in loader:
    print(f"Batch: {images.shape}, Labels: {labels}")
    break