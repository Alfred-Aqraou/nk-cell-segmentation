import os, requests, tarfile, hashlib
from tqdm import tqdm
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torch

class NKCellDataset(Dataset):
    def __init__(self, root_dir, transform=None, metadata_filter=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Paths
        self.fluorescent_with_masks_img = os.path.join(root_dir, 'fluorescent', 'with_masks', 'images')
        self.fluorescent_with_masks_mask = os.path.join(root_dir, 'fluorescent', 'with_masks', 'masks')
        self.fluorescent_without_masks = os.path.join(root_dir, 'fluorescent', 'without_masks')
        
        # Metadata (optional filtering, e.g., t>28h)
        metadata_path = os.path.join(root_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else None
        
        # Collect data
        self.image_paths = []
        self.mask_paths = []  # None for unlabeled
        self.is_labeled = []
        
        # Labeled (with masks)
        if os.path.exists(self.fluorescent_with_masks_img):
            for img_name in os.listdir(self.fluorescent_with_masks_img):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(self.fluorescent_with_masks_img, img_name)
                    mask_name = img_name.replace('.jpg', '_mask.tiff')
                    mask_path = os.path.join(self.fluorescent_with_masks_mask, mask_name)
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.is_labeled.append(True)
        
        # Unlabeled (without masks)
        if os.path.exists(self.fluorescent_without_masks):
            for img_name in os.listdir(self.fluorescent_without_masks):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(self.fluorescent_without_masks, img_name)
                    self.image_paths.append(img_path)
                    self.mask_paths.append(None)
                    self.is_labeled.append(False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        if mask_path:  # Labeled
            mask = Image.open(mask_path).convert('L')
            if self.transform:
                mask = self.transform(mask)
            return image, mask, 1  # 1 indicates labeled
        else:  # Unlabeled
            return image, torch.zeros_like(image[0:1]), 0  # Dummy mask, 0 indicates unlabeled

    def download(self):
        user = os.getenv('USER02466')
        pswd = os.getenv('PASS02466')

        if not (user and pswd):
            print("Credentials not found, set the username and password environment variables. Instructions in README")
            return False

        expected_hash = "3D6D34509E88B9741BF1614C1D6F1990" # MD5
        data_filename = f"dataset.tar"
        url = f"https://mathias.eksworld.dk/files/{data_filename}"

        if not os.path.exists(self.root_dir):
            print("Downloading dataset...")
            try:
                with requests.get(url, auth=(user, pswd), timeout=5, stream=True, verify=False) as response:
                    response.raise_for_status()
                    print("Found file, downloading...")

                    # Get the total file size from the Content-Length header (if available)
                    total_size = int(response.headers.get("content-length", 0))
                    with open(data_filename, 'wb') as f:
                        with tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading") as pbar:
                            # Download in chunks
                            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                                if chunk:  # Filter out keep-alive chunks
                                    size = f.write(chunk)
                                    pbar.update(size)

                # Verify MD5 checksum
                with open(data_filename, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest().upper()
                
                if file_hash != expected_hash:
                    print(f"Checksum mismatch! Expected {expected_hash}, got {file_hash}")
                    os.remove(data_filename)
                    print("Checksum verification failed")
                    return False
                print("Download completed and checksum verified!")

            except requests.exceptions.RequestException as e:
                print(f"Download failed: {e}")
                return False

            # Extract tar
            print("Extracting dataset...")
            with tarfile.open(data_filename, 'r') as tar:
                tar.extractall(path=".", filter="fully_trusted")
            
            os.remove("dataset.tar")

        else:
            print(f"{data_filename} already exists, skipping download.")
        
        # Return true for successful or existing download
        return True