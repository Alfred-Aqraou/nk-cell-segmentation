import os, requests, tarfile, hashlib
from tqdm import tqdm
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import Tuple

class NKCellDataset(Dataset):
    def __init__(self, dataset_dir="dataset", transform=None, metadata_filter=None):
        self.dataset_dir = dataset_dir
        self._dataset_tar_path = "dataset.tar"
        self.transform = transform

        # Paths
        self.labeled_fluorescent_img = os.path.join(dataset_dir, 'fluorescent', 'with_masks', 'images')
        self.labeled_fluorescent_mask = os.path.join(dataset_dir, 'fluorescent', 'with_masks', 'masks')
        self.unlabeled_fluorescent = os.path.join(dataset_dir, 'fluorescent', 'without_masks')

        # Metadata (optional filtering, e.g., t>28h)
        metadata_path = os.path.join(dataset_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else None

        # Collect data
        self.image_paths = []
        self.mask_paths = []  # None for unlabeled
        self.is_labeled = []
        
        # Labeled (with masks)
        if os.path.exists(self.labeled_fluorescent_img):
            for img_name in os.listdir(self.labeled_fluorescent_img):
                if img_name.endswith('.tif'):
                    img_path = os.path.join(self.labeled_fluorescent_img, img_name)
                    mask_name = img_name.replace('.tif', '_GT.tif')
                    mask_path = os.path.join(self.labeled_fluorescent_mask, mask_name)
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.is_labeled.append(True)
        
        # Unlabeled (without masks)
        if os.path.exists(self.unlabeled_fluorescent):
            for img_name in os.listdir(self.unlabeled_fluorescent):
                if img_name.endswith('.tif'):
                    img_path = os.path.join(self.unlabeled_fluorescent, img_name)
                    self.image_paths.append(img_path)
                    self.mask_paths.append(None)
                    self.is_labeled.append(False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[Image.Image, torch.Tensor, int]:
        #print(f"Loading sample: {idx}")  # temporary debug message
        
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

    def _compute_md5(self, file_path) -> str:
        """ Compute the MD5 checksum of a file. """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest().upper()

    def download(self, verify=False) -> bool:
        '''
        Download the dataset

        verify: Uses double space to verify integrity
        '''
        user = os.getenv('USER02466')
        pswd = os.getenv('PASS02466')

        if not (user and pswd):
            print("Credentials not found, set the username and password environment variables. Instructions in README")
            return False

        expected_hash = "957159D273EAD51073D490FE307768E9" # MD5
        url = f"https://mathias.eksworld.dk/files/{self._dataset_tar_path}"

        if verify:
            if os.path.exists(self.dataset_dir):
                print("Validating current dataset...")

                if os.path.exists(self._dataset_tar_path):
                    computed_hash = self._compute_md5(self._dataset_tar_path)
                    if computed_hash == expected_hash:
                        print("Local dataset archive matches expected checksum. Skipping download.")
                        return True
                    else:
                        print(f"Checksum mismatch: Expected {expected_hash}, but got {computed_hash}.", end=" ")
                else:
                    print("Could not find tar file, verifying requires that you use `verify=True` when downloading first time. If deleted, the dataset cannot be verified, but doing so also uses double the space.", end=" ")

        print("Downloading dataset...")
        try:
            with requests.get(url, auth=(user, pswd), timeout=5, stream=True, verify=False) as response:
                response.raise_for_status()
                print("Found file, downloading...")

                # Get the total file size from the Content-Length header (if available)
                total_size = int(response.headers.get("content-length", 0))
                with open(self._dataset_tar_path, 'wb') as f:
                    with tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading") as pbar:
                        # Download in chunks
                        for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                            if chunk:  # Filter out keep-alive chunks
                                size = f.write(chunk)
                                pbar.update(size)

            # Verify downloaded file checksum
            downloaded_hash = self._compute_md5(self._dataset_tar_path)
            if downloaded_hash != expected_hash:
                print(f"Checksum mismatch! Expected {expected_hash}, got {downloaded_hash}")
                os.remove(self._dataset_tar_path)
                print("Redownload required, sorry...")
                return False
            print("Download completed and checksum verified!")

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return False

        # Extract tar file
        print("Extracting dataset...")
        try:
            with tarfile.open(self._dataset_tar_path, 'r') as tar:
                tar.extractall(path=".", filter="fully_trusted", numeric_owner=True)
        except:
            print("Could not extract dataset due to permission errors, please do it manually in terminal with:")
            print("tar -xvf dataset.tar")
            return False

        print("Extracted successfully!")
        if not verify:
            os.remove(self._dataset_tar_path)
        return True