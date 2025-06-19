import os, requests, tarfile, hashlib, re, sys
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from natsort import natsorted
import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import Tuple 
from concurrent.futures import ThreadPoolExecutor
from typing import List

# === Utility for extracting hour from filename ===
import re

def extract_hour_from_filename(filename):
    match = re.search(r'_(\d+)h(\d+)m(\d+)s_', filename)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        total_hours = hours + minutes / 60 + seconds / 3600
        return total_hours
    else:
        return None

class JointTransform:
    def __init__(self, image_transform, mask_transform, seed):
        self.seed = seed
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, img, mask):
        torch.manual_seed(self.seed)
        img = self.image_transform(img)

        torch.manual_seed(self.seed)
        mask = self.mask_transform(mask)

        return img, mask

class NKCellDataset(Dataset):
    def __init__(self, dataset_dir="dataset", transform=None, metadata_filter=None, mode='fluorescent'):
        self.dataset_dir = dataset_dir
        self._dataset_tar_path = "dataset.tar"
        self.transform: JointTransform = transform
        self.is_joint_transform = isinstance(transform, JointTransform)
        self.mode = mode

        # Paths
        self.labeled_fluorescent_img = os.path.join(dataset_dir, 'fluorescent', 'with_masks', 'images')
        self.labeled_fluorescent_mask = os.path.join(dataset_dir, 'fluorescent', 'with_masks', 'masks')
        # self.unlabeled_fluorescent = os.path.join(dataset_dir, 'fluorescent', 'without_masks')

        self.labeled_broadband_img = os.path.join(dataset_dir, 'broadband', 'with_masks', 'images')
        self.labeled_broadband_mask = os.path.join(dataset_dir, 'broadband', 'with_masks', 'masks')
        # self.unlabeled_broadband = os.path.join(dataset_dir, 'broadband', 'without_masks')

        # Metadata (optional filtering, e.g., t>28h)
        metadata_path = os.path.join(dataset_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else None

        # Collect data
        self.image_paths = []
        self.mask_paths = []  # None for unlabeled
        self.is_labeled = []
        
         # === Load based on mode ===
        if self.mode == 'fluorescent':
            image_folder = self.labeled_fluorescent_img
            mask_folder = self.labeled_fluorescent_mask
        elif self.mode == 'broadband':
            image_folder = self.labeled_broadband_img
            mask_folder = self.labeled_broadband_mask

        if os.path.exists(image_folder):
            for img_name in os.listdir(image_folder):
                if img_name.endswith('.tif'):
                    img_path = os.path.join(image_folder, img_name)
                    mask_name = img_name.replace('.tif', '_GT.tif')
                    mask_path = os.path.join(mask_folder, mask_name)
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.is_labeled.append(True)
        
        # Unlabeled (without masks)
        # if os.path.exists(self.unlabeled_fluorescent):
        #     for img_name in os.listdir(self.unlabeled_fluorescent):
        #         if img_name.endswith('.tif'):
        #             img_path = os.path.join(self.unlabeled_fluorescent, img_name)
        #             self.image_paths.append(img_path)
        #             self.mask_paths.append(None)
        #             self.is_labeled.append(False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[Image.Image, torch.Tensor, int, str]:
        #print(f"Loading sample: {idx}")  # temporary debug message
        
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        if mask_path:  # Labeled
            mask = Image.open(mask_path).convert('L')
            if self.transform:
                if self.is_joint_transform:
                    image, mask = self.transform(image, mask)
                else:
                    image = self.transform(image)
                    mask = self.transform(mask)
            return image, mask, 1, img_path  # 1 indicates labeled + return img_path
        else:  # Unlabeled
            if self.transform:
                image = self.transform(image)
            return image, torch.zeros_like(image[0:1]), 0, img_path  # Dummy mask, 0 indicates unlabeled + return img_path

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
        temp_path = self._dataset_tar_path + ".part"

        resume_header = {}
        downloaded_bytes = 0
        if os.path.exists(temp_path):
            downloaded_bytes = os.path.getsize(temp_path)
            resume_header = {'Range': f'bytes={downloaded_bytes}-'}

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
            with requests.get(url, auth=(user, pswd), headers=resume_header, timeout=60, stream=True, verify=False) as response:
                if response.status_code not in [200, 206]:
                    print(f"Unexpected status code {response.status_code}")
                    return False
                # response.raise_for_status()
                # print("Found file, downloading...")

                # Get the total file size from the Content-Length header (if available)
                total_size = int(response.headers.get("content-length", 0))
                mode = 'ab' if downloaded_bytes else 'wb'


                with open(temp_path, mode) as f:
                    with tqdm(total=total_size, initial=downloaded_bytes, unit="iB", unit_scale=True, desc="Downloading", dynamic_ncols=True, file=sys.stdout, ascii=True, mininterval=1) as pbar:
                        # Download in chunks
                        for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                            if chunk:  # Filter out keep-alive chunks
                                size = f.write(chunk)
                                pbar.update(size)
            
            os.rename(temp_path, self._dataset_tar_path)

            # Verify downloaded file checksum
            print("Checking hash...", end="\r")
            downloaded_hash = self._compute_md5(self._dataset_tar_path)
            if downloaded_hash != expected_hash:
                print(f"Checksum mismatch! Expected {expected_hash}, got {downloaded_hash}")
                if str(input("Do you want to delete downloaded file that doesn't match? (y/n): ")).strip().lower() in ("true", "1", "yes", "y", "t"):
                    os.remove(self._dataset_tar_path)
                else:
                    print("You must extract tar manually: `tar -xvf dataset.tar`")
                print("Check that checksum has been modified, or redownload...")
                return False
            print("Download completed and checksum verified!")

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return False
        except KeyboardInterrupt:
            print("Interrupted download")
            return False

        # Extract tar file
        print("Extracting dataset...", end="\r")
        try:
            with tarfile.open(self._dataset_tar_path, 'r') as tar:
                tar.extractall(path=".", filter="fully_trusted", numeric_owner=True)
        except Exception as e:
            print(f"Extraction failed: {e}")
            print("Try extracting manually: `tar -xvf dataset.tar`")
            return False

        print("Extracted successfully!")
        if not verify:
            os.remove(self._dataset_tar_path)
        return True

    def _parse_timestamp(self, timestamp):
        # Assuming this method exists to convert timestamp to seconds for sorting
        # Placeholder implementation
        h, m, s = re.match(r'(\d+)h(\d+)m(\d+)s', timestamp).groups()
        return int(h) * 3600 + int(m) * 60 + int(s)

    def _parse_filename(self, pattern, file):
        # Assuming this method exists to extract timestamp and well_id
        match = pattern.search(os.path.basename(file))
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _get_images_by_time(self, mode='fluorescent', verbose=False):
        """Map timestamps to well IDs and their file paths instead of loading images."""
        folder = os.path.join(self.dataset_dir, mode, 'without_masks')
        files = natsorted(glob(os.path.join(folder, '*.tif')))
        time_map = {}  # {timestamp: {well_id: file_path}}
        pattern = re.compile(r'_(\d+h\d+m\d+s)_([A-H]\d{1,2})_')

        for file in tqdm(files, desc="Mapping image paths by time", disable=not verbose):
            timestamp, well_id = self._parse_filename(pattern, file)
            if not timestamp or not well_id:
                continue
            if timestamp not in time_map:
                time_map[timestamp] = {}
            time_map[timestamp][well_id] = file  # Store file path

        return time_map

    def _load_and_process_image(self, file_path: str, downscale: int):
        """Load and process an image from its file path."""
        img = Image.open(file_path).convert('RGB')
        if self.giftransform:
            img = self.transform(img)
            img = T.ToPILImage()(img)
        if downscale != 1:
            img = img.resize((img.width // downscale, img.height // downscale), Image.LANCZOS)
        return img

    def create_well_grid_gif(self, cpu_cores: int = 8, mode: str = 'fluorescent', downscale: int = 1, frame_duration: int = 200, gif_dir: str = 'illustrations/gifs', verbose: bool = False, transform: bool = False):
        """Create a GIF with lazy loading of images per frame."""
        assert mode in ['fluorescent', 'broadband'], "Mode must be 'fluorescent' or 'broadband'"

        self.giftransform: bool = transform

        os.makedirs(gif_dir, exist_ok=True)
        rows = list("ABCDEFGH")  # 8 rows: A to H
        cols = list(map(str, range(1, 13)))  # 12 columns: 1 to 12

        # Get file paths mapped by timestamp
        time_map = self._get_images_by_time(mode=mode, verbose=verbose)
        sorted_times = sorted(time_map.keys(), key=self._parse_timestamp)

        # Print timestamps to verify chronological order
        # if verbose:
        #     print("Timestamps in chronological order:")
        #     for ts in sorted_times:
        #         total_seconds = self._parse_timestamp(ts)
        #         print(f"  {ts}: {total_seconds} seconds")

        # Determine image size from the first image
        first_timestamp = sorted_times[0]
        first_file = time_map[first_timestamp][next(iter(time_map[first_timestamp]))]
        sample_img = Image.open(first_file).convert('RGB')
        if self.giftransform:
            sample_img = self.transform(sample_img)
            if isinstance(sample_img, torch.Tensor):
                sample_img = T.ToPILImage()(sample_img)
        if downscale != 1:
            sample_img = sample_img.resize((sample_img.width // downscale, sample_img.height // downscale), Image.LANCZOS)
        img_width, img_height = sample_img.size

        font_size = max(16, (img_height // 1.5) // (downscale // 4 + 1))
        padding_height = int(font_size + 20)  # Adjust padding based on downscale

        # Calculate grid dimensions
        grid_width = img_width * 12
        grid_height = img_height * 8 + padding_height
        if grid_width * grid_height > 89478485:
            print("GIF size exceeds pixel limit, skipping GIF creation.")
            return

        frames: List[Image.Image] = []
        for frame_idx, timestamp in tqdm(enumerate(sorted_times), total=len(sorted_times), desc="Creating GIF frames", disable=not verbose):
            # Load images for this timestamp in parallel
            with ThreadPoolExecutor(max_workers=cpu_cores) as executor:
                futures = {executor.submit(self._load_and_process_image, file_path, downscale): well_id 
                           for well_id, file_path in time_map[timestamp].items()}
                images = {}
                for future in futures:
                    well_id = futures[future]
                    try:
                        images[well_id] = future.result()
                    except Exception as e:
                        print(f"Error loading image for {well_id}: {e}")

            # Create canvas for the frame
            canvas = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)

            # Adjust font size based on downscale
            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except IOError:
                font = ImageFont.load_default()

            # Draw timestamp
            draw.text((10, 10), timestamp, fill=(255, 255, 255), font=font)

            # Paste images onto the canvas
            for r_idx, row in enumerate(rows):
                for c_idx, col in enumerate(cols):
                    well_id = f"{row}{col}"
                    img = images.get(well_id)
                    if img:
                        x = c_idx * img_width
                        y = r_idx * img_height + padding_height
                        canvas.paste(img, (x, y))

            frames.append(canvas)

        # Save the GIF
        if frames:
            gif_path = os.path.join(gif_dir, f'{"transformed_" if transform else ""}{mode}_grid_{grid_width}x{grid_height}.gif')
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)
            print(f"Saved animated {mode} grid GIF to {gif_path}")
        else:
            print("No frames generated.")

    @property
    def label_flags(self):
        return [1 if labeled else 0 for labeled in self.is_labeled]

if __name__ == "__main__":
    import torchvision.transforms as T
    transform = T.Compose([
        T.RandomCrop((256, 256)),
        # T.RandomResizedCrop(size=256, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.ToTensor(),
    ])
    dataset = NKCellDataset(dataset_dir="dataset", transform=transform)
    # dataset.download(verify=True)  # Set verify=True for the first download to check integrity
    dataset.create_well_grid_gif(mode='fluorescent', verbose=True, downscale=5, transform=False)
    # dataset.create_well_grid_gif(mode='broadband', verbose=True, downscale=5)