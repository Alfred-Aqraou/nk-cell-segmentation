import os, requests, tarfile, hashlib, re, sys
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from natsort import natsorted
import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import Tuple

class JointTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img, mask):
        seed = torch.seed()
        torch.manual_seed(seed)
        img = self.base_transform(img)

        torch.manual_seed(seed)
        mask = self.base_transform(mask)

        return img, mask

class NKCellDataset(Dataset):
    def __init__(self, dataset_dir="dataset", transform=None, metadata_filter=None):
        self.dataset_dir = dataset_dir
        self._dataset_tar_path = "dataset.tar"
        self.transform = transform
        self.is_joint_transform = isinstance(transform, JointTransform)

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
        # if os.path.exists(self.unlabeled_fluorescent):
        #     for img_name in os.listdir(self.unlabeled_fluorescent):
        #         if img_name.endswith('.tif'):
        #             img_path = os.path.join(self.unlabeled_fluorescent, img_name)
        #             self.image_paths.append(img_path)
        #             self.mask_paths.append(None)
        #             self.is_labeled.append(False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[Image.Image, torch.Tensor, int]:
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
            return image, mask, 1  # 1 indicates labeled
        else:  # Unlabeled
            if self.transform:
                image = self.transform(image)
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

    def _parse_timestamp(self, timestamp):
        # Extract hours, minutes, seconds from timestamp (e.g., '10h42m06s')
        pattern = re.compile(r'(\d+)h(\d+)m(\d+)s')
        match = pattern.match(timestamp)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds
        return 0  # Fallback for invalid timestamps

    def _parse_filename(self, pattern, filename):
        match = pattern.search(filename)
        if match:
            timestamp, well_id = match.groups()
            return timestamp, well_id
        return None, None

    def _get_images_by_time(self, downscale, mode='fluorescent', verbose=False):
        folder = os.path.join(self.dataset_dir, mode, 'without_masks')
        files = natsorted(glob(os.path.join(folder, '*.tif')))
        time_map = {}  # {timestamp: {well_id: Image}}
        pattern = re.compile(r'_(\d+h\d+m\d+s)_([A-H]\d{1,2})_')
        
        for file in tqdm(files, desc="Mapping images by time", disable=not verbose):
            timestamp, well_id = self._parse_filename(pattern, file)
            if not timestamp or not well_id:
                continue
            if timestamp not in time_map:
                time_map[timestamp] = {}
            img = Image.open(file).convert('RGB')
            if downscale != 1:
                img = img.resize((img.width // downscale, img.height // downscale), Image.LANCZOS)  # Downscale
            time_map[timestamp][well_id] = img

        return time_map

    def create_well_grid_gif(self, mode: str = 'fluorescent', downscale: int = 1, frame_duration: int = 150, gif_dir: str = 'illustrations/gifs', verbose: bool = False):
        assert mode in ['fluorescent', 'broadband'], "Mode must be 'fluorescent' or 'broadband'"

        os.makedirs(gif_dir, exist_ok=True)
        rows = list("ABCDEFGH") # A to H
        cols = list(map(str, range(1, 13))) # 1 to 12

        time_map = self._get_images_by_time(mode=mode, downscale=downscale, verbose=verbose)
        sorted_times = sorted(time_map.keys(), key=self._parse_timestamp)  # Ensure chronological order

        # Print timestamps and their total seconds to prove chronological sorting
        print("Timestamps in chronological order:")
        for ts in sorted_times:
            total_seconds = self._parse_timestamp(ts)
            print(f"  {ts}: {total_seconds} seconds")


        # Determine image size from the first image
        sample_img = next(iter(next(iter(time_map.values())).values()))
        img_width, img_height = sample_img.size
        grid_width = img_width * 12

        # Adjust font size based on downscale
        font_size = max(16, (img_height // 1.5) // (downscale // 4 + 1))
        padding_height = int(font_size + 20)  # Adjust padding based on downscale
        grid_height = img_height * 8 + padding_height

        frames = []
        for frame_idx, timestamp in tqdm(enumerate(sorted_times), total=84, position=0, desc="Creating GIF frames", disable=not verbose):
            canvas = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)

            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except IOError:
                font = ImageFont.load_default()

            # Draw timestamp at top-left
            draw.text((10, 10), timestamp, fill=(255, 255, 255), font=font)

            for r_idx, row in enumerate(rows):
                for c_idx, col in enumerate(cols):
                    well_id = f"{row}{col}"
                    img = time_map[timestamp].get(well_id)
                    if img:
                        x = c_idx * img_width
                        y = r_idx * img_height + padding_height # Shift down by padding
                        canvas.paste(img, (x, y))
            frames.append(canvas)

        if frames:
            gif_path = os.path.join(gif_dir, f'{mode}_grid_{grid_width}x{grid_height}.gif')
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)
            print(f"Saved animated {mode} grid GIF to {gif_path}")
        else:
            print("No frames generated.")


if __name__ == "__main__":
    dataset = NKCellDataset()
    # dataset.download(verify=True)  # Set verify=True for the first download to check integrity
    dataset.create_well_grid_gif(mode='fluorescent', downscale=4, verbose=True)
    dataset.create_well_grid_gif(mode='broadband', downscale=4, verbose=True)