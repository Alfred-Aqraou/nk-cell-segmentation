import os, requests, tarfile, hashlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# def md5_checksum(file_path):
#     hasher = hashlib.md5()
#     with open(file_path, 'rb') as f:
#         for chunk in iter(lambda: f.read(4096), b''):
#             hasher.update(chunk)
#     return hasher.hexdigest()

def main():
    user = os.getenv('USER02466')
    pswd = os.getenv('PASS02466')

    if not (user and pswd):
        print("Credentials not found, set the username and password environment variables. Instructions in README")
        return

    extract_dir = "dataset"

    data_filename = "dataset.tar" # MD5 Checksum: 3D6D34509E88B9741BF1614C1D6F1990
    url = f"https://mathias.eksworld.dk/files/{data_filename}"

    if not os.path.exists(extract_dir):
        print("Downloading dataset...")
        try:
            with requests.get(url, auth=(user, pswd), timeout=2, stream=True, verify=False) as response:
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
            expected_hash = "3D6D34509E88B9741BF1614C1D6F1990"
            if file_hash != expected_hash:
                print(f"Checksum mismatch! Expected {expected_hash}, got {file_hash}")
                os.remove(data_filename)
                raise Exception("Checksum verification failed")
            print("Download completed and checksum verified!")

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return


        # Extract tar
        print("Extracting dataset...")
        with tarfile.open(data_filename, 'r') as tar:
            tar.extractall(path=".", filter="fully_trusted")
        
        os.remove("dataset.tar")

    else:
        print(f"{data_filename} already exists, skipping download.")

    # Load with PyTorch
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset_path = os.path.join(extract_dir, "labeled", "train")
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, labels in loader:
        print(f"Batch: {images.shape}, Labels: {labels}")
        break

if __name__ == '__main__':
    main()