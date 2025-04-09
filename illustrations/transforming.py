import os
import random
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from typing import List
from matplotlib.axes import Axes

# Define the transform
transform = T.Compose([
    T.CenterCrop((960, 960)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    T.ToTensor()  # Converts to (C, H, W)
])

# Directory containing images
image_dir = "dataset/fluorescent/with_masks/images"
all_images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

# Randomly select 9 images
selected_images = random.sample(all_images, min(9, len(all_images)))

# Create a 5x5 grid plot
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes: List[Axes] = axes.ravel()  # Flatten for easy iteration

for idx, img_name in enumerate(selected_images):
    img_path = os.path.join(image_dir, img_name)
    img = Image.open(img_path)
    
    # Apply transforms
    transformed_img = transform(img)
    
    # Convert tensor to (H, W, C) for matplotlib
    img_to_plot = transformed_img.permute(1, 2, 0).numpy()
    
    # Plot
    axes[idx].imshow(img_to_plot)
    axes[idx].axis('off')
    axes[idx].set_title(img_name, fontsize=10)  # Shorten filename for title

plt.tight_layout()
plt.savefig("grid.pdf")
plt.savefig("grid.png")
plt.show()