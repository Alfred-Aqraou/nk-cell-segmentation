# segment_image.py
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os, sys, argparse
from PIL import Image
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.unet import create_unet, create_unet_with_dino_resnet18

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, encoder_name='resnet50', freeze_encoder=False):
    if ".ckpt" in encoder_name:
        model = create_unet_with_dino_resnet18(
            ckpt_path=os.path.abspath(encoder_name),
            in_channels=3,
            classes=1,
            freeze_encoder=freeze_encoder
        )
    else:
        model = create_unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            freeze_encoder=freeze_encoder
        )
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def transform_image(img_path, resize):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0), img.resize((resize, resize))  # tensor, resized original PIL

def load_mask(mask_path, resize):
    mask = Image.open(mask_path).convert("L")
    return mask.resize((resize, resize))

def segment_image(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        return output.squeeze().cpu().numpy()

def plot_results(orig_img, pred_mask, output_path, gt_mask=None):
    fig, axs = plt.subplots(1, 3 if gt_mask else 2, figsize=(12, 4))

    axs[0].imshow(orig_img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    if gt_mask:
        masked_img = np.array(orig_img).copy()
        red_mask = np.zeros_like(masked_img)
        red_mask[..., 0] = np.array(gt_mask)
        blended = np.clip(masked_img * 0.6 + red_mask * 0.4, 0, 255).astype(np.uint8)
        axs[1].imshow(blended)
        axs[1].set_title("Image with GT Mask")
        axs[1].axis('off')

    axs[2 if gt_mask else 1].imshow(pred_mask, cmap='gray')
    axs[2 if gt_mask else 1].set_title("Predicted Mask")
    axs[2 if gt_mask else 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Output grid saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--encoder", required=True, help="Encoder name or path to DINO .ckpt")
    parser.add_argument("--output", default="output_mask.png", help="Path to save output visualization")
    parser.add_argument("--resize", type=int, default=256, help="Resize input to this size (square)")
    parser.add_argument("--freeze", action="store_true", help="Whether to freeze encoder")
    parser.add_argument("--mask", help="(Optional) Path to ground-truth mask image")

    args = parser.parse_args()

    image_tensor, resized_img = transform_image(args.image, args.resize)
    model = load_model(args.checkpoint, args.encoder, args.freeze)
    pred_mask = segment_image(model, image_tensor)

    gt_mask = load_mask(args.true_mask, args.resize) if args.true_mask else None
    plot_results(resized_img, pred_mask, args.output, gt_mask=gt_mask)
