# models/unet.py

import segmentation_models_pytorch as smp
import torch

def create_unet(encoder_name: str = "resnet34",
                encoder_weights: str = "imagenet",
                in_channels: int = 3,
                classes: int = 1,
                freeze_encoder: bool = True,
                custom_encoder: torch.nn.Module = None) -> torch.nn.Module:
    """
    Create a U-Net model using segmentation_models_pytorch with the specified configuration.

    Args:
        encoder_name (str): Name of the encoder architecture.
        encoder_weights (str): Pretrained weights for the encoder.
        in_channels (int): Number of input channels.
        classes (int): Number of output classes.
        freeze_encoder (bool): Whether to freeze the encoder weights (default: True).

    Returns:
        model (torch.nn.Module): The configured U-Net model.
    """

    model = smp.Unet(
        encoder_name=None if custom_encoder else encoder_name,
        encoder_weights=None if custom_encoder else encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )

    if custom_encoder is not None:
        # model.encoder.load_state_dict(custom_encoder.state_dict())
        model.encoder = custom_encoder
    
    if freeze_encoder:
        # Freeze encoder layers to preserve pretrained features
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model

# models/unet.py

import segmentation_models_pytorch as smp
import torch
import torchvision.models as models

def create_unet_with_dino_resnet18(ckpt_path,
                                   in_channels=3,
                                   classes=1,
                                   freeze_encoder=True) -> torch.nn.Module:
    """
    Create a U-Net model with ResNet18 encoder initialized from a DINO-pretrained checkpoint.

    Args:
        ckpt_path (str): Path to the DINO checkpoint (.ckpt file).
        in_channels (int): Number of input channels.
        classes (int): Number of output classes.
        freeze_encoder (bool): Whether to freeze the encoder weights.

    Returns:
        model (torch.nn.Module): U-Net model with DINO-initialized encoder.
    """
    # Step 1: Create base Unet with resnet18 encoder
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,  # Don't load imagenet weights — we'll load DINO weights!
        in_channels=in_channels,
        classes=classes,
    )

    # Step 2: Load DINO-pretrained ResNet18 weights
    dino_resnet = models.resnet18()
    ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    # DINO checkpoint usually has keys like "backbone.layer1.0.conv1.weight"
    # We want to load only those into dino_resnet

    new_state_dict = {}
    for k, v in ckpt.items():
        if k.startswith("backbone."):
            # Strip "backbone." prefix to match ResNet keys
            new_key = k[len("backbone."):]
            new_state_dict[new_key] = v

    # Now load into resnet18
    missing_keys, unexpected_keys = dino_resnet.load_state_dict(new_state_dict, strict=False)
    # print(f"[INFO] Loaded DINO weights into ResNet18. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

    # Step 3: Load ResNet18 weights into Unet encoder
    model.encoder.load_state_dict(dino_resnet.state_dict(), strict=False)

    # Step 4: Optionally freeze encoder
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model