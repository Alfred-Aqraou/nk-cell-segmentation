# models/unet.py

import segmentation_models_pytorch as smp
import torch

def create_unet(encoder_name: str = "resnet34",
                encoder_weights: str = "imagenet",
                in_channels: int = 3,
                classes: int = 1,
                freeze_encoder: bool = True) -> torch.nn.Module:
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
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    
    if freeze_encoder:
        # Freeze encoder layers to preserve pretrained features
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model