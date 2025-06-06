# utils/metrics.py

import torch

def dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-8) -> torch.Tensor:
    """
    Compute the Dice Score per sample.

    Args:
        preds (torch.Tensor): Predicted binary tensor of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth binary tensor with the same shape as preds.
        smooth (float, optional): Smoothing constant to avoid division by zero (default: 1e-8).

    Returns:
        torch.Tensor: Dice scores for each sample in the batch.
    """
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice

def mean_dice_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes the mean Dice Score over a batch of predictions.

    Args:
        preds (torch.Tensor): The model output tensor (logits or probabilities) of shape (B, C, H, W).
        targets (torch.Tensor): The ground truth mask tensor with the same shape.
        threshold (float, optional): Threshold to binarize predictions (default: 0.5).

    Returns:
        float: The mean Dice score across the batch.
    """
    # Convert logits to binary predictions if necessary:
    # If preds are not already between 0 and 1, apply sigmoid.
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)
    binary_preds = (preds > threshold).float()
    dice = dice_score(binary_preds, targets)
    return dice.mean().item()