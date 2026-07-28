import torch
import torch.nn.functional as F
from config.config import DICE_SMOOTH, IOU_EPSILON
from typing import Tuple


def get_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # (B,224,224)
    probs = torch.sigmoid(logits)

    # flatten
    batch_size = logits.shape[0]
    # (B,50176)
    probs = probs.view(batch_size, -1)
    # (B,50176)
    targets = targets.view(batch_size, -1)

    # (B,)
    numerator = 2 * (probs * targets).sum(dim=1)
    # (B,)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    # (B,)
    score = (numerator + DICE_SMOOTH) / (denominator + DICE_SMOOTH)
    loss = (1 - score).mean()

    return loss


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    dice_loss = get_dice_loss(logits, targets)
    return bce_loss + dice_loss


def calculate_miou_oiou(
    preds: torch.Tensor, targets: torch.Tensor
) -> Tuple[float, float]:
    # (B,224,224)
    preds = preds.bool()
    # (B,224,224)
    targets = targets.bool()

    # (B,)
    intersection = (preds & targets).sum(dim=(1, 2)).float()
    # (B,)
    union = (preds | targets).sum(dim=(1, 2)).float()

    # (B,)
    sample_ious = intersection / (union + IOU_EPSILON)
    miou = sample_ious.mean().item()

    total_intersection = intersection.sum().item()
    total_union = union.sum().item()
    oiou = total_intersection / (total_union + IOU_EPSILON)

    return miou, oiou


class TrainRecord:
    def __init__(self) -> None:
        self.best_val_loss = float("inf")
        self.no_improve_count = 0
