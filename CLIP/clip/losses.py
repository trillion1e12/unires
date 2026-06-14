from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _ensure_channel_dim(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.shape == reference.shape:
        return mask

    if mask.ndim == reference.ndim - 1 and reference.shape[1] == 1:
        return mask.unsqueeze(1)

    raise ValueError(
        "Target mask shape must match predictions or be missing only the channel dimension. "
        f"Got target={tuple(mask.shape)} and prediction={tuple(reference.shape)}."
    )


def dice_loss_from_probs(predictions: Tensor, targets: Tensor, smooth: float = 1.0, eps: float = 1e-7) -> Tensor:
    """Compute soft Dice loss for binary segmentation masks.

    Both tensors are expected to contain probabilities in [0, 1].
    """
    predictions = predictions.float()
    targets = targets.float()
    targets = _ensure_channel_dim(targets, predictions)

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape after channel alignment. "
            f"Got predictions={tuple(predictions.shape)} and targets={tuple(targets.shape)}."
        )

    reduce_dims = tuple(range(1, predictions.ndim))
    intersection = torch.sum(predictions * targets, dim=reduce_dims)
    denominator = torch.sum(predictions, dim=reduce_dims) + torch.sum(targets, dim=reduce_dims)
    dice_score = (2.0 * intersection + smooth) / (denominator + smooth + eps)
    return 1.0 - dice_score.mean()


def dice_loss_from_logits(logits: Tensor, targets: Tensor, smooth: float = 1.0, eps: float = 1e-7) -> Tensor:
    return dice_loss_from_probs(torch.sigmoid(logits), targets, smooth=smooth, eps=eps)


@dataclass
class SegmentationLossOutput:
    loss: Tensor
    dice_loss: Tensor
    bce_loss: Tensor


class SegmentationLoss(nn.Module):
    """Composite BCE + Dice loss for binary segmentation masks.

    The module accepts either a raw tensor of mask logits/probabilities or the
    dictionary returned by ``CLIP.forward`` in this repository.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.eps = eps

    def forward(
        self,
        predictions: Union[Tensor, Mapping[str, Tensor]],
        targets: Tensor,
    ) -> SegmentationLossOutput:
        if isinstance(predictions, Mapping):
            if "mask_logits" in predictions:
                logits = predictions["mask_logits"]
                probs = torch.sigmoid(logits)
                targets = _ensure_channel_dim(targets.float(), logits)
                bce_loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
            elif "mask" in predictions:
                probs = predictions["mask"].float()
                targets = _ensure_channel_dim(targets.float(), probs)
                bce_loss = F.binary_cross_entropy(probs.clamp(self.eps, 1.0 - self.eps), targets)
            else:
                raise KeyError("Expected a 'mask_logits' or 'mask' entry in the model outputs.")
        else:
            probs = predictions.float()
            targets = _ensure_channel_dim(targets.float(), probs)
            bce_loss = F.binary_cross_entropy(probs.clamp(self.eps, 1.0 - self.eps), targets)

        dice = dice_loss_from_probs(probs, targets, smooth=self.smooth, eps=self.eps)
        loss = self.bce_weight * bce_loss + self.dice_weight * dice

        return SegmentationLossOutput(loss=loss, dice_loss=dice, bce_loss=bce_loss)


def segmentation_loss(
    predictions: Union[Tensor, Mapping[str, Tensor]],
    targets: Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> SegmentationLossOutput:
    """Functional helper mirroring :class:`SegmentationLoss`."""
    return SegmentationLoss(
        dice_weight=dice_weight,
        bce_weight=bce_weight,
        smooth=smooth,
        eps=eps,
    )(predictions, targets)