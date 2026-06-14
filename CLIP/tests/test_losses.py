import pytest

torch = pytest.importorskip("torch")

from clip.losses import SegmentationLoss, dice_loss_from_probs, segmentation_loss


def test_dice_loss_is_zero_for_perfect_prediction():
    prediction = torch.ones(2, 1, 4, 4)
    target = torch.ones(2, 1, 4, 4)

    loss = dice_loss_from_probs(prediction, target)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_segmentation_loss_combines_bce_and_dice():
    predictions = {
        "mask_logits": torch.tensor([[[[0.0, 2.0], [2.0, 0.0]]]], dtype=torch.float32),
    }
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)

    result = SegmentationLoss()(predictions, target)
    functional_result = segmentation_loss(predictions, target)

    assert result.loss.ndim == 0
    assert result.dice_loss.ndim == 0
    assert result.bce_loss.ndim == 0
    assert torch.isclose(result.loss, functional_result.loss)
    assert torch.isclose(result.dice_loss, functional_result.dice_loss)
    assert torch.isclose(result.bce_loss, functional_result.bce_loss)