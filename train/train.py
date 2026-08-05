import os
from typing import Callable

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from config.config import CHECKPOINT_DIR, LOG_INTERVAL, PATIENCE, VAL_INTERVAL
from model.unires import UniRes
from train.utils import TrainRecord, calculate_miou_oiou
from utils.logger import get_logger, log_metrics

logger = get_logger(__name__)


def eval_loop(
    dataloader: DataLoader,
    model: UniRes,
    loss_fn: Callable,
    writer: SummaryWriter,
    device: str,
    global_step: int,
    is_test: bool = False,
) -> float:
    model.eval()

    num_batches = len(dataloader)
    total_loss = 0
    total_accuracy = 0
    total_miou = 0
    total_oiou = 0

    tag = "test" if is_test else "validate"

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Testing" if is_test else "Validating", leave=False)
        ):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["seg_masks"].to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            loss = loss_fn(logits, targets.float())

            preds = logits > 0
            accuracy = (targets == preds).float().mean().item()
            miou, oiou = calculate_miou_oiou(preds, targets)

            total_loss += loss.item()
            total_accuracy += accuracy
            total_miou += miou
            total_oiou += oiou

    avg_loss = total_loss / num_batches
    avg_acc = total_accuracy / num_batches
    avg_miou = total_miou / num_batches
    avg_oiou = total_oiou / num_batches

    writer.add_scalars(
        tag,
        {
            "loss": avg_loss,
            "accuracy": avg_acc,
            "miou": avg_miou,
            "oiou": avg_oiou,
        },
        global_step,
    )

    log_metrics(
        phase=tag,
        loss=avg_loss,
        accuracy=avg_acc,
        miou=avg_miou,
        oiou=avg_oiou,
        step=global_step,
    )
    logger.info("%s | loss=%.4f acc=%.4f miou=%.4f oiou=%.4f step=%d",
                 tag, avg_loss, avg_acc, avg_miou, avg_oiou, global_step)

    return avg_loss


def train_loop(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: UniRes,
    optimizer: Optimizer,
    loss_fn: Callable,
    writer: SummaryWriter,
    device: str,
    epoch_idx: int,
    record: TrainRecord,
) -> None:
    model.train()

    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(
        tqdm(train_loader, desc=f"Training epoch {epoch_idx}")
    ):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["seg_masks"].to(device)

        logits = model(pixel_values, input_ids, attention_mask)
        loss = loss_fn(logits, targets.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = num_batches * epoch_idx + batch_idx

        if batch_idx % LOG_INTERVAL == 0:
            preds = logits > 0
            accuracy = (targets == preds).float().mean().item()
            miou, oiou = calculate_miou_oiou(preds, targets)

            writer.add_scalars(
                "train",
                {
                    "loss": loss,
                    "accuracy": accuracy,
                    "miou": miou,
                    "oiou": oiou,
                },
                global_step,
            )

            log_metrics(
                phase="train",
                epoch=epoch_idx,
                batch=batch_idx,
                loss=loss.item(),
                accuracy=accuracy,
                miou=miou,
                oiou=oiou,
                step=global_step,
            )
            logger.info(
                "train | epoch=%d batch=%d step=%d loss=%.4f acc=%.4f miou=%.4f oiou=%.4f",
                epoch_idx, batch_idx, global_step, loss.item(), accuracy, miou, oiou,
            )

        if batch_idx % VAL_INTERVAL == 0:
            model.eval()
            val_loss = eval_loop(
                val_loader, model, loss_fn, writer, device, global_step
            )
            model.train()

            if val_loss < record.best_val_loss:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
                torch.save(model.state_dict(), checkpoint_path)
                record.best_val_loss = val_loss
                record.no_improve_count = 0
                logger.info("New best val_loss=%.4f — checkpoint saved", val_loss)
            else:
                record.no_improve_count += 1
                if record.no_improve_count > PATIENCE:
                    logger.warning(
                        "Early stopping at epoch %d, batch %d (no improvement for %d validations)",
                        epoch_idx, batch_idx, record.no_improve_count,
                    )
                    break
