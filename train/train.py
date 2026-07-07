import os

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable
from config import CHECKPOINT_DIR, LOG_INTERVAL, PATIENCE, VAL_INTERVAL
from model.unires import UniRes
from train.utils import TrainRecord, calculate_miou_oiou


def eval_loop(
    dataloader: DataLoader,
    model: UniRes,
    loss_fn: Callable,
    writer: SummaryWriter,
    device: str,
    global_step: int,
) -> float:
    model.eval()

    num_batches = len(dataloader)
    total_loss = 0
    total_accuracy = 0
    total_miou = 0
    total_oiou = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Evaluating", leave=False)
        ):
            # extract batch data
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["seg_masks"].to(device)

            # forward & loss
            logits = model(pixel_values, input_ids, attention_mask)
            loss = loss_fn(logits, targets.float())

            # logging
            preds = logits > 0
            accuracy = (targets == preds).float().mean().item()
            miou, oiou = calculate_miou_oiou(preds, targets)

            total_loss += loss.item()
            total_accuracy += accuracy
            total_miou += miou
            total_oiou += oiou

    loss = total_loss / num_batches

    # write log
    writer.add_scalars(
        "validate",
        {
            "loss": loss,
            "accuracy": total_accuracy / num_batches,
            "miou": total_miou / num_batches,
            "oiou": total_oiou / num_batches,
        },
        global_step,
    )

    return loss


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
        # extract batch data
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["seg_masks"].to(device)

        # forward & loss
        logits = model(pixel_values, input_ids, attention_mask)
        loss = loss_fn(logits, targets.float())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = num_batches * epoch_idx + batch_idx
        # logging
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

        # validation, checkpoint, and early stopping
        if batch_idx % VAL_INTERVAL == 0:
            val_loss = eval_loop(
                val_loader, model, loss_fn, writer, device, global_step
            )

            if val_loss < record.best_val_loss:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
                torch.save(model.state_dict(), checkpoint_path)
                record.best_val_loss = val_loss
                record.no_improve_count = 0
            else:
                record.no_improve_count += 1
                if record.no_improve_count > PATIENCE:
                    print(f"Early stopping at epoch {epoch_idx}, batch {batch_idx}")
                    break
