import os

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from config import (
    CHECKPOINT_DIR,
    LEARNING_RATE,
    LOG_DIR,
    MAX_EPOCHS,
    PATIENCE,
    SAVE_DATA_PATH,
    SEED,
)
from data.load_data import get_dataloaders
from model.load_model import get_unires_model
from train.train import train_loop
from train.utils import TrainRecord, loss_fn


def main():
    # set up model and dataloader
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = get_dataloaders(SAVE_DATA_PATH)
    model = get_unires_model()
    model.to(device)
    optimizer = AdamW(model.parameters(), LEARNING_RATE)
    writer = SummaryWriter(LOG_DIR)
    record = TrainRecord()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # training
    for epoch in range(MAX_EPOCHS):
        train_loop(
            train_loader,
            val_loader,
            model,
            optimizer,
            loss_fn,
            writer,
            device,
            epoch,
            record,
        )

        if record.no_improve_count > PATIENCE:
            break


if __name__ == "__main__":
    main()
