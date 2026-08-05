import argparse
import os
import subprocess
import sys

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from config.config import (
    CHECKPOINT_DIR,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    SAVE_DATA_PATH,
    SEED,
    TB_LOG_DIR,
    LOG_DIR,
    LOG_FILE,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOG_CONSOLE_LEVEL,
    LOG_FILE_LEVEL,
)
from data.load_data import get_dataloaders
from model.load_model import get_unires_model
from train.train import eval_loop, train_loop
from train.utils import TrainRecord, loss_fn
from utils.logger import setup_logger, get_logger


def main():
    setup_logger(
        log_dir=LOG_DIR,
        log_file=LOG_FILE,
        max_bytes=LOG_MAX_BYTES,
        backup_count=LOG_BACKUP_COUNT,
        console_level=LOG_CONSOLE_LEVEL,
        file_level=LOG_FILE_LEVEL,
    )
    logger = get_logger(__name__)

    logger.info("UniRes training started")
    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Config: max_epochs=%d, lr=%s, patience=%d", MAX_EPOCHS, LEARNING_RATE, PATIENCE)

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading dataloaders from %s", SAVE_DATA_PATH)
    train_loader, val_loader, test_loader = get_dataloaders(SAVE_DATA_PATH)
    logger.info("Dataloaders loaded — train=%d val=%d test=%d batches",
                 len(train_loader), len(val_loader), len(test_loader))

    logger.info("Building UniRes model")
    model = get_unires_model()
    model.to(device)
    logger.info("Model built, parameters: %d", sum(p.numel() for p in model.parameters()))

    optimizer = AdamW(model.parameters(), LEARNING_RATE)
    writer = SummaryWriter(TB_LOG_DIR)
    record = TrainRecord()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    logger.info("Starting training loop (max_epochs=%d, patience=%d)", MAX_EPOCHS, PATIENCE)
    try:
        for epoch in range(MAX_EPOCHS):
            logger.info("Epoch %d/%d starting", epoch, MAX_EPOCHS)
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
                logger.info("Early stopping triggered after epoch %d", epoch)
                break

        logger.info("Training completed. Running final test evaluation.")
        eval_loop(test_loader, model, loss_fn, writer, device, 0, True)
        logger.info("Final test evaluation done.")
    except Exception:
        logger.exception("Training failed with error")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniRes Training")
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Launch real-time log monitor alongside training",
    )
    args = parser.parse_args()

    if args.monitor:
        monitor_script = os.path.join(os.path.dirname(__file__), "monitor.py")
        subprocess.Popen(
            [sys.executable, monitor_script],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    main()
