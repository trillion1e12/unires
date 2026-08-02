from typing import Tuple

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, DataCollatorWithPadding

from config.config import BATCH_SIZE, CLIP_MODEL_NAME, NUM_WORKERS


def get_dataloaders(path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds = load_from_disk(path)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        ds["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        ds["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    return train_loader, val_loader, test_loader
