from torch.utils.data import DataLoader
from typing import Tuple
from transformers import DataCollatorWithPadding, CLIPTokenizer
from datasets import load_from_disk

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
NUM_WORKERS = 2


def get_dataloaders(path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds = load_from_disk(path)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
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
