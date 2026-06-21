from data.load_data import get_dataloaders

DATA_PATH = "data/.dataset/"
train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH)

for batch in train_loader:
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
    break
