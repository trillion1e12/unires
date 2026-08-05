import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# clip
CLIP_MODEL_NAME: str = config["clip"]["model_name"]

# dataset
DATASET_NAME: str = config["dataset"]["name"]
SAVE_DATA_PATH: str = config["dataset"]["save_data_path"]

# dataloader
BATCH_SIZE: int = config["dataloader"]["batch_size"]
NUM_WORKERS: int = config["dataloader"]["num_workers"]

# model
NUM_LOW_TOKENS: int = config["model"]["num_low_tokens"]
NUM_HIGH_TOKENS: int = config["model"]["num_high_tokens"]
D_MODEL: int = config["model"]["d_model"]
NHEAD: int = config["model"]["nhead"]
DIM_FEEDFORWARD: int = config["model"]["dim_feedforward"]
NUM_LAYERS: int = config["model"]["num_layers"]

# training
TB_LOG_DIR: str = config["training"]["log_dir"]
CHECKPOINT_DIR: str = config["training"]["checkpoint_dir"]
LOG_INTERVAL: int = config["training"]["log_interval"]
VAL_INTERVAL: int = config["training"]["val_interval"]
SEED: int = config["training"]["seed"]
LEARNING_RATE = float(config["training"]["learning_rate"])
MAX_EPOCHS: int = config["training"]["max_epochs"]
PATIENCE: int = config["training"]["patience"]
DICE_SMOOTH = float(config["training"]["dice_smooth"])
IOU_EPSILON = float(config["training"]["iou_epsilon"])
FREEZE_CLIP: bool = config["training"]["freeze_clip"]

# logging
LOG_DIR: str = config["logging"]["log_dir"]
LOG_FILE: str = config["logging"]["log_file"]
LOG_MAX_BYTES: int = config["logging"]["max_bytes"]
LOG_BACKUP_COUNT: int = config["logging"]["backup_count"]
LOG_CONSOLE_LEVEL: str = config["logging"]["console_level"]
LOG_FILE_LEVEL: str = config["logging"]["file_level"]
