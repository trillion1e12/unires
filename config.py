import os
from dotenv import load_dotenv

load_dotenv()

CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME")
DATASET_NAME: str = os.getenv("DATASET_NAME")
SAVE_DATA_PATH: str = os.getenv("SAVE_DATA_PATH")

# dataloader
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS"))

# model
NUM_LOW_TOKENS = int(os.getenv("NUM_LOW_TOKENS"))
NUM_HIGH_TOKENS = int(os.getenv("NUM_HIGH_TOKENS"))
D_MODEL = int(os.getenv("D_MODEL"))
NHEAD = int(os.getenv("NHEAD"))
DIM_FEEDFORWARD = int(os.getenv("DIM_FEEDFORWARD"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS"))

# training
LOG_DIR: str = os.getenv("LOG_DIR")
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL"))
SEED = int(os.getenv("SEED"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS"))
PATIENCE = int(os.getenv("PATIENCE"))
DICE_SMOOTH = float(os.getenv("DICE_SMOOTH"))
IOU_EPSILON = float(os.getenv("IOU_EPSILON"))
