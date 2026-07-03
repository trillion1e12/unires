from transformers import CLIPVisionModel, CLIPTextModel
from config import (
    CLIP_MODEL_NAME,
    NUM_LOW_TOKENS,
    NUM_HIGH_TOKENS,
    D_MODEL,
    NHEAD,
    DIM_FEEDFORWARD,
    NUM_LAYERS,
)
from model.unires import UniRes


def get_unires_model() -> UniRes:
    clip_vision_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
    clip_text_model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME)
    model = UniRes(
        clip_vision_model,
        clip_text_model,
        NUM_LOW_TOKENS,
        NUM_HIGH_TOKENS,
        D_MODEL,
        NHEAD,
        DIM_FEEDFORWARD,
        NUM_LAYERS,
    )
    return model
