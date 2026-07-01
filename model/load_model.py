from transformers import (
    CLIPVisionModel,
    CLIPTextModel,
)
from model.unires import UniRes

MODEL_NAME = "openai/clip-vit-base-patch32"


def get_unires_model() -> UniRes:
    clip_vision_model = CLIPVisionModel.from_pretrained(MODEL_NAME)
    clip_text_model = CLIPTextModel.from_pretrained(MODEL_NAME)
    model = UniRes(clip_vision_model, clip_text_model)
    return model
