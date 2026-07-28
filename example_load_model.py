import torch
from transformers import CLIPTokenizer

from model.load_model import get_unires_model

MODEL_NAME = "openai/clip-vit-base-patch32"

model = get_unires_model()
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

sample_texts = [
    "i am subaru",
    "as the strongest curse jogoat",
    "This TransformerEncoder layer implements the original architecture",
    "an instance of the TransformerEncoderLayer() class (required)",
    "the number of sub-encoder-layers in the encoder (required).",
    "Pass the input through the encoder layers in turn.",
    "applies a causal mask as mask",
    " Warning: is_causal provides a hint that mask is the causal mask. Providing incorrect hints can result in incorrect execution, including forward and backward compatibility.",
]

clip_inputs = tokenizer(
    sample_texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

print("Tensor shape:")

pixel_values = torch.rand(8, 3, 224, 224)
input_ids = clip_inputs["input_ids"]
attention_mask = clip_inputs["attention_mask"]

seg_mask = model(pixel_values, input_ids, attention_mask)


print(f"pixel_values: {pixel_values.shape}")
print(f"input_ids: {input_ids.shape}")
print(f"attention_mask: {attention_mask.shape}")
print(f"seg_mask: {seg_mask.shape}")
