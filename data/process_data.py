from transformers import CLIPTokenizer, CLIPImageProcessor
from datasets import load_dataset, concatenate_datasets, DatasetDict
from PIL import Image, ImageDraw
from torchvision import transforms
from config.config import CLIP_MODEL_NAME, DATASET_NAME, SAVE_DATA_PATH

print("Downloading dataset and CLIP tokenizer + image processor")

ds = load_dataset(DATASET_NAME)
combined_ds = concatenate_datasets([ds["val"], ds["test"], ds["testA"], ds["testB"]])

tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
to_tensor = transforms.ToTensor()


def process_data(batch):
    images = batch["image"]
    texts = [" . ".join(answers) for answers in batch["answer"]]
    polygons = batch["segmentation"]

    image_inputs = processor(images)
    text_inputs = tokenizer(texts, truncation=True)
    seg_masks = []

    for image, polygon in zip(images, polygons):
        mask_img = Image.new("1", image.size)
        img_draw = ImageDraw.Draw(mask_img)
        vertices = [
            (polygon[i * 2], polygon[i * 2 + 1]) for i in range(len(polygon) // 2)
        ]
        img_draw.polygon(vertices, fill=1)
        mask_img = mask_img.resize((224, 224))
        mask_tensor = to_tensor(mask_img)
        seg_masks.append(mask_tensor.squeeze(0).bool())

    return {
        "pixel_values": image_inputs["pixel_values"],
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "seg_masks": seg_masks,
    }


print("Processing data to tensor")

processed_ds = combined_ds.map(
    process_data,
    batched=True,
    remove_columns=combined_ds.column_names,
)

split_1 = processed_ds.train_test_split(0.2)
split_2 = split_1["test"].train_test_split(0.5)

ds_train = split_1["train"]
ds_val = split_2["train"]
ds_test = split_2["test"]

print(f"Saving data to {SAVE_DATA_PATH}")

ds_dict = DatasetDict({"train": ds_train, "val": ds_val, "test": ds_test})
ds_dict.save_to_disk(SAVE_DATA_PATH)

print("Data has been processed")
