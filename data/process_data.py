from transformers import CLIPTokenizer, CLIPImageProcessor
from datasets import load_dataset, concatenate_datasets, DatasetDict
from PIL import Image, ImageDraw
from torchvision import transforms
from config.config import CLIP_MODEL_NAME, DATASET_NAME, SAVE_DATA_PATH

print("Downloading dataset and CLIP tokenizer + image processor")

ds = load_dataset(DATASET_NAME)

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


available_splits = list(ds.keys())

HAS_TRAIN = "train" in available_splits

if HAS_TRAIN:

    print("Processing data to tensor (official RefCOCO split: train)")

    val_name = "val" if "val" in available_splits else "validation"
    ds_train = ds["train"].map(
        process_data, batched=True, remove_columns=ds["train"].column_names
    )
    ds_val = ds[val_name].map(
        process_data, batched=True, remove_columns=ds[val_name].column_names
    )

    test_splits = [s for s in ["testA", "testB"] if s in available_splits]
    if test_splits:
        ds_test = concatenate_datasets([
            ds[s].map(process_data, batched=True, remove_columns=ds[s].column_names)
            for s in test_splits
        ])
    else:
        ds_test = ds["test"].map(
            process_data, batched=True, remove_columns=ds["test"].column_names
        )

    train_count = len(ds_train)
    val_count = len(ds_val)
    test_count = len(ds_test)

else:

    print(
        "WARNING: 'train' split not found in dataset %s. "
        "Falling back to combining eval splits (val, test, testA, testB) "
        "and randomly splitting 80/10/10. "
        "For proper evaluation, use a dataset with the official train split "
        "(e.g., jxu124/refcoco with COCO images)." % DATASET_NAME
    )

    eval_splits = [s for s in ["val", "test", "testA", "testB"] if s in available_splits]
    combined_ds = concatenate_datasets([
        ds[s].map(process_data, batched=True, remove_columns=ds[s].column_names)
        for s in eval_splits
    ])

    split_1 = combined_ds.train_test_split(0.2, seed=42)
    split_2 = split_1["test"].train_test_split(0.5, seed=42)
    ds_train = split_1["train"]
    ds_val = split_2["train"]
    ds_test = split_2["test"]

    train_count = len(ds_train)
    val_count = len(ds_val)
    test_count = len(ds_test)

print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
print(f"Saving data to {SAVE_DATA_PATH}")

ds_dict = DatasetDict({"train": ds_train, "val": ds_val, "test": ds_test})
ds_dict.save_to_disk(SAVE_DATA_PATH)

print("Data has been processed")
