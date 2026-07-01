# Paper implementation

## Paper

[Unveiling Parts Beyond Objects:Towards Finer-Granularity Referring Expression Segmentation
](https://arxiv.org/abs/2312.08007)

## Data

- Logics for data processing and loading are in `data/`.

    - **Dataset:** RefCOCO from [lmms-lab/RefCOCO](https://huggingface.co/datasets/lmms-lab/RefCOCO)
    - **Process dataset:** `data/process_data.ipynb`, this notebook was ran once to process the raw dataset
    - **Load data:** `data/load_data.py`, have the function `get_dataloaders` for loading the 3 dataloaders (train/val/test).

- Instructions for loading data:

    1. Download and process data to store on disk:

        ```bash
        python data/process_data.py
        ```
    
    2. Confirm that data has been stored at `data/.dataset/` by manually checking that directory.
    3. From your training python script, import `get_dataloaders` from `data/load_data.py` (see `example_load_data.py` for example usage)

        ```python
        from data.load_data import get_dataloaders

        DATA_PATH = "data/.dataset/"
        train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH)
        ```

- Overview shapes of the processed dataloaders, with $B$ as batch size and $S$ as text sequence length:
    - `pixel_values`: $(B, 3, 224, 224)$ - the image tensor
    - `input_ids`: $(B, S)$ - text input id tensor
    - `attention_mask`: $(B, S)$ - text attention mask tensor
    - `seg_masks`: $(B, 1, 224, 224)$ - segmentation mask tensor, with binary value 0 and 1.

- You can run `python example_load_data.py` or read it to see how it work.



## Model

- Logics for model architecture and loading are in `model/`.

    - **Pretrain model:** CLIP from [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
    - **Model architecture:** `model/unires.py`
    - **Load model:** `model/load_model.py`, have the function `get_unires_model` for loading model.

- Instructions for loading model: From your training python script, import `get_unires_model` from `model/load_model.py` (see `example_load_model.py` for example usage)

        ```python
        from model.load_model import get_unires_model

        model = get_unires_model()
        ```

- Overview input and output shapes of model, with $B$ as batch size and $S$ as text sequence length:
    - `pixel_values`: $(B, 3, 224, 224)$ - the image tensor
    - `input_ids`: $(B, S)$ - text input id tensor
    - `attention_mask`: $(B, S)$ - text attention mask tensor
    - `seg_masks`: $(B, 1, 224, 224)$ - segmentation mask logit tensor, with float values.

- You can run `python example_load_model.py` or read it to see how it work.

## Training

...