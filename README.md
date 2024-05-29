# diffusion-dialogue

## References
DiffuSeq
- refer `train.py`

- used example dataset

Use the data_loader.py to load the dataset from google drive
`
from data_loader import prepare_datasets, TRAIN_FILE_ID, VALID_FILE_ID, TEST_FILE_ID
train_data, valid_data, test_data = prepare_datasets(TRAIN_FILE_ID, VALID_FILE_ID, TEST_FILE_ID)
`


## Datasets
Prepare datasets and put them under the `datasets` folder. Take `datasets/CommonsenseConversation/train.jsonl` as an example. We use four datasets in our paper.

| Task | Datasets | Training Samples | Source | Used in __*DiffusionDialogue*__
|-|-|-|-|-|
| Open-domain Dialogue | Commonsense Conversation | 3382k | [CCM](https://github.com/thu-coai/ccm) | [download](https://drive.google.com/drive/folders/1exENF9Qc5UtXnHlNl9fvaxP3zyyH32qp?usp=sharing) |


<!-- TODO consider using 2nd dataset as discussed in the video -->

## Main Script

The main script for this project is the Jupyter notebook: `simple_diffusion_gpu.ipynb`.

## Components

### Transformer Model

The transformer model used in this project is defined in `transformer_model.py`. It is called within the main script to handle the attention mechanism.

### Diffusion Model

The diffusion model is defined in `diffusion_model.py`. This model is also called within the main script to perform the diffusion operations.


