# diffusion-dialogue

## References
DiffuSeq
- refer `train.py`

- used example dataset



## Datasets
Prepare datasets and put them under the `datasets` folder. Take `datasets/CommonsenseConversation/train.jsonl` as an example. We use four datasets in our paper.

| Task | Datasets | Training Samples | Source | Used in __*DiffusionDialogue*__
|-|-|-|-|-|
| Open-domain Dialogue | Commonsense Conversation | 3382k | [CCM](https://github.com/thu-coai/ccm) | [download](https://drive.google.com/drive/folders/1exENF9Qc5UtXnHlNl9fvaxP3zyyH32qp?usp=sharing) |


## Deployment 
Run `git clone`
Place datasets in `/datasets`. 
Run jobs.

## Jobs
Training job: `job_train.slurm` - trains the model and saved transformer state dictionary for later inference.
Training saves a transformer model state dict under `checkpoints/trained_model.pth`. Uses train and eval datasets to perform training and backpropagate losses.

Test job: `job_test.slurm` - runs test using a sample input sentence to generate example inference output.
The output (same as stdout) will be saved in `test_log.txt`. Uses test dataset to generate final scores.

Run using 
```sh
sbatch <job_name>.slurm
```

Monitor status using
```sh
squeue -u <your username>
```

## Components

### Transformer Model

The transformer model used in this project is defined in `transformer_model.py`. It is called within the main script to handle the attention mechanism.

### Diffusion Model

The diffusion model is defined in `diffusion_model.py`. This model is also called within the main script to perform the diffusion operations.

<!--
Use the data_loader.py to load the dataset from google drive
`
from data_loader import prepare_datasets, TRAIN_FILE_ID, VALID_FILE_ID, TEST_FILE_ID
train_data, valid_data, test_data = prepare_datasets(TRAIN_FILE_ID, VALID_FILE_ID, TEST_FILE_ID)
`
-->
