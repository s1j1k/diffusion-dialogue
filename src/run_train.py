import torch 
from transformers import BertTokenizer
import json
from torch.utils.data import DataLoader, RandomSampler #Dataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from functools import partial
import logging as log
# from torch.utils.data import Dataset
import numpy as np
import torch

# Custom classes
from train_utils import TrainLoop
from transformer_model import TransformerNetModel
from diffusion_model import GaussianDiffusion
from data_utils import load_data, tokenize_function, merge_and_mask, pad_function, TextDataset, infinite_data_loader

# TODO save a trained model for inference in future
# TODO save hyper paramters in a script config file
# TODO better logging 
# TODO separate the inference stage (test script)

# TODO make it more different to the below source
# -> different data set, check result 

# TODO run evaluation & generate plots as a separate file

'''

Main training file for simplified sequence-to-sequence text generation using diffusion models.

Authors: 
Group 14
Sally Arnold - 992316
Yun Chu - 1342245
Thet Htut Aung - 940976

Adapted from:

@inproceedings{gong2022diffuseq,
  author = {Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  booktitle = {International Conference on Learning Representations, ICLR},
  title = {{DiffuSeq}: Sequence to Sequence Text Generation with Diffusion Models},
  year = 2023
}

@article{gong2023diffuseqv2,
  title={DiffuSeq-v2: Bridging Discrete and Continuous Text Spaces for Accelerated Seq2Seq Diffusion Models},
  author={Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2310.05793},
  year={2023}
}

'''

# Basic logger to log to file and to stdout
log.basicConfig(filename='logs.log', level=log.DEBUG, format="%(asctime)s:%(levelname)s: %(message)s")
log.getLogger().addHandler(log.StreamHandler())

# Suppress matplotlib font-related messages
log.getLogger('matplotlib').setLevel(log.WARNING)

# FIXME check if we should have infinite data loader for training?

# # Set training parameters, mainly using default values from DiffuSeq
config = {
    "embedding_dim": 128, # embedding dimension
    "hidden_dim": 128, # hidden dimension
    "seq_len": 128, # maximum sequence length
    "output_dims": 128, # output dimension
    "num_diffusion_timesteps": 2000, # number of diffusion timesteps
    "batch_size": 32, # batch size
    "lr": 0.001, # learning rate
    "ema_rate": 0.999, # exponential moving average rate
    "weight_decay": 0.01, # weight decay
    # NOTE was 100
    "learning_steps": 500, # total steps of learning # NOTE this was a very small number, check
    "eval_interval": 1 # total steps of learning
}

# Main function
def main():
    # Use GPU if available
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        log.info("GPU is available")
    else:
        device = torch.device("cpu")
        log.info("GPU not available, CPU used")

    # Print training config to file
    with open('training_config.json', 'w') as fp:
        json.dump(config, fp)
        log.info("Training config saved to file.")

    # Get tokenizer from BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    log.info("Vocab size", vocab_size)

    # Initialize an embedding layer for the tokenizer's vocabulary with the chosen embedding dimension
    model_emb = torch.nn.Embedding(tokenizer.vocab_size, config['embedding_dim'])

    # Initialize random embeddings
    torch.nn.init.normal_(model_emb.weight)
    log.info("Embedding layer", model_emb)

    # Dataset path definition
    data_dir = "./datasets/CommonsenseConversation"
    train_path = f'{data_dir}/train_full.jsonl'
    valid_path = f'{data_dir}/valid_full.jsonl'

    # Load datasets with size restriction
    train_limit = 1000  # Limit the size of the training set
    valid_limit = 200   # Limit the size of the validation set

    train_data = load_data(train_path, limit=train_limit)
    valid_data = load_data(valid_path, limit=valid_limit)

    log.debug("Training Data Samples:", len(train_data['src']))
    log.debug("Validation Data Samples:", len(valid_data['src']))

    log.debug(train_data['src'][0])
    raw_datasets = HFDataset.from_dict(train_data)
    log.debug(raw_datasets)
    log.debug(raw_datasets[0])

    # Tokenize dataset
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    print("Vocabulary Size:", vocab_size)

    # Use partial to pass the tokenizer to the tokenize_function
    tokenize_function_with_tokenizer = partial(tokenize_function, tokenizer=tokenizer)

    # Create datasets
    train_dataset = HFDataset.from_dict(train_data)
    valid_dataset = HFDataset.from_dict(valid_data)

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(
        tokenize_function_with_tokenizer,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Tokenizing training dataset",
    )

    tokenized_valid_dataset = valid_dataset.map(
        tokenize_function_with_tokenizer,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Tokenizing validation dataset",
    )

    # Combine into DatasetDict
    tokenized_datasets = DatasetDict({
        'train': tokenized_train_dataset,
        'validation': tokenized_valid_dataset,
    })

    log.info("Tokenization complete.")
    log.debug("Training Set:", len(tokenized_datasets['train']))
    log.debug("Validation Set:", len(tokenized_datasets['validation']))

    # Apply merge and mask to the tokenized datasets
    tokenized_datasets = DatasetDict({
        'train': tokenized_train_dataset,
        'validation': tokenized_valid_dataset,
    })

    # Use partial to pass the tokenizer to merge_and_mask
    merge_and_mask_with_tokenizer = partial(merge_and_mask, tokenizer=tokenizer, config=config)
    pad_function_with_tokenizer = partial(pad_function, tokenizer=tokenizer, config=config)

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Merging and masking"
    )

    # Apply padding to the datasets
    lm_datasets = tokenized_datasets.map(
        pad_function_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Padding"
    )

    log.info("Merging, masking, and padding complete.")
    log.debug("Training Set:", len(lm_datasets['train']))
    log.debug("Validation Set:", len(lm_datasets['validation']))
    log.debug('Padded Dataset:', lm_datasets)

    # Create datasets for training, validation, and test sets
    train_dataset = TextDataset(lm_datasets, 'train', model_emb=model_emb)
    valid_dataset = TextDataset(lm_datasets, 'validation', model_emb=model_emb)

    # Create data loaders with RandomSampler
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=RandomSampler(train_dataset))
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], sampler=RandomSampler(valid_dataset))

    # Convert the data loaders to infinite data loaders
    train_loader_infinite = infinite_data_loader(train_loader, device)  # Ensure train_loader returns batches on the correct device
    valid_loader_infinite = infinite_data_loader(valid_loader, device)  # Ensure valid_loader returns batches on the correct device

    # Sample data from the infinite data loaders
    train_data_iter = iter(train_loader_infinite)
    valid_data_iter = iter(valid_loader_infinite)

    log.debug("Sample from train dataset:", next(train_data_iter))
    log.debug("Sample from validation dataset:", next(valid_data_iter))

    # Define the noise schedule
    scale = 1000 / config['num_diffusion_timesteps']
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, config['num_diffusion_timesteps'], dtype=np.float64)

    # Instantiate the diffusion model & transformer
    diffusion = GaussianDiffusion(betas=betas)
    model = TransformerNetModel(vocab_size=vocab_size, input_dims=config['embedding_dim'], hidden_t_dim=config['hidden_dim'], output_dims=config['output_dims']).to(device)
    
    # Save the transformer model state dict for inference stage
    torch.save(model.state_dict(), 'saved_model_state_dict.pth')

    # Log the total number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    log.info(f'The parameter count is {pytorch_total_params}')

    train_loader = infinite_data_loader(train_loader, device)  # Ensure train_loader returns batches on the correct device
    valid_loader = infinite_data_loader(valid_loader, device)  # Ensure valid_loader returns batches on the correct device

    TrainLoop(
    model=model.to(device),
    diffusion=diffusion,
    data=iter(train_loader),
    batch_size=config['batch_size'],
    lr=config['lr'],
    ema_rate=config['ema_rate'],
    weight_decay=config['weight_decay'],
    learning_steps=config['learning_steps'],
    eval_data=iter(valid_loader),
    eval_interval=config['eval_interval'],
    device=device,
    ).run_loop()

if __name__ == "__main__":
    main()
    
