import torch 
import json
from torch.utils.data import DataLoader, RandomSampler #Dataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from functools import partial
import logging as log
# from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import BertModel

# Custom classes
from train_utils import TrainLoop, CustomLogger
from transformer_model import TransformerNetModel
from diffusion_model import GaussianDiffusion
from data_utils import load_data, tokenize_function, merge_and_mask, pad_function, TextDataset, infinite_data_loader, CustomBertTokenizer


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

# Import logger
log = CustomLogger().get_logger()

# # Set training parameters, mainly using default values from DiffuSeq
config = {
    # Note that the word embedding dimension is fixed at 768 by the choice of BERT pre traind model
    "embedding_dim": 768, # embedding dimension, as default by BERT base model
    # TODO we expect input dim == embedding dim, is that true?
    "hidden_t_dim": 128, # hidden time embedding dimension
    "seq_len": 128, # maximum sequence length
    "output_dims": 128, # output dimension
    "num_diffusion_timesteps": 2000, # number of diffusion timesteps
    "batch_size": 32, # batch size
    "lr": 0.001, # learning rate
    "ema_rate": 0.999, # exponential moving average rate
    "weight_decay": 0.01, # weight decay
    # NOTE was 100
    # NOTE learning steps should be as high as possible (40k for DiffuSeq)
    # NOTE should be greater than num diffusion timesteps
    "learning_steps": 4000, # total steps of learning # NOTE this was a very small number, check
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
    with open('./src/training_config.json', 'w') as fp:
        json.dump(config, fp)
        log.info("Training config saved to file.")

    # Get tokenizer from BERT, will be reloaded later
    tokenizer = CustomBertTokenizer()
    vocab_size = tokenizer.vocab_size
    log.info("Vocab size %s", vocab_size)

    
    # Create a BertModel from pretrained
    # Has hidden dropout prob = 0.1 (default)
    # Has hidden_size 768 (embedding dimension)
    temp_bert = BertModel.from_pretrained('bert-base-uncased')
    temp_bert.save_pretrained("checkpoints/temp_bert")

    # Create word embeddings layer
    model_emb = temp_bert.embeddings.word_embeddings
    log.debug("Embedding model {}", model_emb)

    # Dataset path definition - Note relative to the /diffusion-dialogue level
    data_dir = "./datasets/CommonsenseConversation"
    train_path = f'{data_dir}/train.jsonl'
    valid_path = f'{data_dir}/valid.jsonl'

    # Load datasets with size restriction
    train_limit = 1000  # Limit the size of the training set
    valid_limit = 200   # Limit the size of the validation set

    train_data = load_data(train_path, limit=train_limit)
    valid_data = load_data(valid_path, limit=valid_limit)

    log.debug("Training Data Samples: %d", len(train_data['src']))
    log.debug("Validation Data Samples: %d", len(valid_data['src']))

    log.debug(train_data['src'][0])
    raw_datasets = HFDataset.from_dict(train_data)
    log.debug(raw_datasets)
    log.debug(raw_datasets[0])

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
    log.debug("Training Set: %d", len(tokenized_datasets['train']))
    log.debug("Validation Set: %d", len(tokenized_datasets['validation']))

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
    log.debug("Training Set: %d", len(lm_datasets['train']))
    log.debug("Validation Set: %d", len(lm_datasets['validation']))
    log.debug('Padded Dataset: %s', lm_datasets)

    # Check input size
    log.debug("Training Set shape input_id_x: %s, input_id_y: %s, input_ids: %s, input_mask: %s",
               lm_datasets['train']["input_id_x"].shape, lm_datasets['train']["input_id_y"].shape,
               lm_datasets['train']["input_ids"].shape, lm_datasets['train']["input_mask"].shape)

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

    log.debug("Sample from train dataset: %s", next(train_data_iter))
    log.debug("Sample from validation dataset: %s", next(valid_data_iter))

    # Define the noise schedule
    scale = 1000 / config['num_diffusion_timesteps']
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, config['num_diffusion_timesteps'], dtype=np.float64)

    # Instantiate the diffusion model & transformer
    diffusion = GaussianDiffusion(betas=betas)
    model = TransformerNetModel(vocab_size=vocab_size, input_dims=config['embedding_dim'], hidden_t_dim=config['hidden_t_dim'], output_dims=config['output_dims']).to(device)

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
    
