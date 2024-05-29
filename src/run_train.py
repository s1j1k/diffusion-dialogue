import torch 
import torch as th
from transformers import BertTokenizer
import json
import datasets
from torch.utils.data import DataLoader, RandomSampler #Dataset
from datasets import Dataset, DatasetDict
from functools import partial
import logging as log
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
import numpy as np
import os
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm


# Custom classes
from train_utils import TrainLoop
from transformer_model import TransformerNetModel
from diffusion_model import GaussianDiffusion

# TODO save a trained model for inference in future
# TODO save hyper paramters in a script config file
# TODO better logging 
# TODO separate the inference stage (test script)

# TODO make it more different to the below source
# -> different data set, check result 

'''

Main training file for simplified sequence-to-sequence text generation using diffusion models.

Authors: 
Group 14
Yun Chu - 1342245
Thet Htut Aung - 940976
Sally Arnold - 992316

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

# FIXME check if we should have infinite data loader for training?

# Set training parameters, mainly using default values from DiffuSeq
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
    "learning_steps": 40000, # total steps of learning # NOTE this was a very small number, check
    "eval_interval": 1 # total steps of learning
}

# Helper functions
# Load data from json file
def load_data(path, limit=None):
    sentence_lst = {'src':[], 'trg': []}
    with open(path, 'r') as f_reader:
        for i, row in enumerate(f_reader):
            if limit and i >= limit:
                break
            content = json.loads(row)
            sentence_lst['src'].append(content['src'].strip())
            sentence_lst['trg'].append(content['trg'].strip())
    return sentence_lst

def tokenize_function(examples, tokenizer):
    input_id_x = tokenizer(examples['src'], add_special_tokens=True)['input_ids']
    input_id_y = tokenizer(examples['trg'], add_special_tokens=True)['input_ids']
    result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
    return result_dict

# Function to merge and mask sequences
def merge_and_mask(group_lst, tokenizer):
    lst = []
    mask = []
    for i in range(len(group_lst['input_id_x'])):
        end_token = group_lst['input_id_x'][i][-1]
        src = group_lst['input_id_x'][i][:-1]
        trg = group_lst['input_id_y'][i][:-1]
        while len(src) + len(trg) > config.seq_len - 3:
            if len(src) > len(trg):
                src.pop()
            elif len(src) < len(trg):
                trg.pop()
            else:
                src.pop()
                trg.pop()
        src.append(end_token)
        trg.append(end_token)

        lst.append(src + [tokenizer.sep_token_id] + trg)
        mask.append([0] * (len(src) + 1))
    group_lst['input_ids'] = lst
    group_lst['input_mask'] = mask
    return group_lst

# Function to pad sequences
def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

def pad_function(group_lst, tokenizer):
    max_length = config.seq_len
    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], tokenizer.pad_token_id, max_length)
    group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
    return group_lst

# Helper to parse text data sets 
class TextDataset(Dataset):
    def __init__(self, text_datasets, split, model_emb=None):
        self.text_datasets = text_datasets[split]
        self.length = len(self.text_datasets)
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            input_ids = self.text_datasets[idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids))

            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets[idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets[idx]['input_mask'])

            return arr, out_kwargs

def infinite_data_loader(data_loader, device):
    while True:
        for batch in data_loader:
            batch_data = batch[0].to(device)
            batch_cond = {k: v.to(device) for k, v in batch[1].items()}
            yield batch_data, batch_cond



    


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
    model_emb = torch.nn.Embedding(tokenizer.vocab_size, config.embedding_dim)

    # Initialize random embeddings
    torch.nn.init.normal_(model_emb.weight)
    log.info("Embedding layer", model_emb)

    # Dataset path definition
    data_dir = "./datasets/CommonsenseConversation"
    train_path = f'{data_dir}/train_full.jsonl'
    valid_path = f'{data_dir}/valid_full.jsonl'
    test_path = f'{data_dir}/test_full.jsonl'

    # Load datasets with size restriction
    train_limit = 1000  # Limit the size of the training set
    valid_limit = 200   # Limit the size of the validation set
    test_limit = 200    # Limit the size of the test set

    train_data = load_data(train_path, limit=train_limit)
    valid_data = load_data(valid_path, limit=valid_limit)
    test_data = load_data(test_path, limit=test_limit)

    log.debug("Training Data Samples:", len(train_data['src']))
    log.debug("Validation Data Samples:", len(valid_data['src']))
    log.debug("Test Data Samples:", len(test_data['src']))

    log.debug(train_data['src'][0])
    raw_datasets = Dataset.from_dict(train_data)
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
    train_dataset = Dataset.from_dict(train_data)
    valid_dataset = Dataset.from_dict(valid_data)
    test_dataset = Dataset.from_dict(test_data)

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

    tokenized_test_dataset = test_dataset.map(
        tokenize_function_with_tokenizer,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Tokenizing test dataset",
    )

    # Combine into DatasetDict
    tokenized_datasets = DatasetDict({
        'train': tokenized_train_dataset,
        'validation': tokenized_valid_dataset,
        'test': tokenized_test_dataset
    })

    log.info("Tokenization complete.")
    log.debug("Training Set:", len(tokenized_datasets['train']))
    log.debug("Validation Set:", len(tokenized_datasets['validation']))
    log.debug("Test Set:", len(tokenized_datasets['test']))

    # Apply merge and mask to the tokenized datasets
    tokenized_datasets = DatasetDict({
        'train': tokenized_train_dataset,
        'validation': tokenized_valid_dataset,
        'test': tokenized_test_dataset
    })

    tokenized_datasets = tokenized_datasets.map(
    merge_and_mask,
    batched=True,
    num_proc=1,
    tokenizer=tokenizer,
    desc="Merging and masking"
    )

    # Apply padding to the datasets
    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        tokenizer=tokenizer,
        desc="Padding"
    )

    log.info("Merging, masking, and padding complete.")
    log.debug("Training Set:", len(lm_datasets['train']))
    log.debug("Validation Set:", len(lm_datasets['validation']))
    log.debug("Test Set:", len(lm_datasets['test']))
    log.debug('Padded Dataset:', lm_datasets)

    # Create datasets for training, validation, and test sets
    train_dataset = TextDataset(lm_datasets, 'train', model_emb=model_emb)
    valid_dataset = TextDataset(lm_datasets, 'validation', model_emb=model_emb)
    test_dataset = TextDataset(lm_datasets, 'test', model_emb=model_emb)

    # Create data loaders with RandomSampler
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=RandomSampler(train_dataset))
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, sampler=RandomSampler(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=RandomSampler(test_dataset))

    # Convert the data loaders to infinite data loaders
    train_loader_infinite = infinite_data_loader(train_loader, device)  # Ensure train_loader returns batches on the correct device
    valid_loader_infinite = infinite_data_loader(valid_loader, device)  # Ensure valid_loader returns batches on the correct device

    # Sample data from the infinite data loaders
    train_data_iter = iter(train_loader_infinite)
    valid_data_iter = iter(valid_loader_infinite)
    test_data_iter = iter(test_loader)  # Test data is usually not infinite

    log.debug("Sample from train dataset:", next(train_data_iter))
    log.debug("Sample from validation dataset:", next(valid_data_iter))
    log.debug("Sample from test dataset:", next(test_data_iter))

    # Define the noise schedule
    scale = 1000 / config.num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, config.num_diffusion_timesteps, dtype=np.float64)

    # Instantiate the diffusion model & transformer
    diffusion = GaussianDiffusion(betas=betas)
    model = TransformerNetModel(vocab_size=vocab_size, input_dims=config.embedding_dim, hidden_t_dim=config.hidden_dim, output_dims=config.output_dims).to(device)
    
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
    batch_size=config.batch_size,
    lr=config.lr,
    ema_rate=config.ema_rate,
    weight_decay=config.weight_decay,
    learning_steps=config.learning_steps,
    eval_data=iter(valid_loader),
    eval_interval=config.eval_interval,
    device=device,
    ).run_loop()

    # TODO delete the below since it should be in run_test.py
    # TODO check the inference stage is all correct
    # TODO add comments everywhere else and clean up
    # Initialize the model
    # TODO how to save the vocab_size
    model = TransformerNetModel(
        vocab_size=vocab_size, 
        input_dims=config.embedding_dim, 
        hidden_t_dim=config.hidden_dim, 
        output_dims=config.output_dims
    )
    model.to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load('saved_model_state_dict.pth'))
    model.eval()

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def temperature_sampling(logits, temperature):
        logits = logits / temperature
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    def generate_text(model, tokenizer, prompt, max_length=128, num_timesteps=2000, temperature=1.0):
        model.eval()
        with torch.no_grad():
            # Tokenize the input prompt
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # Get the embeddings for the input prompt
            input_embeds = model.word_embedding(input_ids).to(device)

            # Initialize noise
            noise = torch.randn_like(input_embeds).to(device)

            # Set up the diffusion process
            diffusion = GaussianDiffusion(betas=np.linspace(1e-4, 0.02, num_timesteps))

            # Sample from the model using p_sample_loop
            samples = diffusion.p_sample_loop(
                model=model,
                shape=input_embeds.shape,
                noise=noise,
                device=device,
                progress=True,
                clamp_step=None  # Set this to a specific value if needed
            )

            # Convert the generated embeddings back to tokens
            generated_ids = model.lm_head(samples[-1].to(device))

            # Apply temperature sampling
            generated_ids = generated_ids.view(-1, generated_ids.size(-1))  # Reshape to (batch_size * seq_len, vocab_size)
            sampled_ids = temperature_sampling(generated_ids, temperature)
            sampled_ids = sampled_ids.view(1, -1)  # Reshape back to (1, seq_len)
            
            generated_text = tokenizer.decode(sampled_ids.squeeze().tolist(), skip_special_tokens=True)
            
            # Print input IDs and output IDs
            print(f"Input IDs: {input_ids}")
            print(f"Output IDs: {sampled_ids}")

            return generated_text

    # Example usage
    # taking first sample from raw test data
    # {"src": "this fucking shit pisses me off to no end , when these fucking liberal hypocrites imply the only group of people capable of being racist are the whites .", "trg": "as a brown guy the most racist people i 've met are not white . they 're the self proclaimed liberals ."}
    prompt = "this fucking shit pisses me off to no end , when these fucking liberal hypocrites imply the only group of people capable of being racist are the whites ."
    generated_response = generate_text(model, tokenizer, prompt, temperature=0.7)
    # NOTE what is temperature
    print(f"Prompt: {prompt}")
    print(f"Generated Response: {generated_response}")


if __name__ == "__main__":
    main()
    
