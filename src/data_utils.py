"""
Data utilities for diffusion models text processing.

Authors: 
Group 14
Sally Arnold - 992316
Yun Chu - 1342245
Thet Htut Aung - 940976

Data processing functions adapted from:

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
"""
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
from train_utils import log

def load_data(path, limit=None):
    """
    Load data from a JSON file.

    Parameters:
    path (str): The path to the JSON file.
    limit (int, optional): The maximum number of data points to load. Defaults to None, meaning all data points will be loaded.

    Returns:
    dict: A dictionary containing the source and target sentences.
    """
    sentence_lst = {'src':[], 'trg': []}
    with open(path, 'r') as f_reader:
        for i, row in enumerate(f_reader):
            if limit and i >= limit:
                break
            content = json.loads(row)
            sentence_lst['src'].append(content['src'].strip())
            sentence_lst['trg'].append(content['trg'].strip())
    return sentence_lst

class CustomBertTokenizer():
    """
    Load tokenizer from bert config 
    """
    def __init__(self, config_name='bert-base-uncased'):
        tokenizer = BertTokenizer.from_pretrained(config_name)
        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        tokenizer.save_pretrained('./checkpoints/tokenizer') 
        self.vocab_size = len(self.tokenizer)
    
    def encode_token(self, sentences):
        input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        return input_ids
        
    def decode_token(self, seq):
        seq = seq.squeeze(-1).tolist()
        while len(seq)>0 and seq[-1] == self.pad_token_id:
            seq.pop()
        tokens = self.tokenizer.decode(seq)
        return tokens

def tokenize_function(examples, tokenizer):
    """
    Tokenize the source and target sentences.

    Parameters:
    examples (dict): A dictionary containing the source and target sentences.
    tokenizer (Tokenizer): The tokenizer to use for tokenization.

    Returns:
    dict: A dictionary containing the tokenized input IDs for the source and target sentences.
    """
    input_id_x = tokenizer.encode_token(examples['src'])
    input_id_y = tokenizer.encode_token(examples['trg'])
    result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
    return result_dict

def merge_and_mask(group_lst, tokenizer, config):
    """
    Merge and mask the source and target sentences.

    Parameters:
    group_lst (dict): A dictionary containing the tokenized input IDs for the source and target sentences.
    tokenizer (Tokenizer): The tokenizer to use for tokenization.
    config (dict): The configuration parameters.

    Returns:
    dict: A dictionary containing the merged and masked input IDs for the source and target sentences.
    """
    lst = []
    mask = []
    for i in range(len(group_lst['input_id_x'])):
        end_token = group_lst['input_id_x'][i][-1]
        src = group_lst['input_id_x'][i][:-1]
        trg = group_lst['input_id_y'][i][:-1]
        while len(src) + len(trg) > config["seq_len"] - 3:
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

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    """
    Helper function to collate batches.

    Parameters:
    examples (list): A list of examples.
    pad_token_id (int): The ID of the padding token.
    max_length (int): The maximum length of the examples.
    return_mask (bool, optional): Whether to return a mask. Defaults to False.

    Returns:
    tuple: A tuple containing the collated examples and the mask (if return_mask is True).
    """
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

def pad_function(group_lst, tokenizer, config):
    """
    Pad the input IDs and masks.

    Parameters:
    group_lst (dict): A dictionary containing the merged and masked input IDs for the source and target sentences.
    tokenizer (Tokenizer): The tokenizer to use for tokenization.
    config (dict): The configuration parameters.

    Returns:
    dict: A dictionary containing the padded input IDs and masks.
    """
    max_length = config["seq_len"]
    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], tokenizer.pad_token_id, max_length)
    group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
    return group_lst

class TextDataset(Dataset):
    """
    A custom dataset for text data.
    """
    def __init__(self, text_datasets, split, model_emb=None):
        """
        Initialize the TextDataset.

        Parameters:
        text_datasets (dict): A dictionary containing the tokenized input IDs for the source and target sentences.
        split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        model_emb (nn.Module, optional): The model for embedding the input IDs. Defaults to None.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.text_datasets = text_datasets[split]
        self.length = len(self.text_datasets)
        self.model_emb = model_emb.to(device)

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
        int: The length of the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Parameters:
        idx (int): The index of the item to retrieve.

        Returns:
        tuple: A tuple containing the embedded hidden state and the input IDs and mask.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        with torch.no_grad():
            input_ids = self.text_datasets[idx]['input_ids']
            # FIXME data issues with device
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            # Assuming self.model_emb is already on GPU, you can check its device first
            log.debug("model embedding weights device %s", self.model_emb.weight.device)

            # Ensure that model_emb parameters are on GPU
            for param in self.model_emb.parameters():
                if param.device != device:
                    param.data = param.data.to(device)

            input_ids_tensor = torch.tensor(input_ids)

            hidden_state = self.model_emb(input_ids_tensor)

            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets[idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets[idx]['input_mask'])

            return arr, out_kwargs

def infinite_data_loader(data_loader, device):
    """
    Create an infinite data loader.

    Parameters:
    data_loader (DataLoader): The original data loader.
    device (torch.device): The device to use for loading the data.

    Returns:
    generator: An infinite generator that yields batches of data.
    """
    while True:
        for batch in data_loader:
            batch_data = batch[0].to(device)
            batch_cond = {k: v.to(device) for k, v in batch[1].items()}
            yield batch_data, batch_cond