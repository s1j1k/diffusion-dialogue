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

def merge_and_mask(group_lst, tokenizer, config):
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
    max_length = config["seq_len"]
    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], tokenizer.pad_token_id, max_length)
    group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
    return group_lst

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
