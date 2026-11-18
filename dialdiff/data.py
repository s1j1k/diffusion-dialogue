"""Dialogue pairs to fixed-length token sequences.

Each example becomes  [CLS] source [SEP] target [SEP] [PAD]...  of length
seq_len, with a mask that is 1 over the part the model generates (target and
padding) and 0 over the conditioning part ([CLS] source [SEP]). Padding is
generated too, which is how the model chooses the response length.
"""
import json

import torch
from torch.utils.data import Dataset


def load_pairs(path, limit=None):
    """Read a jsonl file of {"src": ..., "trg": ...} records."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(pairs) >= limit:
                break
            row = json.loads(line)
            pairs.append((row["src"].strip(), row["trg"].strip()))
    return pairs


def encode_pair(src_ids, trg_ids, tokenizer, seq_len):
    """Pack token id lists into (ids, generate_mask), truncating the longer side first."""
    src, trg = list(src_ids), list(trg_ids)
    budget = seq_len - 3  # [CLS], [SEP], [SEP]
    while len(src) + len(trg) > budget:
        (src if len(src) >= len(trg) else trg).pop()

    ids = [tokenizer.cls_token_id] + src + [tokenizer.sep_token_id] + trg + [tokenizer.sep_token_id]
    n_cond = len(src) + 2
    ids += [tokenizer.pad_token_id] * (seq_len - len(ids))
    mask = [0] * n_cond + [1] * (seq_len - n_cond)
    return ids, mask


class DialogueDataset(Dataset):
    def __init__(self, pairs, tokenizer, seq_len):
        self.pairs = pairs
        srcs = tokenizer([s for s, _ in pairs], add_special_tokens=False)["input_ids"]
        trgs = tokenizer([t for _, t in pairs], add_special_tokens=False)["input_ids"]
        encoded = [encode_pair(s, t, tokenizer, seq_len) for s, t in zip(srcs, trgs)]
        self.ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
        self.mask = torch.tensor([e[1] for e in encoded], dtype=torch.bool)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"ids": self.ids[i], "mask": self.mask[i]}
