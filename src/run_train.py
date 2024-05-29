import torch 
from transformers import BertTokenizer
import numpy as np
import torch
from transformers import BertTokenizer
from datasets import Dataset
import json
import datasets
from torch.utils.data import DataLoader, RandomSampler #Dataset
import torch as th
from datasets import Dataset, DatasetDict
from functools import partial
from transformer_model import TransformerNetModel
from diffusion_model import GaussianDiffusion

'''
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

# SET PARAMS
# TODO use a script to set params using the arg parser
embedding_dim = 128    # choose embedding dimension = 128
hidden_dim = 128       # hidden size of time embedding
seq_len = 128          # :param seq_len: the max sequence length (one-side).
output_dims = 128      # TODO good value for this
num_diffusion_timesteps = 2000 # Same as diffuSeq

batch_size = 32
lr = 0.001             # learning rate
ema_rate = 0.999
weight_decay = 0.01
learning_steps = 2000  # Adjusting the learning steps: larger = train longer
eval_interval = 1      # Validation interval: Smaller = More frequent


# Use GPU if available
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    

# Get tokenizer from BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size

# TODO check that logging is working properly
# TODO implement logging
print("LOG: VOCAB SIZE: ", vocab_size)

# Initialize an embedding layer for the tokenizer's vocabulary with the chosen embedding dimension
model_emb = torch.nn.Embedding(tokenizer.vocab_size, embedding_dim)

# Initialize random embeddings
torch.nn.init.normal_(model_emb.weight)

print("LOG: MODEL EMB", model_emb)

# Dataset path definition
data_dir = "./datasets/CommonsenseConversation"
train_path = f'{data_dir}/train_full.jsonl'
valid_path = f'{data_dir}/valid_full.jsonl'
test_path = f'{data_dir}/test_full.jsonl'

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

# Load datasets with size restriction
train_limit = 1000  # Limit the size of the training set
valid_limit = 200   # Limit the size of the validation set
test_limit = 200    # Limit the size of the test set

train_data = load_data(train_path, limit=train_limit)
valid_data = load_data(valid_path, limit=valid_limit)
test_data = load_data(test_path, limit=test_limit)

print("Training Data Samples:", len(train_data['src']))
print("Validation Data Samples:", len(valid_data['src']))
print("Test Data Samples:", len(test_data['src']))

train_data['src'][0]
raw_datasets = Dataset.from_dict(train_data)
raw_datasets
raw_datasets[0]

# Tokenize dataset
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size
print("Vocabulary Size:", vocab_size)

def tokenize_function(examples, tokenizer):
    input_id_x = tokenizer(examples['src'], add_special_tokens=True)['input_ids']
    input_id_y = tokenizer(examples['trg'], add_special_tokens=True)['input_ids']
    result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
    return result_dict

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

print("Tokenization complete.")
print("Training Set:", len(tokenized_datasets['train']))
print("Validation Set:", len(tokenized_datasets['validation']))
print("Test Set:", len(tokenized_datasets['test']))

# tokenized_datasets
# len(tokenized_datasets['train']["input_id_x"][0])


# Function to merge and mask sequences
def merge_and_mask(group_lst):
    lst = []
    mask = []
    for i in range(len(group_lst['input_id_x'])):
        end_token = group_lst['input_id_x'][i][-1]
        src = group_lst['input_id_x'][i][:-1]
        trg = group_lst['input_id_y'][i][:-1]
        while len(src) + len(trg) > seq_len - 3:
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

def pad_function(group_lst):
    max_length = seq_len
    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], tokenizer.pad_token_id, max_length)
    group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
    return group_lst

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
    desc="Merging and masking"
)

# Apply padding to the datasets
lm_datasets = tokenized_datasets.map(
    pad_function,
    batched=True,
    num_proc=1,
    desc="Padding"
)

print("Merging, masking, and padding complete.")
print("Training Set:", len(lm_datasets['train']))
print("Validation Set:", len(lm_datasets['validation']))
print("Test Set:", len(lm_datasets['test']))

print(lm_datasets, 'padded dataset')

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np

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

# Define model embedding
model_emb = lambda x: x  # Placeholder: Replace with actual model embedding function

# Create datasets for training, validation, and test sets
train_dataset = TextDataset(lm_datasets, 'train', model_emb=model_emb)
valid_dataset = TextDataset(lm_datasets, 'validation', model_emb=model_emb)
test_dataset = TextDataset(lm_datasets, 'test', model_emb=model_emb)

# Create data loaders with RandomSampler
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=RandomSampler(valid_dataset))
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=RandomSampler(test_dataset))

def infinite_data_loader(data_loader, device):
    while True:
        for batch in data_loader:
            batch_data = batch[0].to(device)
            batch_cond = {k: v.to(device) for k, v in batch[1].items()}
            yield batch_data, batch_cond

# Convert the data loaders to infinite data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader_infinite = infinite_data_loader(train_loader, device)  # Ensure train_loader returns batches on the correct device
valid_loader_infinite = infinite_data_loader(valid_loader, device)  # Ensure valid_loader returns batches on the correct device

# Sample data from the infinite data loaders
train_data_iter = iter(train_loader_infinite)
valid_data_iter = iter(valid_loader_infinite)
test_data_iter = iter(test_loader)  # Test data is usually not infinite

print("Sample from train dataset:", next(train_data_iter))
print("Sample from validation dataset:", next(valid_data_iter))
print("Sample from test dataset:", next(test_data_iter))

class UniformSampler():
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    Sampler performs unbiased importance sampling, in which the
    objective's mean is unchanged.
    TODO confirm & update comment
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long()#.to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float()#.to(device)
        return indices, weights

# Helper functions for training loop
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        # logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            # logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

import numpy as np
import os
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW  # Make sure to import AdamW optimizer

# Define the noise schedule
scale = 1000 / num_diffusion_timesteps
beta_start = scale * 0.0001
beta_end = scale * 0.02
betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

# Instantiate the diffusion model & transformer
diffusion = GaussianDiffusion(betas=betas)


model = TransformerNetModel(vocab_size=vocab_size, input_dims=embedding_dim, hidden_t_dim=hidden_dim, output_dims=output_dims).to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'### The parameter count is {pytorch_total_params}')

class TrainLoop():
    def __init__(self, model, diffusion, data, batch_size, lr, ema_rate, weight_decay=0.0, learning_steps=0, eval_data=None, eval_interval=-1):
        self.model = model.to(device)
        self.ddp_model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.schedule_sampler = UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.eval_data = eval_data
        self.eval_interval = eval_interval
        self.step = 0
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]
        self.train_losses = []
        self.eval_losses = []

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = self.step / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def run_step(self, batch, cond):
        batch = batch.to(device)
        cond = {k: v.to(device) for k, v in cond.items()}
        self.forward_backward(batch, cond)
        self.optimize_normal()

    def forward_only(self, batch, cond):
        with torch.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(device)  # Move batch to GPU
                micro_cond = {k: v[i: i + self.microbatch].to(device) for k, v in cond.items()}  # Move cond to GPU
                t, weights = self.schedule_sampler.sample(micro.shape[0], device)
                weights = weights.to(device)  # Ensure weights is on GPU
                losses = self.diffusion.training_losses(self.ddp_model, micro, t, model_kwargs=micro_cond)
                log_loss_dict(self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()})
                self.eval_losses.append(losses['loss'].mean().item())

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(device)  # Move batch to GPU
            micro_cond = {k: v[i: i + self.microbatch].to(device) for k, v in cond.items()}  # Move cond to GPU
            t, weights = self.schedule_sampler.sample(micro.shape[0], device)
            weights = weights.to(device)  # Ensure weights is on GPU
            losses = self.diffusion.training_losses(self.ddp_model, micro, t, model_kwargs=micro_cond)
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            loss.backward()
            self.train_losses.append(loss.item())

    def run_loop(self):
        try:
            with tqdm(total=self.learning_steps, desc="Training Progress") as pbar:
                while not self.learning_steps or self.step < self.learning_steps:
                    batch, cond = next(self.data)
                    self.run_step(batch, cond)
                    if self.eval_data is not None and self.step % self.eval_interval == 0:
                        batch_eval, cond_eval = next(self.eval_data)
                        self.forward_only(batch_eval, cond_eval)
                    self.step += 1
                    pbar.update(1)
        except StopIteration:
            print("Data loader exhausted. Saving the model...")

        # Save the model after training is completed
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(self.model.state_dict(), 'checkpoints/trained_model.pth')
        print("Model saved successfully.")
        
        # Plotting the training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        if self.eval_losses:
            plt.plot(self.eval_losses, label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss over Time')
        plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = infinite_data_loader(train_loader, device)  # Ensure train_loader returns batches on the correct device
valid_loader = infinite_data_loader(valid_loader, device)  # Ensure valid_loader returns batches on the correct device

TrainLoop(
    model=model.to(device),
    diffusion=diffusion,
    data=iter(train_loader),
    batch_size=batch_size,
    lr=lr,
    ema_rate=ema_rate,
    weight_decay=weight_decay,
    learning_steps=learning_steps,
    eval_data=iter(valid_loader),
    eval_interval=eval_interval
).run_loop()


# TODO move to separate file 
# Inference stage

import numpy as np
import torch
from transformers import BertTokenizer

# Ensure the device is set correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = TransformerNetModel(
    vocab_size=vocab_size, 
    input_dims=embedding_dim, 
    hidden_t_dim=hidden_dim, 
    output_dims=output_dims
)
model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load('checkpoints/trained_model.pth', map_location=device))
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
