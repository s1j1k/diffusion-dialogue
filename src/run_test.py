import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from transformer_model import TransformerNetModel
from diffusion_model import GaussianDiffusion
import json
import logging as log
from functools import partial
from data_utils import load_data, tokenize_function, merge_and_mask, pad_function, TextDataset, infinite_data_loader, CustomBertTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from train_utils import log

# Ensure the device is set correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get tokenizer from training (can also just load with same settings)
tokenizer = CustomBertTokenizer('./checkpoints/tokenizer')
vocab_size = tokenizer.vocab_size
log.info("Vocab size %s", vocab_size)

# Load the params from a file
with open('./src/training_config.json') as f:
    config = json.load(f)
    log.debug("Training config from file: %s", config)

# Configure betas for Guassian diffusion
scale = 1000 / config['num_diffusion_timesteps']
beta_start = scale * 0.0001
beta_end = scale * 0.02
betas = np.linspace(beta_start, beta_end, config['num_diffusion_timesteps'], dtype=np.float64)

def evaluate_model(model, tokenizer, test_loader):
    all_generated_texts = []
    all_target_texts = []
    all_losses = []
    all_source_texts = []

    criterion = torch.nn.CrossEntropyLoss()

    for batch in test_loader:
        input_ids_x =  batch[1]['input_ids'].to(device)
        x_start = model.get_embeds(input_ids_x.cpu()).to(device)
        input_ids_mask = batch[1]['input_mask'].to(device)
        input_ids_mask_ori = input_ids_mask

        noise = torch.randn_like(x_start).to(device)
        input_ids_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(device)
        x_noised = torch.where(input_ids_mask == 0, x_start, noise)

        sample_shape = (x_start.shape[0], config["seq_len"], config["embedding_dim"])

        # Set up the diffusion process
        diffusion = GaussianDiffusion(betas=betas)

        sample_fn = diffusion.p_sample_loop
        
        log.info("Running sampling process on the input ids")
        samples = sample_fn(
            model=model,
            shape=sample_shape,
            noise=x_noised,
            device=device,
            progress=True,
            clamp_step=None,
            mask=input_ids_mask,
            x_start=x_start,
        )

        sample = samples[-1]
        sample.to(device)
        model.to(device)
        logits = model.get_logits(sample)
        cands = torch.topk(logits, k=1, dim=-1)

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = config["seq_len"] - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            len_x = config["seq_len"]  - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))
            
        # Convert target ids to LongTensor
        ids = input_ids_x.long()
        loss = criterion(logits.view(-1, logits.size(-1)), ids.view(-1))
        all_losses.append(loss.item())

        all_source_texts.extend(word_lst_source)

        # Target text
        all_target_texts.extend(word_lst_ref)

        # Generated text
        all_generated_texts.extend(word_lst_recover)

    # Calculate BLEU score for evaluation
    smoothing_function = SmoothingFunction().method1
    bleu_scores = [sentence_bleu([target.split()], gen.split(), smoothing_function=smoothing_function) 
                   for target, gen in zip(all_target_texts, all_generated_texts)]

    avg_bleu_score = np.mean(bleu_scores)
    avg_loss = np.mean(all_losses)

    log.info(f"Average BLEU Score: {avg_bleu_score}")
    log.info(f"Average Loss: {avg_loss}")

    # Log a few examples
    for i in range(min(5, len(all_target_texts))):
        log.info(f"\n------------------EXAMPLE #{i+1}-------------------\n")
        log.info(f"Prompt Text: {all_source_texts[i]}\n")
        log.info(f"\tTarget Response Text: {all_target_texts[i]}\n")
        log.info(f"\tGenerated Response Text: {all_generated_texts[i]}\n")
        log.info(f"\n------------------------------------------------\n")

    return avg_bleu_score, avg_loss

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

    # Initialize the model
    model = TransformerNetModel(
        vocab_size=vocab_size, 
        input_dims=config['embedding_dim'], 
        hidden_t_dim=config['hidden_t_dim'], 
        output_dims=config['output_dims']
    )
    model.to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load('./checkpoints/trained_model.pth'))
    model.eval()

    # Load embeddings and use the weights from the model
    model_emb = model.word_embedding.eval().requires_grad_(False).to(device)
    log.info("Embedding layer %s", model_emb)

    # Dataset path definition
    data_dir = "./datasets/CommonsenseConversation"
    test_path = f'{data_dir}/test.jsonl'

    # Load datasets with size restriction
    test_limit = 4    # Limit the size of the test set

    test_data = load_data(test_path, limit=test_limit)
    log.debug("Test Data Samples: %s", len(test_data['src']))
    log.debug(test_data['src'][0])
    raw_datasets = HFDataset.from_dict(test_data)
    log.debug(raw_datasets)
    log.debug(raw_datasets[0])

    # Use partial to pass the tokenizer to the tokenize_function
    tokenize_function_with_tokenizer = partial(tokenize_function, tokenizer=tokenizer)

    # Create datasets
    test_dataset = HFDataset.from_dict(test_data)

    # Tokenize datasets
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
        'test': tokenized_test_dataset
    })

    log.info("Tokenization complete.")
    log.debug("Test Set: %s", len(tokenized_datasets['test']))

    # Apply merge and mask to the tokenized datasets
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
    log.debug("Test Set Length: %d", len(lm_datasets['test']))
    log.debug('Padded Dataset: %s', lm_datasets)

    # Create datasets for test sets
    test_dataset = TextDataset(lm_datasets, 'test', model_emb=model_emb)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], sampler=RandomSampler(test_dataset))

    # Perform evaluation
    avg_bleu_score, avg_loss = evaluate_model(model, tokenizer, test_loader)

    log.info(f"Average BLEU Score on Test Set: {avg_bleu_score}")
    log.info(f"Average Loss on Test Set: {avg_loss}")

if __name__ == "__main__":
    main()
