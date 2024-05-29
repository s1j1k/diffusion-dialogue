import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from transformer_model import TransformerNetModel
from diffusion_model import GaussianDiffusion
import json
import logging as log
from functools import partial
from data_utils import load_data, tokenize_function, merge_and_mask, pad_function, TextDataset, infinite_data_loader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

log.basicConfig(filename='logs.log', level=log.DEBUG, format="%(asctime)s:%(levelname)s: %(message)s")
log.getLogger().addHandler(log.StreamHandler())

# Ensure the device is set correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get tokenizer from BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size
log.info("Vocab size", vocab_size)

# Load the params from a file
with open('training_config.json') as f:
    config = json.load(f)
    log.debug("Training config from file:", config)

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
    test_path = f'{data_dir}/test_full.jsonl'

    # Load datasets with size restriction
    test_limit = 4    # Limit the size of the test set

    test_data = load_data(test_path, limit=test_limit)
    log.debug("Test Data Samples:", len(test_data['src']))
    log.debug(test_data['src'][0])
    raw_datasets = HFDataset.from_dict(test_data)
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
    log.debug("Test Set:", len(tokenized_datasets['test']))

    # Apply merge and mask to the tokenized datasets
    tokenized_datasets = DatasetDict({
        'test': tokenized_test_dataset
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
    log.debug("Test Set:", len(lm_datasets['test']))
    log.debug('Padded Dataset:', lm_datasets)

    # Create datasets for test sets
    test_dataset = TextDataset(lm_datasets, 'test', model_emb=model_emb)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], sampler=RandomSampler(test_dataset))
    test_data_iter = iter(test_loader)

    # Initialize the model
    model = TransformerNetModel(
        vocab_size=vocab_size, 
        input_dims=config['embedding_dim'], 
        hidden_t_dim=config['hidden_dim'], 
        output_dims=config['output_dims']
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

    def generate_text(model, tokenizer, input_ids, max_length=128, num_timesteps=2000, temperature=0.5):
        model.eval()
        with torch.no_grad():
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
            
            return generated_text

    def evaluate_model(model, tokenizer, test_loader):
        all_generated_texts = []
        all_target_texts = []
        all_losses = []

        criterion = torch.nn.CrossEntropyLoss()

        for batch in test_loader:
            input_ids = batch[1]['input_ids'].to(device)
            target_ids = batch[1]['input_ids'].to(device)
            target_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]

            generated_texts = []
            for ids in input_ids:
                generated_text = generate_text(model, tokenizer, ids.unsqueeze(0), temperature=0.5)
                generated_texts.append(generated_text)

                # Compute the loss
                input_embeds = model.word_embedding(ids.unsqueeze(0)).to(device)
                noise = torch.randn_like(input_embeds).to(device)
                diffusion = GaussianDiffusion(betas=np.linspace(1e-4, 0.02, config['num_diffusion_timesteps']))
                samples = diffusion.p_sample_loop(
                    model=model,
                    shape=input_embeds.shape,
                    noise=noise,
                    device=device,
                    progress=True,
                    clamp_step=None
                )
                logits = model.lm_head(samples[-1].to(device))
                
                # Convert target ids to LongTensor
                ids = ids.long()
                loss = criterion(logits.view(-1, logits.size(-1)), ids.view(-1))
                all_losses.append(loss.item())

            all_generated_texts.extend(generated_texts)
            all_target_texts.extend(target_texts)

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
            log.info(f"Target Text: {all_target_texts[i]}")
            log.info(f"Generated Text: {all_generated_texts[i]}")

        return avg_bleu_score, avg_loss


    # Perform evaluation
    avg_bleu_score, avg_loss = evaluate_model(model, tokenizer, test_loader)

    log.info(f"Average BLEU Score on Test Set: {avg_bleu_score}")
    log.info(f"Average Loss on Test Set: {avg_loss}")

if __name__ == "__main__":
    main()