# Inference stage

import numpy as np
import torch
from transformers import BertTokenizer
from transformer_model import TransformerNetModel
from diffusion_model import GaussianDiffusion

# Ensure the device is set correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO load the params from a file

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