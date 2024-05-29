# FIXME need to define TransformerNetModel
#  The full Transformer model with attention and timestep embedding.

# Adapted from diffuSeq

# TODO code the transformer from scratch
# TODO design the transformer from scratch

from transformers import BertConfig, BertModel
import torch.nn as nn
import torch as th
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    device = timesteps.device
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32, device=device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class TransformerNetModel(nn.Module):
    def __init__(self, vocab_size, input_dims, hidden_t_dim, output_dims):
        super().__init__()

        config = BertConfig.from_pretrained('bert-base-uncased')
        # FIXME set to an actual value
        config.hidden_dropout_prob = 0

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        # self.dropout = dropout
        self.hidden_size = config.hidden_size
#         print("transformer self.hidden_size", self.hidden_size)

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        # Generate logits for hidden representation
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # Time embeddings
        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            # params as input features * output features
            nn.Linear(hidden_t_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        # Function to deal with having a hidden size not equal to input size, project to hidden size (?)
        if self.input_dims != config.hidden_size:
            # NOTE input_dims = 128
            # hidden_size = 768
            # self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
            #                                 nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
            # FIXME this is trying to convert to hidden size 768 why??? it's already in the hidden siz 
            # FIXME this actually doesn't seeem necessary to do????
            self.input_up_proj = nn.Sequential(nn.Linear(config.hidden_size, input_dims),
                                             nn.Tanh(), nn.Linear(input_dims, input_dims))

        # FIXME why is this temporary 
        temp_bert = BertModel.from_pretrained('bert-base-uncased', config=config)
        self.word_embedding = temp_bert.embeddings.word_embeddings
       
       # FIXME why do we do this 2 times????
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight
        # self.lm_head.weight.requires_grad = False
        # self.word_embedding.weight.requires_grad = False
            
        # TODO explain what is happening
        self.input_transformers = temp_bert.encoder
        # TODO explain what is doing
        self.register_buffer("position_ids", th.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = temp_bert.embeddings.position_embeddings
        self.LayerNorm = temp_bert.embeddings.LayerNorm
     
        del temp_bert.embeddings
        del temp_bert.pooler

        # FIXME When does this get used
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # FIXME what is happening here
        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))
    
    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
    # NOTE we make a simplifying assumption get the logits from linear layer
        return self.lm_head(hidden_repr)
                

    # FIXME what is the difference btween BertModel, BertConfig, BertTokenizer, maybe define it all in one place config for tokenizer + embeddings?


    def forward(self, x, timesteps):
        emb_t = self.time_embed(timestep_embedding(timesteps.to(self.time_embed[0].weight.device), self.hidden_t_dim))  # Ensure timesteps are on the same device
        emb_x = x
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length].to(x.device)  # Ensure position_ids are on the same device
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h