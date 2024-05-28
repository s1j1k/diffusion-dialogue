import math
from transformers import BertConfig, BertModel
import torch.nn as nn
import torch as th

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TransformerNetModel(nn.Module):
    def __init__(self, vocab_size, input_dims, hidden_t_dim, output_dims):
        super().__init__()

        config = BertConfig.from_pretrained('bert-base-uncased')
        config.hidden_dropout_prob = 0

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.hidden_size = config.hidden_size

        self.lm_head = nn.Linear(self.input_dims, vocab_size)

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_t_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                               nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))

        temp_bert = BertModel.from_pretrained('bert-base-uncased', config=config)
        self.word_embedding = temp_bert.embeddings.word_embeddings
        
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight
        
        self.input_transformers = temp_bert.encoder
        self.register_buffer("position_ids", th.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = temp_bert.embeddings.position_embeddings
        self.LayerNorm = temp_bert.embeddings.LayerNorm
        
        del temp_bert.embeddings
        del temp_bert.pooler

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                  nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x, timesteps):
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h
