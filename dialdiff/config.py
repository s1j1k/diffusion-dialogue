"""Model and training configuration, saved inside every checkpoint."""
from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # Data
    tokenizer: str = "bert-base-uncased"
    seq_len: int = 128             # [CLS] source [SEP] target [SEP] + padding

    # Model
    emb_dim: int = 128             # word embedding space the diffusion runs in
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    time_dim: int = 128

    # Diffusion
    num_steps: int = 2000
    schedule: str = "sqrt"
    nll_fraction: float = 1.0      # share of positions used for the rounding loss

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    train_steps: int = 50_000
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    seed: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
