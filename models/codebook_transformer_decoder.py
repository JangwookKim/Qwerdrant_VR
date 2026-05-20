import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class CodebookAwareTransformerGestureDecoder(nn.Module):
    def __init__(
        self,
        gesture_vocab: int,
        char_vocab: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
        codebook_dim: int = 26,
        pad_idx: Optional[int] = None,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})."
            )

        self.pad_idx = pad_idx

        self.gesture_embedding = nn.Embedding(
            gesture_vocab,
            embed_dim,
            padding_idx=pad_idx if pad_idx is not None else None,
        )

        self.codebook_proj = nn.Sequential(
            nn.Linear(codebook_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.input_norm = nn.LayerNorm(embed_dim)

        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, char_vocab),
        )

    def forward(self, input_ids, codebook_vectors):
        """
        input_ids:        [batch, seq_len]
        codebook_vectors: [batch, seq_len, 26]
        return:           [batch, seq_len, char_vocab]
        """

        padding_mask = None
        if self.pad_idx is not None:
            padding_mask = input_ids.eq(self.pad_idx)

        x_g = self.gesture_embedding(input_ids)
        x_c = self.codebook_proj(codebook_vectors)

        x = self.input_norm(x_g + x_c)
        x = self.positional_encoding(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        logits = self.fc(x)
        return logits