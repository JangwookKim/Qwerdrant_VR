# transformer_decoder.py

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

        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerGestureDecoder(nn.Module):
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
        pad_idx: Optional[int] = None,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})."
            )

        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(
            gesture_vocab,
            embed_dim,
            padding_idx=pad_idx if pad_idx is not None else None,
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      [batch, seq_len]
        return: [batch, seq_len, char_vocab]
        """

        padding_mask = None
        if self.pad_idx is not None:
            padding_mask = x.eq(self.pad_idx)  # [batch, seq_len]

        x = self.embedding(x)                  # [B, T, D]
        x = self.positional_encoding(x)        # [B, T, D]

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )                                      # [B, T, D]

        logits = self.fc(x)                    # [B, T, char_vocab]
        return logits