# models/contextual_transformer.py

import math
from typing import Optional

import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class ContextualTransformer(nn.Module):
    """
    Character-level seq2seq Transformer.

    input:
        noisy text ids

    output:
        clean text ids
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        max_input_len: int = 256,
        max_target_len: int = 256,
        pad_idx: Optional[int] = None,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})."
            )

        self.pad_idx = pad_idx
        self.embed_dim = embed_dim

        self.src_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx if pad_idx is not None else None,
        )

        self.tgt_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx if pad_idx is not None else None,
        )

        self.src_positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_input_len,
            dropout=dropout,
        )

        self.tgt_positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_target_len,
            dropout=dropout,
        )

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, vocab_size),
        )

    def generate_square_subsequent_mask(self, size: int, device):
        """
        Causal mask for decoder self-attention.
        """
        mask = torch.triu(
            torch.ones(size, size, device=device),
            diagonal=1,
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        input_ids:         [B, S]
        decoder_input_ids: [B, T]
        attention_mask:    [B, S], 1 for valid, 0 for pad

        return:
            logits: [B, T, vocab_size]
        """

        device = input_ids.device

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask.eq(0)

        tgt_key_padding_mask = None
        if self.pad_idx is not None:
            tgt_key_padding_mask = decoder_input_ids.eq(self.pad_idx)

        tgt_mask = self.generate_square_subsequent_mask(
            decoder_input_ids.size(1),
            device=device,
        )

        src = self.src_embedding(input_ids) * math.sqrt(self.embed_dim)
        tgt = self.tgt_embedding(decoder_input_ids) * math.sqrt(self.embed_dim)

        src = self.src_positional_encoding(src)
        tgt = self.tgt_positional_encoding(tgt)

        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.fc(out)
        return logits