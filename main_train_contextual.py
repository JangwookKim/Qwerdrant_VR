# main_train_contextual.py

import argparse
import torch

from data.contextual_dataset import (
    ContextualDataset,
    build_contextual_char_vocab,
)
from data.vocab import PAD
from models.contextual_transformer import ContextualTransformer
from training.train_contextual import train_contextual

# python main_train_contextual.py --train_path datasets/contextual_train_from_codebook_transformer.00000000-00009999.jsonl --val_path datasets/contextual_val_from_codebook_transformer.00000000-00009999.jsonl --save_path checkpoints/contextual_transformer_same_length-best.pt --max_len 256 --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --batch_size 32 --epochs 100 --lr 0.0001
# python main_train_contextual.py --train_path datasets/contextual_train_from_codebook_transformer.00010000-00019999.jsonl --val_path datasets/contextual_val_from_codebook_transformer.00010000-00019999.jsonl --save_path checkpoints/contextual_transformer_same_length-continued-best.pt --resume_path checkpoints/contextual_transformer_same_length-best.pt --max_len 256 --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --batch_size 32 --epochs 100 --lr 0.0001

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--resume_path", type=str, default=None)

    p.add_argument("--max_len", type=int, default=256)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    char2idx, idx2char = build_contextual_char_vocab()

    train_dataset = ContextualDataset(
        jsonl_path=args.train_path,
        char2idx=char2idx,
        max_len=args.max_len,
        require_same_length=True,
    )

    val_dataset = ContextualDataset(
        jsonl_path=args.val_path,
        char2idx=char2idx,
        max_len=args.max_len,
        require_same_length=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")

    model = ContextualTransformer(
        vocab_size=len(char2idx),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_idx=char2idx[PAD],
    ).to(device)

    if args.resume_path is not None:
        print("Loading checkpoint:", args.resume_path)
        state = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state)

    model = train_contextual(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        char2idx=char2idx,
        idx2char=idx2char,
        pad_idx=char2idx[PAD],
        device=device,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
    )

    print("contextual decoder training finished.")


if __name__ == "__main__":
    main()