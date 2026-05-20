# main_infer_contextual.py

import argparse
import torch

from data.contextual_dataset import build_contextual_char_vocab
from data.vocab import PAD
from models.contextual_transformer import ContextualTransformer
from inference.contextual_decode import contextual_greedy_decode

#python main_infer_contextual.py --ckpt_path checkpoints/contextual_transformer-best.pt --embed_dim 128 --num_heads 4 --ff_dim 512 --num_encoder_layers 4 --num_decoder_layers 4 --dropout 0.1 --max_input_len 256 --max_target_len 256

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt_path", type=str, required=True)

    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=256)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--num_encoder_layers", type=int, default=4)
    p.add_argument("--num_decoder_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    return p.parse_args()


def build_model(args, device, vocab_size, pad_idx):
    model = ContextualTransformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        pad_idx=pad_idx,
    ).to(device)

    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    char2idx, idx2char = build_contextual_char_vocab()

    model = build_model(
        args=args,
        device=device,
        vocab_size=len(char2idx),
        pad_idx=char2idx[PAD],
    )

    print("Contextual decoder loaded.")
    print("Type noisy decoded text. Type 'quit' to exit.")

    while True:
        text = input("noisy> ").strip()

        if text.lower() == "quit":
            break

        corrected = contextual_greedy_decode(
            model=model,
            noisy_text=text,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len,
        )

        print("corrected:", corrected)


if __name__ == "__main__":
    main()