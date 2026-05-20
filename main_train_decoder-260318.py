import os
import random
import numpy as np
import torch

from data.vocab import build_char_vocab, build_gesture_vocab
from data.dataset_decode import GestureDataset
from models.bilstm_decoder import BiLSTMDecoder
from training.train_decoder import train_decoder
from argparse import ArgumentParser

from data.codebook import encode_text
from inference.decode import decode_gesture


def parse_args():
    parser = ArgumentParser(description="Train BiLSTM decoder baseline")

    parser.add_argument("--train_path", type=str, default="datasets/news.en.train.00000000-00009999.txt", help="dataset for training")
    parser.add_argument("--val_path", type=str, default="datasets/news.en.val.00000000-00009999.txt", help="dataset for validation")
    parser.add_argument("--save_path", type=str, default="checkpoints/bilstm_decoder_best.pt")
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--save_metric", type=str, default="wer", choices=["wer", "cer", "loss"])
    parser.add_argument("--clip_pred_to_target_len", action="store_true")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run_name", type=str, default="bilstm_decoder_only")

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        return [x.strip().lower() for x in f if x.strip()]


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    char2idx, idx2char = build_char_vocab()
    gesture2idx, idx2gesture = build_gesture_vocab()

    train_sentences = load_sentences(args.train_path)
    val_sentences = load_sentences(args.val_path)

    train_dataset = GestureDataset(
        sentences=train_sentences,
        char2idx=char2idx,
        gesture2idx=gesture2idx,
        max_len=args.max_len
    )

    val_dataset = GestureDataset(
        sentences=val_sentences,
        char2idx=char2idx,
        gesture2idx=gesture2idx,
        max_len=args.max_len
    )

    model = BiLSTMDecoder(
        gesture_vocab=len(gesture2idx),
        char_vocab=len(char2idx),
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    model = train_decoder(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        char2idx=char2idx,
        idx2char=idx2char,
        device=device,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_metric=args.save_metric,
        clip_pred_to_target_len=args.clip_pred_to_target_len
    )

    print("training finished.")

    sample = "while we were there"
    encoded = encode_text(sample)

    pred = decode_gesture(
        model=model,
        gesture_text=encoded,
        gesture2idx=gesture2idx,
        idx2char=idx2char,
        device=device,
        max_len=256
    )

    print("sample original:", sample)
    print("sample encoded :", encoded)
    print("sample decoded :", pred)


if __name__ == "__main__":
    main()