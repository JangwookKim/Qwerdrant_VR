import argparse
import torch
import random
import numpy as np

from data.vocab import build_char_vocab, build_gesture_vocab, PAD
from models.bilstm_decoder import BiLSTMDecoder
from models.transformer_decoder import TransformerGestureDecoder
from models.codebook_transformer_decoder import CodebookAwareTransformerGestureDecoder
from training.train_decoder import train_decoder_across_shards

from data.codebook import encode_text
from inference.decode import decode_gesture


def parse_args():
    p = argparse.ArgumentParser()

    # shard
    #p.add_argument("--train_pattern", type=str, default="datasets/news.en.train.000{:01d}0000-000{:01d}9999.txt")
    #p.add_argument("--val_pattern", type=str, default="datasets/news.en.val.000{:01d}0000-000{:01d}9999.txt")
    p.add_argument("--data_dir", type=str, default="datasets")
    p.add_argument("--num_shards", type=int, default=10)
    p.add_argument("--shard_size", type=int, default=10000)
    p.add_argument("--start_shard", type=int, default=0)
    p.add_argument("--epochs_per_shard", type=int, default=5)

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_len", type=int, default=256)
    
    # model
    p.add_argument("--model_type", type=str, default="transformer", choices=["bilstm", "transformer", "codebook_transformer"])

    p.add_argument("--use_codebook_features", action="store_true")
    p.add_argument("--codebook_dim", type=int, default=26)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)

    # transformer only
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=512)

    # saving
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_prefix", type=str, default="bilstm_decoder")
    p.add_argument("--save_metric", type=str, default="wer")

    # misc
    p.add_argument("--clip_pred_to_target_len", action="store_true")
    p.add_argument("--resume_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()

#python main_train_decoder.py --model_type bilstm --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --batch_size 32 --lr 0.0005 --max_len 256 --epochs_per_shard 5
#python main_train_decoder.py --model_type transformer --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --batch_size 32 --lr 0.0005 --max_len 256 --epochs_per_shard 5
#python main_train_decoder.py --model_type codebook_transformer --save_prefix codebook_transformer_decoder --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_shard_paths(split="train", num_shards=10, shard_size=10000, data_dir="datasets"):
    paths = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size - 1
        path = f"{data_dir}/news.en.{split}.{start:08d}-{end:08d}.txt"
        paths.append(path)
    return paths

def main():
    args = parse_args()
    if args.save_prefix == "bilstm_decoder" and args.model_type == "transformer":
        args.save_prefix = "transformer_decoder"
    
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    char2idx, idx2char = build_char_vocab()
    gesture2idx, _ = build_gesture_vocab()

    if args.model_type == "bilstm":
        model = BiLSTMDecoder(
            gesture_vocab=len(gesture2idx),
            char_vocab=len(char2idx),
            embed_dim=args.embed_dim,
            hidden=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    elif args.model_type == "transformer":
        model = TransformerGestureDecoder(
            gesture_vocab=len(gesture2idx),
            char_vocab=len(char2idx),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_len=args.max_len,
            pad_idx=gesture2idx[PAD],
        )

    elif args.model_type == "codebook_transformer":
        model = CodebookAwareTransformerGestureDecoder(
            gesture_vocab=len(gesture2idx),
            char_vocab=len(char2idx),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_len=args.max_len,
            codebook_dim=26,
            pad_idx=gesture2idx[PAD],
        )
    
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model.to(device)

    train_decoder_across_shards(
        model=model,
        char2idx=char2idx,
        idx2char=idx2char,
        gesture2idx=gesture2idx,
        device=device,
        data_dir=args.data_dir,
        num_shards=args.num_shards,
        shard_size=args.shard_size,
        start_shard=args.start_shard,
        epochs_per_shard=args.epochs_per_shard,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        save_dir=args.save_dir,
        save_prefix=args.save_prefix,
        save_metric=args.save_metric,
        clip_pred_to_target_len=args.clip_pred_to_target_len,
        use_codebook_features = (args.model_type == "codebook_transformer"),
        resume_path=args.resume_path
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
        max_len=256,
        codebook_aware=(args.model_type == "codebook_transformer"),
    )

    print("sample original:", sample)
    print("sample encoded :", encoded)
    print("sample decoded :", pred)


if __name__ == "__main__":
    main()