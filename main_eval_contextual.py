# main_eval_contextual.py

import argparse
import json
import torch
from tqdm import tqdm

from data.contextual_dataset import build_contextual_char_vocab
from data.vocab import PAD
from models.contextual_transformer import ContextualTransformer
from inference.contextual_decode import contextual_greedy_decode
from training.metrics import cer, wer

#python main_eval_contextual.py --ckpt_path checkpoints/contextual_transformer-best.pt --eval_path datasets/contextual_val_from_codebook_transformer.00000000-00009999.jsonl --output_path checkpoints/contextual_eval_predictions.00000000-00009999.txt --embed_dim 128 --num_heads 4 --ff_dim 512 --num_encoder_layers 4 --num_decoder_layers 4 --dropout 0.1 --max_input_len 256 --max_target_len 256

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--eval_path", type=str, required=True)
    p.add_argument("--output_path", type=str, default="checkpoints/contextual_eval_predictions.txt")

    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=256)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--num_encoder_layers", type=int, default=4)
    p.add_argument("--num_decoder_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    return p.parse_args()


def load_jsonl(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            samples.append({
                "argmax": obj["argmax"],
                "target": obj["target"],
            })

    return samples


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


def save_predictions(path, noisy_texts, preds, targets):
    with open(path, "w", encoding="utf-8") as f:
        for noisy, pred, target in zip(noisy_texts, preds, targets):
            f.write(f"NOISY\t{noisy}\n")
            f.write(f"PRD\t{pred}\n")
            f.write(f"GT\t{target}\n\n")


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

    samples = load_jsonl(args.eval_path)

    noisy_texts = []
    preds = []
    targets = []

    for sample in tqdm(samples):
        noisy = sample["argmax"]
        target = sample["target"]

        pred = contextual_greedy_decode(
            model=model,
            noisy_text=noisy,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len,
        )

        noisy_texts.append(noisy)
        preds.append(pred)
        targets.append(target)

    noisy_cer = cer(noisy_texts, targets)
    noisy_wer = wer(noisy_texts, targets)

    pred_cer = cer(preds, targets)
    pred_wer = wer(preds, targets)

    print(f"Eval samples: {len(targets)}")
    print()
    print("=== Before Contextual Decoder ===")
    print(f"CER: {noisy_cer:.6f}")
    print(f"WER: {noisy_wer:.6f}")
    print()
    print("=== After Contextual Decoder ===")
    print(f"CER: {pred_cer:.6f}")
    print(f"WER: {pred_wer:.6f}")

    print()
    print("=== Sample predictions ===")
    for i in range(min(10, len(targets))):
        print(f"[{i}] NOISY: {noisy_texts[i]}")
        print(f"[{i}] PRD  : {preds[i]}")
        print(f"[{i}] GT   : {targets[i]}")
        print()

    save_predictions(args.output_path, noisy_texts, preds, targets)


if __name__ == "__main__":
    main()