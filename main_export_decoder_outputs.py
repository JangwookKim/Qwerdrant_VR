# main_export_decoder_outputs.py

import argparse
import json
import torch
from torch.utils.data import DataLoader

from data.vocab import build_char_vocab, build_gesture_vocab, PAD
from data.dataset_decode import GestureDataset

from models.bilstm_decoder import BiLSTMDecoder
from models.transformer_decoder import TransformerGestureDecoder
from models.codebook_transformer_decoder import CodebookAwareTransformerGestureDecoder

from inference.decode import run_code_decoder_batch

#python main_export_decoder_outputs.py --model_type codebook_transformer --ckpt_path checkpoints/codebook_transformer_decoder-shard09-best.pt --input_path datasets/news.en.train.00000000-00009999.txt --output_path datasets/contextual_train_from_codebook_transformer.jsonl --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --max_len 256 --batch_size 32 --topk 5

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["bilstm", "transformer", "codebook_transformer"],
    )
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)

    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--topk", type=int, default=5)

    # shared
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)

    # bilstm
    p.add_argument("--hidden", type=int, default=256)

    # transformer
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=512)

    p.add_argument("--pretty_json", action="store_true")

    return p.parse_args()


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        return [x.strip().lower() for x in f if x.strip()]


def labels_to_text(batch_labels, idx2char):
    texts = []

    for seq in batch_labels:
        chars = []

        for token in seq:
            token = token.item()
            ch = idx2char[token]

            if ch == PAD:
                break

            chars.append(ch)

        texts.append("".join(chars))

    return texts


def get_target_lengths(batch_labels, idx2char):
    lengths = []

    for seq in batch_labels:
        length = 0

        for token in seq:
            token = token.item()
            ch = idx2char[token]

            if ch == PAD:
                break

            length += 1

        lengths.append(length)

    return lengths


def build_model(args, device, char2idx, gesture2idx):
    if args.model_type == "bilstm":
        model = BiLSTMDecoder(
            gesture_vocab=len(gesture2idx),
            char_vocab=len(char2idx),
            embed_dim=args.embed_dim,
            hidden=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
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

    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    char2idx, idx2char = build_char_vocab()
    gesture2idx, idx2gesture = build_gesture_vocab()

    model = build_model(args, device, char2idx, gesture2idx)

    use_codebook_features = args.model_type == "codebook_transformer"

    sentences = load_sentences(args.input_path)

    dataset = GestureDataset(
        sentences=sentences,
        char2idx=char2idx,
        gesture2idx=gesture2idx,
        max_len=args.max_len,
        use_codebook_features=use_codebook_features,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    total = 0

    with open(args.output_path, "w", encoding="utf-8") as f:
        for batch in loader:
            target_lengths = get_target_lengths(batch["labels"], idx2char)
            target_texts = labels_to_text(batch["labels"], idx2char)

            result = run_code_decoder_batch(
                model=model,
                batch_input_ids=batch["input_ids"],
                idx2char=idx2char,
                device=device,
                target_lengths=target_lengths,
                codebook_vectors=batch.get("codebook_vectors", None),
                topk=args.topk,
                return_logits=False,
            )

            argmax_texts = result["argmax_texts"]
            topk_outputs = result["topk"]

            for target, argmax, topk in zip(
                target_texts,
                argmax_texts,
                topk_outputs,
            ):
                row = {
                    "target": target,
                    "argmax": argmax,
                    "topk": topk,
                }

                #f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
                if args.pretty_json:
                    text = json.dumps(row, ensure_ascii=False, indent=2)
                else:
                    text = json.dumps(row, ensure_ascii=False)
                
                f.write(text + "\n")
                
                total += 1

    print(f"Exported {total} samples to {args.output_path}")


if __name__ == "__main__":
    main()