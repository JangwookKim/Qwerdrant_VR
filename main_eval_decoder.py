import argparse
import torch
from torch.utils.data import DataLoader

from data.vocab import build_char_vocab, build_gesture_vocab, PAD
from data.dataset_decode import GestureDataset
from models.bilstm_decoder import BiLSTMDecoder
from models.transformer_decoder import TransformerGestureDecoder
from models.codebook_transformer_decoder import CodebookAwareTransformerGestureDecoder
from inference.decode import decode_batch
from training.metrics import cer, wer

#python main_eval_decoder.py --model_type bilstm --ckpt_path checkpoints/bilstm_decoder-shard09-best.pt --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --max_len 256
#python main_eval_decoder.py --model_type transformer --ckpt_path checkpoints/transformer_decoder-shard09-best.pt --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --max_len 256
#python main_eval_decoder.py --model_type codebook_transformer --ckpt_path checkpoints/codebook_transformer_decoder-shard09-best.pt --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --max_len 256


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_type", type=str, default="bilstm",
                   choices=["bilstm", "transformer", "codebook_transformer"])

    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--eval_path", type=str,
                   default="datasets/news.en.val.00000000-00009999.txt")
    p.add_argument("--output_path", type=str,
                   default="checkpoints/eval_predictions.txt")

    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)

    # shared
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)

    # bilstm
    p.add_argument("--hidden", type=int, default=256)

    # transformer
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=512)

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


def evaluate():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    char2idx, idx2char = build_char_vocab()
    gesture2idx, idx2gesture = build_gesture_vocab()

    model = build_model(args, device, char2idx, gesture2idx)

    sentences = load_sentences(args.eval_path)

    dataset = GestureDataset(
        sentences=sentences,
        char2idx=char2idx,
        gesture2idx=gesture2idx,
        max_len=args.max_len,
        use_codebook_features=(args.model_type == "codebook_transformer"),
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    for batch in loader:
        target_lengths = get_target_lengths(batch["labels"], idx2char)

        pred_texts = decode_batch(
            model=model,
            batch_input_ids=batch["input_ids"],
            idx2char=idx2char,
            device=device,
            target_lengths=target_lengths,
            codebook_vectors=batch.get("codebook_vectors", None),
        )

        target_texts = labels_to_text(batch["labels"], idx2char)

        all_preds.extend(pred_texts)
        all_targets.extend(target_texts)

    total_cer = cer(all_preds, all_targets)
    total_wer = wer(all_preds, all_targets)

    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Eval samples: {len(all_targets)}")
    print(f"CER: {total_cer:.6f}")
    print(f"WER: {total_wer:.6f}")
    print()

    print("=== Sample predictions ===")
    for i in range(min(10, len(all_targets))):
        print(f"[{i}] GT : {all_targets[i]}")
        print(f"[{i}] PRD: {all_preds[i]}")
        print()

    save_predictions(args.output_path, all_preds, all_targets)


def save_predictions(path, preds, targets):
    with open(path, "w", encoding="utf-8") as f:
        for p, t in zip(preds, targets):
            f.write(f"GT\t{t}\n")
            f.write(f"PRD\t{p}\n\n")


if __name__ == "__main__":
    evaluate()