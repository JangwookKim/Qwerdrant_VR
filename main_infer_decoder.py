import argparse
import torch

from data.vocab import build_char_vocab, build_gesture_vocab, PAD
from models.bilstm_decoder import BiLSTMDecoder
from models.transformer_decoder import TransformerGestureDecoder
from inference.decode import decode_gesture

#python main_infer_decoder.py --model_type transformer --ckpt_path checkpoints/transformer_decoder-shard09-best.pt --embed_dim 128 --num_heads 4 --ff_dim 512 --num_layers 4 --dropout 0.1 --max_len 256

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_type", type=str, default="bilstm",
                   choices=["bilstm", "transformer"])
    p.add_argument("--ckpt_path", type=str, required=True)

    p.add_argument("--max_len", type=int, default=256)

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

    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.ckpt_path}")
    print("Type encoded gesture string. Type 'quit' to exit.")

    while True:
        text = input("gesture> ").strip()

        if text.lower() == "quit":
            break

        try:
            pred = decode_gesture(
                model=model,
                gesture_text=text,
                gesture2idx=gesture2idx,
                idx2char=idx2char,
                device=device,
                max_len=args.max_len,
            )
            print("decoded:", pred)
        except Exception as e:
            print("error:", e)


if __name__ == "__main__":
    main()