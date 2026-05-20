import torch

from data.vocab import build_char_vocab, build_gesture_vocab
from models.bilstm_decoder import BiLSTMDecoder
from inference.decode import decode_gesture


MAX_LEN = 256
CKPT_PATH = "checkpoints/bilstm_decoder_only-embed128-hidden256-best.pt"


def load_model(device):
    char2idx, idx2char = build_char_vocab()
    gesture2idx, idx2gesture = build_gesture_vocab()

    model = BiLSTMDecoder(
        gesture_vocab=len(gesture2idx),
        char_vocab=len(char2idx),
        embed_dim=128,
        hidden=256
    ).to(device)

    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, char2idx, idx2char, gesture2idx, idx2gesture


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, char2idx, idx2char, gesture2idx, idx2gesture = load_model(device)

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
                max_len=MAX_LEN
            )
            print("decoded:", pred)
        except Exception as e:
            print("error:", e)


if __name__ == "__main__":
    main()