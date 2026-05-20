import torch
from torch.utils.data import DataLoader

from data.vocab import build_char_vocab, build_gesture_vocab
from data.dataset_decode import GestureDataset
from models.bilstm_decoder import BiLSTMDecoder
from inference.decode import decode_batch
from training.metrics import cer, wer


MAX_LEN = 256
BATCH_SIZE = 32
CKPT_PATH = "checkpoints/bilstm_decoder_only-embed128-hidden256-best.pt"
EVAL_PATH = "datasets/news.en.val.00000000-00009999.txt"   # 필요하면 test.txt로 바꿔도 됨


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
            if ch == "<pad>":
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
            if ch == "<pad>":
                break
            length += 1
        lengths.append(length)
    return lengths


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


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, char2idx, idx2char, gesture2idx, idx2gesture = load_model(device)

    sentences = load_sentences(EVAL_PATH)

    dataset = GestureDataset(
        sentences=sentences,
        char2idx=char2idx,
        gesture2idx=gesture2idx,
        max_len=MAX_LEN
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_targets = []

    for batch in loader:
        target_lengths = get_target_lengths(batch["labels"], idx2char)

        pred_texts = decode_batch(
            model=model,
            batch_input_ids=batch["input_ids"],
            idx2char=idx2char,
            device=device,
            target_lengths=target_lengths
        )

        target_texts = labels_to_text(batch["labels"], idx2char)

        all_preds.extend(pred_texts)
        all_targets.extend(target_texts)

    total_cer = cer(all_preds, all_targets)
    total_wer = wer(all_preds, all_targets)

    print(f"Eval samples: {len(all_targets)}")
    print(f"CER: {total_cer:.6f}")
    print(f"WER: {total_wer:.6f}")
    print()

    print("=== Sample predictions ===")
    for i in range(min(10, len(all_targets))):
        print(f"[{i}] GT : {all_targets[i]}")
        print(f"[{i}] PRD: {all_preds[i]}")
        print()
    
    save_predictions("checkpoints/eval_predictions.txt", all_preds, all_targets)

def save_predictions(path, preds, targets):
    with open(path, "w", encoding="utf-8") as f:
        for p, t in zip(preds, targets):
            f.write(f"GT\t{t}\n")
            f.write(f"PRD\t{p}\n\n")

if __name__ == "__main__":
    evaluate()