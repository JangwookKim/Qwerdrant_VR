# inference/contextual_decode.py

import torch

from data.contextual_dataset import BOS, EOS
from data.vocab import PAD, UNK


def encode_noisy_text(text, char2idx, max_input_len):
    pad_id = char2idx[PAD]
    unk_id = char2idx[UNK]

    text = text[:max_input_len]

    ids = [char2idx.get(ch, unk_id) for ch in text]
    attention_mask = [1] * len(ids)

    pad_len = max_input_len - len(ids)

    ids += [pad_id] * pad_len
    attention_mask += [0] * pad_len

    return ids, attention_mask


def decode_ids(token_ids, idx2char):
    chars = []

    for idx in token_ids:
        idx = idx.item() if hasattr(idx, "item") else idx
        ch = idx2char[idx]

        if ch == EOS:
            break

        if ch in [PAD, BOS]:
            continue

        chars.append(ch)

    return "".join(chars)


def contextual_greedy_decode(
    model,
    noisy_text,
    char2idx,
    idx2char,
    device,
    max_input_len=256,
    max_target_len=256,
):
    model.eval()

    input_ids, attention_mask = encode_noisy_text(
        text=noisy_text,
        char2idx=char2idx,
        max_input_len=max_input_len,
    )

    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    decoder_ids = [char2idx[BOS]]

    with torch.no_grad():
        for _ in range(max_target_len - 1):
            decoder_input_ids = torch.tensor(
                [decoder_ids],
                dtype=torch.long,
            ).to(device)

            logits = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )

            next_id = int(logits[0, -1].argmax(dim=-1).item())

            decoder_ids.append(next_id)

            if idx2char[next_id] == EOS:
                break

    return decode_ids(decoder_ids, idx2char)