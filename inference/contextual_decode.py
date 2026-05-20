# inference/contextual_decode.py

import torch

from data.contextual_dataset import is_editable_char
from data.vocab import PAD, UNK


def encode_noisy_text(text, char2idx, max_len):
    pad_id = char2idx[PAD]
    unk_id = char2idx[UNK]

    text = text[:max_len]

    ids = [char2idx.get(ch, unk_id) for ch in text]
    attention_mask = [1] * len(ids)

    pad_len = max_len - len(ids)

    ids += [pad_id] * pad_len
    attention_mask += [0] * pad_len

    return ids, attention_mask


def contextual_greedy_decode(
    model,
    noisy_text,
    char2idx,
    idx2char,
    device,
    max_len=256,
    copy_non_alpha=True,
    correction_threshold=0.80,
):
    """
    Same-length contextual decoding.

    - Output length is always the same as noisy_text length.
    - If copy_non_alpha=True, non alphabet characters are copied
      directly from the input.
    """

    model.eval()

    original_len = min(len(noisy_text), max_len)

    input_ids, attention_mask = encode_noisy_text(
        text=noisy_text,
        char2idx=char2idx,
        max_len=max_len,
    )

    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    #pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
    probs = torch.softmax(logits, dim=-1)
    pred_probs, pred_ids = probs.max(dim=-1)

    pred_probs = pred_probs[0].cpu().tolist()
    pred_ids = pred_ids[0].cpu().tolist()

    chars = []

    for i in range(original_len):
        src_ch = noisy_text[i]

        if copy_non_alpha and not is_editable_char(src_ch):
            chars.append(src_ch)
            continue

        pred_ch = idx2char[pred_ids[i]]
        pred_prob = pred_probs[i]

        if pred_ch in [PAD, UNK]:
            chars.append(src_ch)
            continue

        if pred_ch != src_ch and pred_prob < correction_threshold:
            chars.append(src_ch)
            continue

        chars.append(pred_ch)

    return "".join(chars)


def contextual_batch_decode(
    model,
    batch_input_ids,
    batch_attention_mask,
    noisy_texts,
    char2idx,
    idx2char,
    device,
    copy_non_alpha=True,
    correction_threshold=0.80,
):
    """
    Batch version for evaluation.

    noisy_texts are needed to force-copy non alphabet characters
    and preserve exact output length.
    """

    model.eval()

    input_ids = batch_input_ids.to(device)
    attention_mask = batch_attention_mask.to(device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    #pred_ids = logits.argmax(dim=-1).cpu().tolist()
    probs = torch.softmax(logits, dim=-1)
    pred_probs, pred_ids = probs.max(dim=-1)

    pred_probs = pred_probs.cpu().tolist()
    pred_ids = pred_ids.cpu().tolist()

    outputs = []

    for b, noisy_text in enumerate(noisy_texts):
        original_len = min(len(noisy_text), len(pred_ids[b]))

        chars = []

        for i in range(original_len):
            src_ch = noisy_text[i]

            if copy_non_alpha and not is_editable_char(src_ch):
                chars.append(src_ch)
                continue

            pred_ch = idx2char[pred_ids[b][i]]
            pred_prob = pred_probs[b][i]

            if pred_ch in [PAD, UNK]:
                chars.append(src_ch)
                continue

            if pred_ch != src_ch and pred_prob < correction_threshold:
                chars.append(src_ch)
                continue

            chars.append(pred_ch)

        outputs.append("".join(chars))

    return outputs