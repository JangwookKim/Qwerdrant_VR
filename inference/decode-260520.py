import torch

from data.vocab import PAD
from data.codebook import build_codebook_vectors, get_codebook_vector, codebook


def encode_gesture_string(
    gesture_text,
    gesture2idx,
    max_len,
    keep_unknown=True,
):
    """
    Convert a gesture string into input ids.

    If keep_unknown=True:
        Unknown characters are fed as PAD to the model,
        but their original characters are stored in raw_chars
        so that they can be copied back to the final output.
    """

    gesture_text = gesture_text[:max_len]

    pad_id = gesture2idx[PAD]

    ids = []
    raw_chars = []

    for ch in gesture_text:
        if ch in gesture2idx:
            ids.append(gesture2idx[ch])
            raw_chars.append(None)
        else:
            ids.append(pad_id)
            raw_chars.append(ch if keep_unknown else None)

    pad_len = max_len - len(ids)

    ids += [pad_id] * pad_len
    raw_chars += [None] * pad_len

    return ids, raw_chars


def build_codebook_feature_sequence(
    gesture_text,
    max_len,
    codebook_dim=26,
):
    """
    Build [max_len, codebook_dim] codebook feature sequence
    for a single gesture string.
    """

    gesture_text = gesture_text[:max_len]

    codebook_vecs = build_codebook_vectors(codebook)

    cb = [
        get_codebook_vector(ch, codebook_vecs)
        for ch in gesture_text
    ]

    pad_len = max_len - len(cb)
    cb += [[0.0] * codebook_dim] * pad_len

    return cb


def ids_to_text(
    token_ids,
    idx2char,
    max_chars=None,
    raw_chars=None,
):
    """
    Convert predicted token ids to text.

    If raw_chars is given, non-None entries are copied directly
    instead of using model prediction.
    """

    chars = []

    for i, idx in enumerate(token_ids):
        if max_chars is not None and i >= max_chars:
            break

        if raw_chars is not None and i < len(raw_chars):
            if raw_chars[i] is not None:
                chars.append(raw_chars[i])
                continue

        idx = idx.item() if hasattr(idx, "item") else idx
        ch = idx2char[idx]

        if ch == PAD:
            break

        chars.append(ch)

    return "".join(chars)


def forward_model_for_decode(
    model,
    input_ids,
    codebook_vectors=None,
):
    if codebook_vectors is not None:
        return model(input_ids, codebook_vectors)

    return model(input_ids)


def decode_gesture(
    model,
    gesture_text,
    gesture2idx,
    idx2char,
    device,
    max_len=256,
    codebook_aware=False,
    keep_unknown=True,
):
    """
    Decode a single gesture string.

    - Known gesture tokens are decoded by the model.
    - Unknown characters can be copied to output unchanged
      when keep_unknown=True.
    """

    model.eval()

    ids, raw_chars = encode_gesture_string(
        gesture_text=gesture_text,
        gesture2idx=gesture2idx,
        max_len=max_len,
        keep_unknown=keep_unknown,
    )

    x = torch.tensor([ids], dtype=torch.long).to(device)

    codebook_vectors = None
    if codebook_aware:
        cb = build_codebook_feature_sequence(
            gesture_text=gesture_text,
            max_len=max_len,
            codebook_dim=26,
        )
        codebook_vectors = torch.tensor([cb], dtype=torch.float).to(device)

    with torch.no_grad():
        logits = forward_model_for_decode(
            model=model,
            input_ids=x,
            codebook_vectors=codebook_vectors,
        )

    pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    return ids_to_text(
        token_ids=pred_ids,
        idx2char=idx2char,
        raw_chars=raw_chars if keep_unknown else None,
    )


def decode_batch(
    model,
    batch_input_ids,
    idx2char,
    device,
    target_lengths=None,
    codebook_vectors=None,
):
    """
    Decode a batch.

    This is mainly used for evaluation.
    Batch decoding does not apply raw character pass-through,
    because dataset samples are already encoded/padded tensors.
    """

    model.eval()

    x = batch_input_ids.to(device)

    if codebook_vectors is not None:
        codebook_vectors = codebook_vectors.to(device)

    with torch.no_grad():
        logits = forward_model_for_decode(
            model=model,
            input_ids=x,
            codebook_vectors=codebook_vectors,
        )

    pred_ids = logits.argmax(dim=-1).cpu()

    texts = []

    for i, seq in enumerate(pred_ids):
        max_chars = None

        if target_lengths is not None:
            max_chars = target_lengths[i]

        text = ids_to_text(
            token_ids=seq,
            idx2char=idx2char,
            max_chars=max_chars,
        )

        texts.append(text)

    return texts
    

import torch.nn.functional as F


def get_decoder_logits(
    model,
    batch_input_ids,
    device,
    codebook_vectors=None,
):
    """
    Return raw decoder logits.

    output:
        logits: [batch, seq_len, char_vocab]
    """
    model.eval()

    x = batch_input_ids.to(device)

    if codebook_vectors is not None:
        codebook_vectors = codebook_vectors.to(device)

    with torch.no_grad():
        logits = forward_model_for_decode(
            model=model,
            input_ids=x,
            codebook_vectors=codebook_vectors,
        )

    return logits


def logits_to_topk(
    logits,
    idx2char,
    k=5,
    target_lengths=None,
    remove_pad=True,
):
    """
    Convert logits to top-k character candidates.

    return:
        batch_topk: list[list[list[dict]]]

    Shape concept:
        batch_topk[batch_i][time_t] =
            [
                {"char": "e", "prob": 0.62, "idx": 6},
                {"char": "a", "prob": 0.14, "idx": 2},
                ...
            ]
    """
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

    top_probs = top_probs.cpu()
    top_ids = top_ids.cpu()

    batch_topk = []

    for b in range(top_ids.size(0)):
        seq_topk = []

        seq_len = top_ids.size(1)
        if target_lengths is not None:
            seq_len = min(seq_len, target_lengths[b])

        for t in range(seq_len):
            candidates = []

            for j in range(k):
                idx = int(top_ids[b, t, j].item())
                ch = idx2char[idx]

                if remove_pad and ch == "<pad>":
                    continue

                candidates.append({
                    "char": ch,
                    "idx": idx,
                    "prob": float(top_probs[b, t, j].item()),
                })

            seq_topk.append(candidates)

        batch_topk.append(seq_topk)

    return batch_topk


def logits_to_argmax_texts(
    logits,
    idx2char,
    target_lengths=None,
):
    pred_ids = logits.argmax(dim=-1).cpu()

    texts = []

    for i, seq in enumerate(pred_ids):
        max_chars = None
        if target_lengths is not None:
            max_chars = target_lengths[i]

        text = ids_to_text(
            token_ids=seq,
            idx2char=idx2char,
            max_chars=max_chars,
        )

        texts.append(text)

    return texts