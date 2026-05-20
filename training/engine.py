import torch
import torch.nn as nn
from tqdm import tqdm


def ids_to_text_with_limit(seq, idx2char, max_chars=None):
    chars = []
    for token in seq:
        token = token.item() if hasattr(token, "item") else token
        ch = idx2char[token]
        if ch == "<pad>":
            break
        chars.append(ch)

    text = "".join(chars)

    if max_chars is not None:
        text = text[:max_chars]

    return text

def forward_model(model, batch, device):
    input_ids = batch["input_ids"].to(device)

    if "codebook_vectors" in batch:
        codebook_vectors = batch["codebook_vectors"].to(device)
        return model(input_ids, codebook_vectors)

    return model(input_ids)

def train_epoch(model, loader, optimizer, char_pad, device):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=char_pad)

    total_loss = 0.0

    for batch in tqdm(loader):
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        logits = forward_model(model, batch, device)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(
    model,
    loader,
    char_pad,
    idx2char,
    cer_fn,
    wer_fn,
    device,
    clip_pred_to_target_len=False
):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=char_pad)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits = forward_model(model, batch, device)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()

            pred_ids = logits.argmax(dim=-1).cpu()
            y_cpu = y.cpu()

            for pred_seq, tgt_seq in zip(pred_ids, y_cpu):
                target_text = ids_to_text_with_limit(tgt_seq, idx2char)

                if clip_pred_to_target_len:
                    pred_text = ids_to_text_with_limit(
                        pred_seq,
                        idx2char,
                        max_chars=len(target_text)
                    )
                else:
                    pred_text = ids_to_text_with_limit(pred_seq, idx2char)

                all_preds.append(pred_text)
                all_targets.append(target_text)

    avg_loss = total_loss / len(loader)
    val_cer = cer_fn(all_preds, all_targets)
    val_wer = wer_fn(all_preds, all_targets)

    return avg_loss, val_cer, val_wer