from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from .engine import train_epoch, validate_epoch
from .metrics import cer, wer


def train_decoder(
    model,
    train_dataset,
    val_dataset,
    char2idx,
    idx2char,
    device,
    save_path,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    save_metric="wer",
    clip_pred_to_target_len=False
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_score = float("inf")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            char_pad=char2idx["<pad>"],
            device=device
        )

        val_loss, val_cer, val_wer = validate_epoch(
            model=model,
            loader=val_loader,
            char_pad=char2idx["<pad>"],
            idx2char=idx2char,
            cer_fn=cer,
            wer_fn=wer,
            device=device,
            clip_pred_to_target_len=clip_pred_to_target_len
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_CER={val_cer:.4f} "
            f"val_WER={val_wer:.4f}"
        )

        if save_metric == "loss":
            current_score = val_loss
        elif save_metric == "cer":
            current_score = val_cer
        else:
            current_score = val_wer

        if current_score < best_score:
            best_score = current_score
            torch.save(model.state_dict(), save_path)
            print(f"best model saved: {save_path} ({save_metric}={best_score:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model