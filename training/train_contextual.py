# training/train_contextual.py

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.contextual_decode import contextual_greedy_decode


def train_contextual_epoch(
    model,
    loader,
    optimizer,
    pad_idx,
    device,
):
    model.train()
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        reduction="none",
    )

    total_loss = 0.0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss_per_token = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        attention_mask_flat = attention_mask.reshape(-1).float()
        change_mask = batch["change_mask"].to(device).reshape(-1).float()

        same_mask = (
            (attention_mask_flat == 1)
            & (change_mask == 0)
        ).float()

        weights = torch.ones_like(loss_per_token)
        weights = weights + change_mask * 8.0
        weights = weights + same_mask * 0.5

        loss = (loss_per_token * weights * attention_mask_flat).sum() / (
            (weights * attention_mask_flat).sum() + 1e-8
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_contextual_epoch(
    model,
    loader,
    pad_idx,
    device,
):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            total_loss += loss.item()

    return total_loss / len(loader)


def preview_contextual_predictions(
    model,
    dataset,
    char2idx,
    idx2char,
    device,
    max_len,
    num_samples=3,
):
    print()
    print("=== Sample Contextual Predictions ===")

    changed_indices = [
        i for i, s in enumerate(dataset.samples)
        if s["input"] != s["target"]
    ]

    pool = changed_indices if len(changed_indices) > 0 else list(range(len(dataset)))

    indices = random.sample(
        pool,
        min(num_samples, len(pool)),
    )

    for i, idx in enumerate(indices):
        sample = dataset.samples[idx]

        noisy = sample["input"]
        target = sample["target"]

        pred = contextual_greedy_decode(
            model=model,
            noisy_text=noisy,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            max_len=max_len,
        )

        print(f"[{i}] NOISY: {noisy}")
        print(f"[{i}] PRD  : {pred}")
        print(f"[{i}] GT   : {target}")
        print()


def train_contextual(
    model,
    train_dataset,
    val_dataset,
    char2idx,
    idx2char,
    pad_idx,
    device,
    save_path,
    epochs=5,
    batch_size=32,
    lr=1e-4,
    max_len=256,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_contextual_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            pad_idx=pad_idx,
            device=device,
        )

        val_loss = validate_contextual_epoch(
            model=model,
            loader=val_loader,
            pad_idx=pad_idx,
            device=device,
        )

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        preview_contextual_predictions(
            model=model,
            dataset=val_dataset,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            max_len=max_len,
            num_samples=3,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"saved best: {save_path} (val_loss={best_val_loss:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model