from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from .engine import train_epoch, validate_epoch
from .metrics import cer, wer

def build_shard_paths(split="train", num_shards=10, shard_size=10000, data_dir="datasets"):
    paths = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size - 1
        path = f"{data_dir}/news.en.{split}.{start:08d}-{end:08d}.txt"
        paths.append(path)
    return paths

def train_decoder_one_shard(
    model,
    train_dataset,
    val_dataset,
    char2idx,
    idx2char,
    device,
    save_path,
    epochs=5,
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
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"CER={val_cer:.4f} "
            f"WER={val_wer:.4f}"
        )

        if save_metric == "loss":
            score = val_loss
        elif save_metric == "cer":
            score = val_cer
        else:
            score = val_wer

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), save_path)
            print(f"? saved best: {save_path} ({save_metric}={best_score:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model
    
    
    
import os
import torch

from data.dataset_decode import GestureDataset


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        return [x.strip().lower() for x in f if x.strip()]


def train_decoder_across_shards(
    model,
    char2idx,
    idx2char,
    gesture2idx,
    device,
    data_dir,
    num_shards,
    shard_size,
    start_shard,
    epochs_per_shard,
    batch_size,
    lr,
    max_len,
    save_dir,
    save_prefix,
    save_metric="wer",
    clip_pred_to_target_len=False,
    use_codebook_features=False,
    resume_path=None,
):
    os.makedirs(save_dir, exist_ok=True)

    current_ckpt = resume_path
    
    train_paths = build_shard_paths(
        split="train",
        num_shards=num_shards,
        shard_size=shard_size,
        data_dir=data_dir
    )

    val_paths = build_shard_paths(
        split="val",
        num_shards=num_shards,
        shard_size=shard_size,
        data_dir=data_dir
    )

    for shard_idx in range(start_shard, num_shards):
        print(f"\n========== SHARD {shard_idx:02d} ==========")

        train_path = train_paths[shard_idx]
        val_path = val_paths[shard_idx]

        print("train:", train_path)
        print("val  :", val_path)

        train_sentences = load_sentences(train_path)
        val_sentences = load_sentences(val_path)

        train_dataset = GestureDataset(
            sentences=train_sentences,
            char2idx=char2idx,
            gesture2idx=gesture2idx,
            max_len=max_len,
            use_codebook_features=use_codebook_features
        )

        val_dataset = GestureDataset(
            sentences=val_sentences,
            char2idx=char2idx,
            gesture2idx=gesture2idx,
            max_len=max_len,
            use_codebook_features=use_codebook_features
        )

        # checkpoint 이어받기
        if current_ckpt is not None:
            print("Loading checkpoint:", current_ckpt)
            state = torch.load(current_ckpt, map_location=device)
            model.load_state_dict(state)

        shard_ckpt = os.path.join(
            save_dir,
            f"{save_prefix}-shard{shard_idx:02d}-best.pt"
        )

        model = train_decoder_one_shard(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            save_path=shard_ckpt,
            epochs=epochs_per_shard,
            batch_size=batch_size,
            lr=lr,
            save_metric=save_metric,
            clip_pred_to_target_len=clip_pred_to_target_len
        )

        current_ckpt = shard_ckpt

    return model