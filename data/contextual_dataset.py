# data/contextual_dataset.py

import json
import torch
from torch.utils.data import Dataset

from .vocab import PAD, UNK


def build_contextual_char_vocab():
    chars = list("abcdefghijklmnopqrstuvwxyz")
    chars += [" ", "'", ",", ".", ";", "?"]
    chars += list("0123456789")

    vocab = [PAD, UNK] + chars

    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}

    return char2idx, idx2char


def is_editable_char(ch):
    return "a" <= ch <= "z"


def load_contextual_jsonl(path, require_same_length=True):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            noisy = obj["argmax"]
            target = obj["target"]

            if require_same_length and len(noisy) != len(target):
                continue

            min_len = min(len(noisy), len(target))
            noisy = noisy[:min_len]
            target = target[:min_len]

            samples.append({
                "input": noisy,
                "target": target,
            })

    return samples


class ContextualDataset(Dataset):
    """
    Same-length contextual correction dataset.

    input:
        noisy text

    label:
        corrected text, same length

    editable_mask:
        1 for alphabet a-z positions
        0 for spaces, punctuation, digits, etc.
    """

    def __init__(
        self,
        jsonl_path,
        char2idx,
        max_len=256,
        require_same_length=True,
    ):
        self.samples = load_contextual_jsonl(
            jsonl_path,
            require_same_length=require_same_length,
        )

        self.char2idx = char2idx
        self.max_len = max_len

        self.pad_id = char2idx[PAD]
        self.unk_id = char2idx[UNK]

    def __len__(self):
        return len(self.samples)

    def encode_pair(self, noisy, target):
        noisy = noisy[:self.max_len]
        target = target[:self.max_len]

        input_ids = []
        labels = []
        attention_mask = []
        editable_mask = []
        change_mask = []

        for n_ch, t_ch in zip(noisy, target):
            input_ids.append(self.char2idx.get(n_ch, self.unk_id))
            labels.append(self.char2idx.get(t_ch, self.unk_id))
            attention_mask.append(1)
            editable_mask.append(1 if is_editable_char(n_ch) else 0)
            change_mask.append(1 if is_editable_char(n_ch) and n_ch != t_ch else 0)

        pad_len = self.max_len - len(input_ids)

        input_ids += [self.pad_id] * pad_len
        labels += [self.pad_id] * pad_len
        attention_mask += [0] * pad_len
        editable_mask += [0] * pad_len
        change_mask += [0] * pad_len

        return input_ids, labels, attention_mask, editable_mask, change_mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_ids, labels, attention_mask, editable_mask, change_mask = self.encode_pair(
            sample["input"],
            sample["target"],
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "editable_mask": torch.tensor(editable_mask, dtype=torch.long),
            "change_mask": torch.tensor(change_mask, dtype=torch.float),
        }