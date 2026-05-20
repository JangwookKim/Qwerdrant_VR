# data/contextual_dataset.py

import json
import torch
from torch.utils.data import Dataset

from .vocab import PAD, UNK


BOS = "<bos>"
EOS = "<eos>"


def build_contextual_char_vocab():
    """
    Character vocab for Contextual decoder.
    Adds BOS/EOS to the existing char vocab.
    """
    
    chars = list("abcdefghijklmnopqrstuvwxyz")
    chars += [" ", "'", ",", ".", ";", "?"]
    chars += list("0123456789")

    vocab = [PAD, UNK, BOS, EOS] + chars

    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}

    return char2idx, idx2char


def load_contextual_jsonl(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            samples.append({
                "input": obj["argmax"],
                "target": obj["target"],
            })

    return samples


class ContextualDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        char2idx,
        max_input_len=256,
        max_target_len=256,
    ):
        self.samples = load_contextual_jsonl(jsonl_path)
        self.char2idx = char2idx
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        self.pad_id = char2idx[PAD]
        self.unk_id = char2idx[UNK]
        self.bos_id = char2idx[BOS]
        self.eos_id = char2idx[EOS]

    def __len__(self):
        return len(self.samples)

    def encode_input(self, text):
        text = text[:self.max_input_len]

        ids = [
            self.char2idx.get(ch, self.unk_id)
            for ch in text
        ]

        attention_mask = [1] * len(ids)

        pad_len = self.max_input_len - len(ids)
        ids += [self.pad_id] * pad_len
        attention_mask += [0] * pad_len

        return ids, attention_mask

    def encode_target(self, text):
        # decoder input: <bos> + text
        # labels: text + <eos>
        text = text[: self.max_target_len - 1]

        target_ids = [
            self.char2idx.get(ch, self.unk_id)
            for ch in text
        ]

        decoder_input_ids = [self.bos_id] + target_ids
        labels = target_ids + [self.eos_id]

        pad_len = self.max_target_len - len(decoder_input_ids)

        decoder_input_ids += [self.pad_id] * pad_len
        labels += [self.pad_id] * pad_len

        return decoder_input_ids, labels

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_ids, attention_mask = self.encode_input(sample["input"])
        decoder_input_ids, labels = self.encode_target(sample["target"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }