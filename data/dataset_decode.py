import torch
from torch.utils.data import Dataset

from .codebook import encode_text, codebook, build_codebook_vectors, get_codebook_vector
from .vocab import PAD


class GestureDataset(Dataset):

    def __init__(
        self,
        sentences,
        char2idx,
        gesture2idx,
        max_len=256,
        use_codebook_features=False
    ):
        self.sentences = sentences
        self.char2idx = char2idx
        self.gesture2idx = gesture2idx
        self.max_len = max_len
        self.use_codebook_features = use_codebook_features
        self.codebook_vectors = build_codebook_vectors(codebook)

    def __len__(self):
        return len(self.sentences)

    def encode(self, text):
        text = text[:self.max_len]

        gesture = encode_text(text)

        g = [self.gesture2idx[c] for c in gesture]
        c = [self.char2idx.get(x, 1) for x in text]

        if self.use_codebook_features:
            cb = [get_codebook_vector(x, self.codebook_vectors) for x in gesture]

        pad_len = self.max_len - len(g)

        g += [self.gesture2idx[PAD]] * pad_len
        c += [self.char2idx[PAD]] * pad_len

        if self.use_codebook_features:
            cb += [[0.0] * 26] * pad_len
            return g, cb, c

        return g, c

    def __getitem__(self, idx):
        if self.use_codebook_features:
            g, cb, c = self.encode(self.sentences[idx])
            return {
                "input_ids": torch.tensor(g, dtype=torch.long),
                "codebook_vectors": torch.tensor(cb, dtype=torch.float),
                "labels": torch.tensor(c, dtype=torch.long)
            }

        g, c = self.encode(self.sentences[idx])
        return {
            "input_ids": torch.tensor(g, dtype=torch.long),
            "labels": torch.tensor(c, dtype=torch.long)
        }