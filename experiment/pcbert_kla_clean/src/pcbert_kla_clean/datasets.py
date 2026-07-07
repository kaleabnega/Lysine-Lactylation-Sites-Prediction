from __future__ import annotations

import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcbert_kla_clean.data import KlaSplit


class KlaDataset:
    def __init__(
        self,
        sequences: list[str],
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.sequences = sequences
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray, np.float32]:
        return self.sequences[index], self.features[index], self.labels[index]


def format_sequence(sequence: str, sequence_format: str) -> str:
    if sequence_format == "raw":
        return sequence
    if sequence_format == "protbert_spaced":
        return " ".join(sequence)
    if sequence_format == "prott5_spaced":
        return " ".join(re.sub(r"[UZOB]", "X", sequence))
    raise ValueError(f"Unsupported sequence format: {sequence_format}")


def make_collate_fn(tokenizer, max_length: int, sequence_format: str):
    def collate(batch):
        sequences, features, labels = zip(*batch)
        encoded = tokenizer(
            [format_sequence(sequence, sequence_format) for sequence in sequences],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {
            "encoded": encoded,
            "features": torch.tensor(np.asarray(features), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    return collate


def make_loader(
    split: KlaSplit,
    indices: np.ndarray,
    scaled_features: np.ndarray,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    max_length: int,
    sequence_format: str,
) -> DataLoader:
    dataset = KlaDataset(
        sequences=[split.sequences[index] for index in indices],
        features=scaled_features[indices],
        labels=split.labels[indices],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(
            tokenizer,
            max_length=max_length,
            sequence_format=sequence_format,
        ),
    )
