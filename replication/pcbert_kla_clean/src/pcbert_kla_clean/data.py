from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KlaRecord:
    name: str
    position: int
    label: int
    sequence: str


@dataclass(frozen=True)
class KlaSplit:
    records: list[KlaRecord]
    feature_names: list[str]
    features: np.ndarray

    @property
    def names(self) -> list[str]:
        return [record.name for record in self.records]

    @property
    def sequences(self) -> list[str]:
        return [record.sequence for record in self.records]

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([record.label for record in self.records], dtype=np.int64)


def parse_pcbert_sequence_file(path: str | Path) -> list[KlaRecord]:
    """Parse the upstream FASTA-like files named train.csv/test.csv.

    The files are not conventional CSV. They contain a first line named
    "data", followed by alternating header and sequence lines:
    Protein 0|26|1
    SEQUENCE...
    """
    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")

    if lines[0].lower() == "data":
        lines = lines[1:]

    if len(lines) % 2 != 0:
        raise ValueError(f"{path} has an odd number of header/sequence lines")

    records: list[KlaRecord] = []
    for offset in range(0, len(lines), 2):
        header = lines[offset]
        sequence = lines[offset + 1].upper()
        parts = header.split("|")
        if len(parts) != 3:
            raise ValueError(f"Invalid header at {path}:{offset + 2}: {header!r}")

        name, position_text, label_text = parts
        position = int(position_text)
        label = int(label_text)
        if label not in {0, 1}:
            raise ValueError(f"Invalid label {label} in {header!r}")
        if position < 1 or position > len(sequence):
            raise ValueError(f"Invalid 1-based position {position} in {header!r}")
        if sequence[position - 1] != "K":
            raise ValueError(
                f"Expected lysine K at position {position} for {name}, "
                f"found {sequence[position - 1]!r}"
            )

        records.append(
            KlaRecord(name=name, position=position, label=label, sequence=sequence)
        )

    return records


def load_feature_file(path: str | Path) -> tuple[list[str], list[str], np.ndarray]:
    path = Path(path)
    table = pd.read_csv(path)
    if "ProteinName" not in table.columns:
        raise ValueError(f"{path} must contain a ProteinName column")

    names = table["ProteinName"].astype(str).tolist()
    feature_names = [column for column in table.columns if column != "ProteinName"]
    features = table[feature_names].to_numpy(dtype=np.float32)
    return names, feature_names, features


def load_split(sequence_path: str | Path, feature_path: str | Path) -> KlaSplit:
    records = parse_pcbert_sequence_file(sequence_path)
    feature_names_in_rows, feature_names, features = load_feature_file(feature_path)
    record_names = [record.name for record in records]

    if record_names != feature_names_in_rows:
        raise ValueError(
            "Feature rows do not align with sequence rows. "
            "This replication expects identical ProteinName ordering."
        )
    if features.shape[0] != len(records):
        raise ValueError("Feature and sequence row counts differ")

    return KlaSplit(records=records, feature_names=feature_names, features=features)


def describe_split(split: KlaSplit) -> dict[str, int | dict[int, int]]:
    labels, counts = np.unique(split.labels, return_counts=True)
    class_counts = {int(label): int(count) for label, count in zip(labels, counts)}
    sequence_counts = pd.Series(split.sequences).value_counts()
    return {
        "records": len(split.records),
        "feature_dim": int(split.features.shape[1]),
        "class_counts": class_counts,
        "unique_sequences": int(sequence_counts.shape[0]),
        "duplicate_sequence_rows": int((sequence_counts - 1).clip(lower=0).sum()),
    }


def train_test_overlap_report(train: KlaSplit, test: KlaSplit) -> dict[str, int]:
    train_labels_by_sequence: dict[str, set[int]] = {}
    for record in train.records:
        train_labels_by_sequence.setdefault(record.sequence, set()).add(record.label)

    overlap_rows = 0
    same_label_rows = 0
    different_label_rows = 0
    unique_overlap_sequences: set[str] = set()

    for record in test.records:
        train_labels = train_labels_by_sequence.get(record.sequence)
        if train_labels is None:
            continue
        overlap_rows += 1
        unique_overlap_sequences.add(record.sequence)
        if record.label in train_labels:
            same_label_rows += 1
        else:
            different_label_rows += 1

    return {
        "overlap_test_rows": overlap_rows,
        "unique_overlap_sequences": len(unique_overlap_sequences),
        "same_label_test_rows": same_label_rows,
        "different_label_test_rows": different_label_rows,
    }

