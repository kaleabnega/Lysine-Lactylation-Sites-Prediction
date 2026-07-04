from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WindowRecord:
    split: str
    record_name: str
    window_position: int
    label: int
    sequence: str


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    accession: str
    description: str
    sequence: str


def parse_pcbert_window_file(path: Path, split: str) -> list[WindowRecord]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if lines and lines[0].lower() == "data":
        lines = lines[1:]
    if len(lines) % 2:
        raise ValueError(f"{path} has an odd number of non-empty lines")

    records: list[WindowRecord] = []
    for offset in range(0, len(lines), 2):
        header = lines[offset]
        sequence = lines[offset + 1].upper()
        parts = header.split("|")
        if len(parts) != 3:
            raise ValueError(f"Invalid header in {path}: {header!r}")
        record_name, position_text, label_text = parts
        position = int(position_text)
        label = int(label_text)
        if sequence[position - 1] != "K":
            raise ValueError(
                f"{record_name} has {sequence[position - 1]!r}, not K, at {position}"
            )
        records.append(
            WindowRecord(
                split=split,
                record_name=record_name,
                window_position=position,
                label=label,
                sequence=sequence,
            )
        )
    return records


def parse_fasta(path: Path) -> list[ProteinRecord]:
    records: list[ProteinRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        if header is None:
            return
        sequence = "".join(sequence_parts).upper()
        if not sequence:
            return
        protein_id, accession = parse_fasta_header(header)
        records.append(
            ProteinRecord(
                protein_id=protein_id,
                accession=accession,
                description=header,
                sequence=sequence,
            )
        )

    with path.open() as fasta_file:
        for raw_line in fasta_file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        flush()

    return records


def parse_fasta_header(header: str) -> tuple[str, str]:
    first_token = header.split()[0]
    if "|" in first_token:
        parts = first_token.split("|")
        if len(parts) >= 3:
            return first_token, parts[1]
    return first_token, first_token


def map_windows(
    windows: list[WindowRecord],
    proteins: list[ProteinRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sequence_to_records: dict[str, list[WindowRecord]] = defaultdict(list)
    for record in windows:
        sequence_to_records[record.sequence].append(record)

    matches_by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for protein in proteins:
        for window_sequence in sequence_to_records:
            start = protein.sequence.find(window_sequence)
            while start != -1:
                central_pos = start + 26
                matches_by_sequence[window_sequence].append(
                    {
                        "protein_id": protein.protein_id,
                        "accession": protein.accession,
                        "description": protein.description,
                        "match_start_1based": start + 1,
                        "match_end_1based": start + len(window_sequence),
                        "central_lysine_full_position_1based": central_pos,
                    }
                )
                start = protein.sequence.find(window_sequence, start + 1)

    rows: list[dict[str, object]] = []
    unique_rows: list[dict[str, object]] = []
    for record in windows:
        matches = matches_by_sequence.get(record.sequence, [])
        status = classify_matches(matches)
        base = {
            "split": record.split,
            "record_name": record.record_name,
            "label": record.label,
            "window_position": record.window_position,
            "window_sequence": record.sequence,
            "match_status": status,
            "match_count": len(matches),
        }

        if not matches:
            rows.append(
                {
                    **base,
                    "protein_id": "",
                    "accession": "",
                    "match_start_1based": "",
                    "match_end_1based": "",
                    "central_lysine_full_position_1based": "",
                    "description": "",
                }
            )
            continue

        for match in matches:
            row = {**base, **match}
            rows.append(row)
            if status == "unique_match":
                unique_rows.append(row)

    return rows, unique_rows


def classify_matches(matches: list[dict[str, object]]) -> str:
    if not matches:
        return "no_match"
    unique_sites = {
        (
            match["accession"],
            match["central_lysine_full_position_1based"],
        )
        for match in matches
    }
    if len(unique_sites) == 1:
        return "unique_match"
    return "multiple_matches"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]], protein_count: int) -> None:
    status_counts = Counter(row["match_status"] for row in rows)
    unique_window_status = {}
    for row in rows:
        unique_window_status.setdefault(row["window_sequence"], row["match_status"])
    unique_counts = Counter(unique_window_status.values())

    lines = [
        "# Window-To-Proteome Mapping Summary",
        "",
        f"Proteins searched: {protein_count}",
        f"Benchmark rows reported: {len(rows)}",
        "",
        "## Row-Level Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Unique-Window Status Counts", ""])
    for status, count in sorted(unique_counts.items()):
        lines.append(f"- {status}: {count}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map PCBert/DeepKla 51-aa windows to full proteome FASTA records."
    )
    parser.add_argument(
        "--train-windows",
        type=Path,
        default=Path("baselines/PCBert-Kla-original/data/train.csv"),
    )
    parser.add_argument(
        "--test-windows",
        type=Path,
        default=Path("baselines/PCBert-Kla-original/data/test.csv"),
    )
    parser.add_argument("--proteome-fasta", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/structure_aware_kla/results/window_mapping"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = [
        *parse_pcbert_window_file(args.train_windows, split="train"),
        *parse_pcbert_window_file(args.test_windows, split="test"),
    ]
    proteins = parse_fasta(args.proteome_fasta)
    rows, unique_rows = map_windows(windows, proteins)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_window_matches.csv", rows)
    write_csv(args.output_dir / "unique_site_matches.csv", unique_rows)
    write_summary(args.output_dir / "summary.md", rows, protein_count=len(proteins))
    print(f"proteins searched: {len(proteins)}")
    print(f"benchmark rows reported: {len(rows)}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
