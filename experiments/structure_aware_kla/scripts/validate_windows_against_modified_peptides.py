from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from map_windows_to_proteome import WindowRecord, parse_pcbert_window_file


@dataclass(frozen=True)
class ModifiedPeptide:
    plain_sequence: str
    modified_index: int
    source_row: int


def parse_modified_sequence(value: str, source_row: int) -> ModifiedPeptide | None:
    cleaned = re.sub(r"[^A-Z()0-9]", "", value.upper())
    marker = "(1)"
    marker_start = cleaned.find(marker)
    if marker_start == -1:
        return None

    plain_before_marker = cleaned[:marker_start].replace(marker, "")
    modified_index = len(plain_before_marker) - 1
    plain_sequence = cleaned.replace(marker, "")
    if modified_index < 0 or modified_index >= len(plain_sequence):
        return None
    if plain_sequence[modified_index] != "K":
        return None

    return ModifiedPeptide(
        plain_sequence=plain_sequence,
        modified_index=modified_index,
        source_row=source_row,
    )


def load_modified_peptides(path: Path, column: str) -> list[ModifiedPeptide]:
    peptides: list[ModifiedPeptide] = []
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise SystemExit(f"Column {column!r} not found. Available: {available}")
        for row_number, row in enumerate(reader, start=2):
            peptide = parse_modified_sequence(row[column], source_row=row_number)
            if peptide is not None:
                peptides.append(peptide)
    return peptides


def matching_peptides(
    window: WindowRecord,
    peptides: list[ModifiedPeptide],
) -> list[ModifiedPeptide]:
    central_index = window.window_position - 1
    matches: list[ModifiedPeptide] = []
    for peptide in peptides:
        start = window.sequence.find(peptide.plain_sequence)
        while start != -1:
            if start + peptide.modified_index == central_index:
                matches.append(peptide)
                break
            start = window.sequence.find(peptide.plain_sequence, start + 1)
    return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate benchmark windows against source-table modified peptides. "
            "This is useful when protein accessions changed between the source "
            "supplement and current UniProt."
        )
    )
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--site-table-csv", type=Path, required=True)
    parser.add_argument("--modified-sequence-column", default="Modified sequence")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = parse_pcbert_window_file(args.windows, split=args.windows.stem)
    peptides = load_modified_peptides(args.site_table_csv, args.modified_sequence_column)

    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for window in windows:
        matches = matching_peptides(window, peptides)
        status = "matched" if matches else "unmatched"
        key = f"label_{window.label}_{status}"
        counts[key] += 1
        rows.append(
            {
                "record_name": window.record_name,
                "label": window.label,
                "window_position": window.window_position,
                "window_sequence": window.sequence,
                "match_status": status,
                "match_count": len(matches),
                "source_rows": ";".join(str(match.source_row) for match in matches),
                "matched_peptides": ";".join(match.plain_sequence for match in matches),
            }
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"windows: {len(windows)}")
    print(f"modified_peptides: {len(peptides)}")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    if args.output is not None:
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
