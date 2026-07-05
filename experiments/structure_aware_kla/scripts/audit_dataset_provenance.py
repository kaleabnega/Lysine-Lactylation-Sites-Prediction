from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from map_windows_to_proteome import (
    WindowRecord,
    map_windows,
    parse_fasta,
    parse_pcbert_window_file,
)
from validate_windows_against_modified_peptides import (
    load_modified_peptides,
    matching_peptides,
)


def parse_deepkla_fasta(path: Path, split: str) -> list[WindowRecord]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 2:
        raise ValueError(f"{path} has an odd number of non-empty lines")

    records: list[WindowRecord] = []
    for offset in range(0, len(lines), 2):
        header = lines[offset]
        sequence = lines[offset + 1].upper()
        if not header.startswith(">"):
            raise ValueError(f"Invalid DeepKla header in {path}: {header!r}")
        label = int(header[1:])
        window_position = 26
        if len(sequence) != 51 or sequence[window_position - 1] != "K":
            raise ValueError(f"Invalid 51-aa lysine-centered window in {path}: {sequence}")
        records.append(
            WindowRecord(
                split=split,
                record_name=f"{split}_{offset // 2}",
                window_position=window_position,
                label=label,
                sequence=sequence,
            )
        )
    return records


def multiset_summary(left: list[WindowRecord], right: list[WindowRecord]) -> dict[str, object]:
    left_counter = Counter((record.sequence, record.label) for record in left)
    right_counter = Counter((record.sequence, record.label) for record in right)
    return {
        "left_records": len(left),
        "right_records": len(right),
        "sequence_label_multiset_equal": left_counter == right_counter,
        "left_only_sequence_label_pairs": sum((left_counter - right_counter).values()),
        "right_only_sequence_label_pairs": sum((right_counter - left_counter).values()),
        "left_label_counts": dict(Counter(record.label for record in left)),
        "right_label_counts": dict(Counter(record.label for record in right)),
        "left_unique_sequences": len({record.sequence for record in left}),
        "right_unique_sequences": len({record.sequence for record in right}),
    }


def source_table_summary(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(path.open(newline="")))
    keys = [
        (row["Protein accession"], row["Position"], row["Amino acid"])
        for row in rows
        if row.get("Protein accession") and row.get("Position")
    ]
    return {
        "data_rows": len(rows),
        "nonempty_accession_position_rows": len(keys),
        "unique_accession_position_amino_acid_sites": len(set(keys)),
        "unique_protein_accessions": len({key[0] for key in keys}),
        "amino_acid_counts": dict(Counter(key[2] for key in keys)),
        "duplicate_site_keys": sum(count > 1 for count in Counter(keys).values()),
    }


def peptide_evidence_summary(
    windows: list[WindowRecord],
    source_table: Path,
    modified_sequence_column: str,
) -> dict[str, object]:
    peptides = load_modified_peptides(source_table, modified_sequence_column)
    counts: Counter[str] = Counter()
    for window in windows:
        status = "matched" if matching_peptides(window, peptides) else "unmatched"
        counts[f"label_{window.label}_{status}"] += 1
    return {
        "windows": len(windows),
        "modified_peptides": len(peptides),
        "counts": dict(sorted(counts.items())),
    }


def table_s1_window_overlap(
    source_table: Path,
    proteome_fasta: Path,
    benchmark_windows: list[WindowRecord],
) -> dict[str, object]:
    proteins = {record.accession: record.sequence for record in parse_fasta(proteome_fasta)}
    rows = list(csv.DictReader(source_table.open(newline="")))
    source_accessions = {row["Protein accession"] for row in rows if row.get("Protein accession")}
    benchmark_sequences = {record.sequence for record in benchmark_windows}

    valid_k_sites = 0
    wrong_amino_acid = 0
    out_of_range = 0
    reconstructed_windows: list[str] = []
    for row in rows:
        accession = row.get("Protein accession", "")
        position_text = row.get("Position", "")
        if not accession or not position_text:
            continue
        sequence = proteins.get(accession)
        if sequence is None:
            continue
        position = int(float(position_text))
        if position < 1 or position > len(sequence):
            out_of_range += 1
            continue
        if sequence[position - 1] != "K":
            wrong_amino_acid += 1
            continue
        valid_k_sites += 1
        start = position - 26
        end = position + 25
        if start >= 0 and end <= len(sequence):
            reconstructed_windows.append(sequence[start:end])

    return {
        "proteins_searched": len(proteins),
        "table_s1_accessions": len(source_accessions),
        "table_s1_accessions_present": sum(1 for accession in source_accessions if accession in proteins),
        "valid_k_sites": valid_k_sites,
        "wrong_amino_acid": wrong_amino_acid,
        "out_of_range": out_of_range,
        "reconstructed_51aa_windows": len(reconstructed_windows),
        "unique_reconstructed_51aa_windows": len(set(reconstructed_windows)),
        "overlap_with_unique_benchmark_windows": len(set(reconstructed_windows) & benchmark_sequences),
    }


def mapping_summary(windows: list[WindowRecord], proteome_fasta: Path) -> dict[str, object]:
    proteins = parse_fasta(proteome_fasta)
    rows, _ = map_windows(windows, proteins)
    row_counts = Counter(row["match_status"] for row in rows)
    unique_window_status: dict[str, str] = {}
    for row in rows:
        unique_window_status.setdefault(str(row["window_sequence"]), str(row["match_status"]))
    return {
        "proteins_searched": len(proteins),
        "row_status_counts": dict(sorted(row_counts.items())),
        "unique_window_status_counts": dict(sorted(Counter(unique_window_status.values()).items())),
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Dataset Provenance Audit",
        "",
        "## DeepKla Versus PCBert-Kla",
        "",
        "These checks compare sequence-label multisets, ignoring superficial header format differences.",
        "",
        "```json",
        json.dumps(report["deepkla_vs_pcbert"], indent=2, sort_keys=True),
        "```",
        "",
        "## Meng 2021 Table S1 Integrity",
        "",
        "```json",
        json.dumps(report["meng_table_s1"], indent=2, sort_keys=True),
        "```",
        "",
        "## Meng 2021 Peptide Evidence Against Benchmark Windows",
        "",
        "```json",
        json.dumps(report["meng_peptide_evidence"], indent=2, sort_keys=True),
        "```",
    ]

    if "meng_table_s1_window_overlap" in report:
        lines.extend(
            [
                "",
                "## Meng 2021 Reconstructed Window Overlap",
                "",
                "```json",
                json.dumps(report["meng_table_s1_window_overlap"], indent=2, sort_keys=True),
                "```",
            ]
        )

    if "botrytis_mapping" in report:
        lines.extend(
            [
                "",
                "## Botrytis Proteome Mapping",
                "",
                "```json",
                json.dumps(report["botrytis_mapping"], indent=2, sort_keys=True),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "PCBert-Kla and public DeepKla use the same benchmark sequence-label "
                "multisets. Meng et al. 2021 Table S1 is internally consistent and "
                "matches the reported rice lactylome scale, but it does not currently "
                "reconstruct the public benchmark training windows by exact sequence."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DeepKla/PCBert-Kla dataset provenance.")
    parser.add_argument("--pcbert-train", type=Path, required=True)
    parser.add_argument("--pcbert-test", type=Path, required=True)
    parser.add_argument("--deepkla-train", type=Path, required=True)
    parser.add_argument("--deepkla-test", type=Path, required=True)
    parser.add_argument("--meng-table-s1", type=Path, required=True)
    parser.add_argument("--rice-proteome-fasta", type=Path)
    parser.add_argument("--botrytis-proteome-fasta", type=Path)
    parser.add_argument("--modified-sequence-column", default="Modified sequence")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pcbert_train = parse_pcbert_window_file(args.pcbert_train, split="train")
    pcbert_test = parse_pcbert_window_file(args.pcbert_test, split="test")
    deepkla_train = parse_deepkla_fasta(args.deepkla_train, split="train")
    deepkla_test = parse_deepkla_fasta(args.deepkla_test, split="test")

    report: dict[str, object] = {
        "deepkla_vs_pcbert": {
            "train": multiset_summary(pcbert_train, deepkla_train),
            "test": multiset_summary(pcbert_test, deepkla_test),
        },
        "meng_table_s1": source_table_summary(args.meng_table_s1),
        "meng_peptide_evidence": {
            "train": peptide_evidence_summary(
                pcbert_train, args.meng_table_s1, args.modified_sequence_column
            ),
            "test": peptide_evidence_summary(
                pcbert_test, args.meng_table_s1, args.modified_sequence_column
            ),
        },
    }

    if args.rice_proteome_fasta is not None:
        report["meng_table_s1_window_overlap"] = table_s1_window_overlap(
            args.meng_table_s1, args.rice_proteome_fasta, pcbert_train
        )

    if args.botrytis_proteome_fasta is not None:
        report["botrytis_mapping"] = {
            "train": mapping_summary(pcbert_train, args.botrytis_proteome_fasta),
            "test": mapping_summary(pcbert_test, args.botrytis_proteome_fasta),
            "combined": mapping_summary(
                [*pcbert_train, *pcbert_test], args.botrytis_proteome_fasta
            ),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(args.output_md, report)
    print(f"wrote: {args.output_json}")
    print(f"wrote: {args.output_md}")


if __name__ == "__main__":
    main()
