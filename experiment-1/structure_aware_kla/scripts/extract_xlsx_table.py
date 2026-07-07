from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


SPREADSHEET_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}


def load_shared_strings(workbook: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("x:si", SPREADSHEET_NS):
        parts = [text.text or "" for text in item.findall(".//x:t", SPREADSHEET_NS)]
        strings.append("".join(parts))
    return strings


def sheet_paths(workbook: ZipFile) -> dict[str, str]:
    workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))

    rels = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rel_root.findall("r:Relationship", REL_NS)
    }

    paths: dict[str, str] = {}
    for sheet in workbook_root.findall(".//x:sheet", SPREADSHEET_NS):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rels[rel_id]
        if not target.startswith("/"):
            target = "xl/" + target
        paths[name] = target.lstrip("/")
    return paths


def column_index(cell_reference: str) -> int:
    column_letters = re.match(r"[A-Z]+", cell_reference)
    if column_letters is None:
        raise ValueError(f"Invalid cell reference: {cell_reference}")
    index = 0
    for char in column_letters.group(0):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find("x:v", SPREADSHEET_NS)
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        parts = [text.text or "" for text in cell.findall(".//x:t", SPREADSHEET_NS)]
        return "".join(parts)
    if value is None:
        return ""
    raw_value = value.text or ""
    if cell_type == "s":
        return shared_strings[int(raw_value)]
    return raw_value


def read_sheet(workbook: ZipFile, path: str) -> list[list[str]]:
    shared_strings = load_shared_strings(workbook)
    root = ET.fromstring(workbook.read(path))

    rows: list[list[str]] = []
    for row in root.findall(".//x:sheetData/x:row", SPREADSHEET_NS):
        values: list[str] = []
        for cell in row.findall("x:c", SPREADSHEET_NS):
            ref = cell.attrib.get("r", "")
            index = column_index(ref) if ref else len(values)
            while len(values) < index:
                values.append("")
            values.append(cell_value(cell, shared_strings))
        while values and values[-1] == "":
            values.pop()
        rows.append(values)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the first or named sheet from an XLSX file to CSV."
    )
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("--sheet", help="Worksheet name. Defaults to the first sheet.")
    parser.add_argument(
        "--header-row-contains",
        action="append",
        help=(
            "Drop leading rows before the first row containing this exact cell value. "
            "May be provided multiple times; all values must be present in the row."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with ZipFile(args.xlsx) as workbook:
        paths = sheet_paths(workbook)
        if args.sheet is None:
            sheet_name, path = next(iter(paths.items()))
        else:
            if args.sheet not in paths:
                available = ", ".join(paths)
                raise SystemExit(f"Sheet {args.sheet!r} not found. Available: {available}")
            sheet_name = args.sheet
            path = paths[sheet_name]
        rows = read_sheet(workbook, path)

    if args.header_row_contains is not None:
        required_cells = set(args.header_row_contains)
        for row_index, row in enumerate(rows):
            if required_cells.issubset(set(row)):
                rows = rows[row_index:]
                break
        else:
            raise SystemExit(
                f"Header cells {sorted(required_cells)!r} not found in sheet {sheet_name!r}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerows(rows)

    print(f"sheet: {sheet_name}")
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
