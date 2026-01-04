#!/usr/bin/env python3
"""Convert first sheet of an .xlsx file to CSV using openpyxl.

Usage:
  python convert_xlsx_to_csv.py input.xlsx output.csv

If output path is omitted, a CSV with the same name as input will be written
next to the input file.
"""
import sys
from pathlib import Path

def main():
    try:
        import openpyxl
    except Exception as e:
        print("Missing dependency 'openpyxl'. Install with: pip install openpyxl", file=sys.stderr)
        sys.exit(2)

    if len(sys.argv) < 2:
        print("Usage: python convert_xlsx_to_csv.py INPUT_XLSX [OUTPUT_CSV]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(3)

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = in_path.with_suffix('.csv')

    wb = openpyxl.load_workbook(in_path, read_only=True, data_only=True)
    sheet = wb.active

    import csv
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in sheet.iter_rows(values_only=True):
            # Convert None to empty string to avoid literal 'None' in CSV
            writer.writerow(["" if v is None else v for v in row])

    print(f"Wrote CSV: {out_path}")

if __name__ == '__main__':
    main()
