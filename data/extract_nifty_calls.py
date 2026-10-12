"""Extract the NIFTY CE sample used by project1-quant from a large merged CSV.

Example:
  python data/extract_nifty_calls.py \
    --input ~/Downloads/final_merged_output.csv \
    --output data/nifty_calls_600pt_expiry_2024-10-03_recreated.csv

The source is processed in chunks so the full merged file is never loaded into RAM.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_ANCHOR_SPOT = 25898.35
DEFAULT_RANGE_POINTS = 600.0
DEFAULT_EXPIRY = "2024-10-03"
DEFAULT_START = "2024-09-25"
DEFAULT_END = "2024-10-03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Large merged option/spot CSV")
    parser.add_argument("--output", type=Path, default=Path("data/nifty_calls_600pt_expiry_2024-10-03_extracted.csv"))
    parser.add_argument("--anchor-spot", type=float, default=DEFAULT_ANCHOR_SPOT)
    parser.add_argument("--range-points", type=float, default=DEFAULT_RANGE_POINTS)
    parser.add_argument("--expiry", default=DEFAULT_EXPIRY, help="Expiry date YYYY-MM-DD")
    parser.add_argument("--start", default=DEFAULT_START, help="First quote date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="Last quote date YYYY-MM-DD")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--force", action="store_true", help="Overwrite the output if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser()
    output = args.output.expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.range_points <= 0 or args.chunksize <= 0:
        raise ValueError("range-points and chunksize must be positive")
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}. Choose another path or pass --force.")

    low_strike = args.anchor_spot - args.range_points
    high_strike = args.anchor_spot + args.range_points
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    wrote_header = False
    required = {
        "strike_price", "option_type", "expiry", "timestamp",
        "underlying_spot_price", "ltp",
    }

    # Start a fresh output, then append each selected chunk immediately.
    output.write_text("", encoding="utf-8")
    for chunk_number, chunk in enumerate(
        pd.read_csv(source, chunksize=args.chunksize, low_memory=False), start=1
    ):
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"Source CSV is missing required columns: {sorted(missing)}")

        strike = pd.to_numeric(chunk["strike_price"], errors="coerce")
        timestamp = pd.to_datetime(chunk["timestamp"], errors="coerce")
        expiry_date = pd.to_datetime(chunk["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
        mask = (
            chunk["option_type"].astype(str).str.strip().str.upper().eq("CE")
            & strike.between(low_strike, high_strike, inclusive="both")
            & expiry_date.eq(args.expiry)
            & timestamp.dt.strftime("%Y-%m-%d").between(args.start, args.end, inclusive="both")
        )
        selected = chunk.loc[mask].copy()
        if not selected.empty:
            selected.to_csv(
                output,
                mode="a",
                index=False,
                header=not wrote_header,
            )
            wrote_header = True
            rows_written += len(selected)
        if chunk_number % 20 == 0:
            print(f"Scanned {chunk_number:,} chunks; kept {rows_written:,} rows")

    if not wrote_header:
        raise ValueError("No rows matched; check the input file and filter settings.")
    print(f"Saved {rows_written:,} rows to {output}")
    print(f"Filter: CE, expiry {args.expiry}, dates {args.start} to {args.end}, "
          f"strikes {low_strike:g} to {high_strike:g}")


if __name__ == "__main__":
    main()
