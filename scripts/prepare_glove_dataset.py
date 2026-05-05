import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset


def iter_glove(path: Path, limit: int | None):
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            parts = line.rstrip().split(" ")
            if len(parts) != 201:
                continue
            yield {"word": parts[0], "vector": [float(x) for x in parts[1:]]}


def main():
    parser = argparse.ArgumentParser(description="Prepare glove.6B.200d.txt as a Hugging Face dataset.")
    parser.add_argument("glove_txt", type=Path, help="Path to glove.6B.200d.txt")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a smaller demo dataset")
    parser.add_argument("--out", type=Path, default=None, help="Optional local parquet output path")
    parser.add_argument("--push-to-hub", default=None, help="Optional HF dataset repo id")
    args = parser.parse_args()

    rows = list(iter_glove(args.glove_txt, args.limit))
    if not rows:
        raise SystemExit("No GloVe rows parsed. Expected 200-dimensional glove.6B.200d.txt format.")

    dataset = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(args.out))
        print(f"Wrote {len(dataset)} rows to {args.out}")

    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub)
        print(f"Pushed {len(dataset)} rows to {args.push_to_hub}")

    if not args.out and not args.push_to_hub:
        print(f"Parsed {len(dataset)} rows. Pass --out or --push-to-hub to write the dataset.")


if __name__ == "__main__":
    main()
