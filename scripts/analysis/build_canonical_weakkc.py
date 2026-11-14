#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from canonical_weakkc import build_canonical_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-valid-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="algebra2005")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    manifest = build_canonical_manifest(
        args.train_valid_source, args.test_source, args.out,
        dataset=args.dataset, fold=args.fold)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
