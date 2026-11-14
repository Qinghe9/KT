#!/usr/bin/env python3
"""Fail-fast audit for canonical WeakKC predictions."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--prediction", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_dir = Path(args.manifest_dir)
    manifest = json.loads((manifest_dir / "manifest.json").read_text())
    canonical = [json.loads(line) for line in
                 (manifest_dir / "test.jsonl").read_text().splitlines() if line]
    expected = {(str(row["uid"]), row["target_kc"] + 1): row for row in canonical}
    report = {"protocol_hash": manifest["protocol_hash"],
              "canonical_test_candidates": len(expected), "files": []}
    all_ok = True
    for name in args.prediction:
        path = Path(name)
        rows = list(csv.DictReader(path.open()))
        observed = {}
        for row in rows:
            concept = int(row.get("concept", row.get("concepts", "0").strip("[]")))
            observed[(str(row["uid"]), concept)] = row
        keys_ok = observed.keys() == expected.keys()
        labels_ok = keys_ok and all(int(float(observed[key]["y"])) == value["label"]
                                    for key, value in expected.items())
        supports_ok = keys_ok and all(
            not observed[key].get("future_support") or
            int(float(observed[key]["future_support"])) == value["future_support"]
            for key, value in expected.items())
        ok = keys_ok and labels_ok and supports_ok and len(rows) == len(expected)
        all_ok &= ok
        report["files"].append({"path": str(path.resolve()), "rows": len(rows),
                                "keys_ok": keys_ok, "labels_ok": labels_ok,
                                "future_support_ok": supports_ok, "passed": ok})
    report["passed"] = all_ok
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
