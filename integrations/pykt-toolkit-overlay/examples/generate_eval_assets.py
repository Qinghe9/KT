"""Generate non-window question-test sequences and DIMKT difficulty tables.

This utility is intentionally split-safe: question sequences are derived from
the already fixed test.csv, while difficulty statistics use only
train_valid_sequences.csv.  Large datasets are processed in chunks for the
difficulty tables.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pykt.preprocess.split_datasets import ALL_KEYS, generate_question_sequences


def write_difficulty(counts, correct, destination, level, domain_size):
    # DIMKTDataset indexes these tables directly.  IDs absent from the
    # training split must therefore receive the same default used for rare
    # items, without consulting validation/test answers.
    keys = list(range(domain_size))
    values = []
    for key in keys:
        total = counts.get(key, 0)
        right = correct.get(key, 0)
        value = 1 if total < 30 or right == 0 else int(right / total * level) + 1
        values.append(value)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerow(values)
    os.replace(temporary, destination)


def generate_difficulties(data_dir, level, chunksize, num_q, num_c):
    source = data_dir / "train_valid_sequences.csv"
    concept_counts, concept_correct = defaultdict(int), defaultdict(int)
    question_counts, question_correct = defaultdict(int), defaultdict(int)
    rows = interactions = 0
    for frame in pd.read_csv(source, usecols=["questions", "concepts", "responses"], chunksize=chunksize):
        for row in frame.itertuples(index=False):
            questions = [int(value) for value in row.questions.split(",")]
            concepts = [int(value) for value in row.concepts.split(",")]
            responses = [int(value) for value in row.responses.split(",")]
            for question, concept, response in zip(questions, concepts, responses):
                if response == -1:
                    break
                question_counts[question] += 1
                question_correct[question] += response
                concept_counts[concept] += 1
                concept_correct[concept] += response
                interactions += 1
            rows += 1
        print(f"[{data_dir.name}] difficulty rows={rows:,} interactions={interactions:,}", flush=True)
    write_difficulty(concept_counts, concept_correct, data_dir / f"skills_difficult_{level}.csv", level, num_c)
    write_difficulty(question_counts, question_correct, data_dir / f"questions_difficult_{level}.csv", level, num_q)


def complete_existing_table(path, domain_size):
    with path.open() as handle:
        rows = list(csv.reader(handle))
    values = {int(key): int(value) for key, value in zip(rows[0], rows[1])}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(range(domain_size))
        writer.writerow(values.get(key, 1) for key in range(domain_size))
    os.replace(temporary, path)


def generate_questions(data_dir, maxlen, min_seq_len):
    source = data_dir / "test.csv"
    frame = pd.read_csv(source)
    effective = [key for key in ALL_KEYS if key in frame.columns and key != "cidxs"]
    ok, result = generate_question_sequences(
        frame, effective, window=False, min_seq_len=min_seq_len, maxlen=maxlen
    )
    if not ok:
        raise RuntimeError(f"cannot generate question sequences from {source}")
    destination = data_dir / "test_question_sequences.csv"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f"[{data_dir.name}] question rows={len(result):,} -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data_config.json")
    parser.add_argument("--difficulty-level", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for dataset in args.datasets:
        data_dir = args.data_root / dataset
        question_file = data_dir / "test_question_sequences.csv"
        skill_file = data_dir / f"skills_difficult_{args.difficulty_level}.csv"
        question_difficulty_file = data_dir / f"questions_difficult_{args.difficulty_level}.csv"
        if args.force or not question_file.exists():
            generate_questions(data_dir, int(config[dataset]["maxlen"]), int(config[dataset]["min_seq_len"]))
        else:
            print(f"[{dataset}] keep existing {question_file}", flush=True)
        if args.force or not (skill_file.exists() and question_difficulty_file.exists()):
            generate_difficulties(
                data_dir, args.difficulty_level, args.chunksize,
                int(config[dataset]["num_q"]), int(config[dataset]["num_c"])
            )
        else:
            complete_existing_table(skill_file, int(config[dataset]["num_c"]))
            complete_existing_table(question_difficulty_file, int(config[dataset]["num_q"]))
            print(f"[{dataset}] completed existing DIMKT difficulty ID domains", flush=True)


if __name__ == "__main__":
    main()
