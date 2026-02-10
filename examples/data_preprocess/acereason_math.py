# acereason_math.py
"""
Preprocess the nvidia/AceReason-Math dataset to VERL-style parquet format.

HF dataset:
- split: train
- columns: problem, answer
See dataset card/viewer for details.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def normalize_answer(ans: str) -> str:
    """
    Keep the answer as a compact string.
    AceReason-Math answers are not always plain integers (can be fractions/latex/etc),
    so we do NOT attempt numeric extraction here.
    """
    if ans is None:
        return ""
    return str(ans).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="DEPRECATED. Use --local_save_dir instead.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS destination.")
    parser.add_argument("--local_dataset_path", default=None, help="Optional local dataset path (HF-compatible).")
    parser.add_argument("--local_save_dir", default="~/data/acereason_math", help="Where to write parquet.")
    parser.add_argument(
        "--make_test_split",
        action="store_true",
        help="If set, split the train set into train/test (since this dataset is train-only).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.02,
        help="Used only if --make_test_split is set. Fraction of examples for test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for train/test split if enabled.")

    args = parser.parse_args()

    data_source = "nvidia/AceReason-Math"
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # Load dataset
    if args.local_dataset_path is not None:
        ds = datasets.load_dataset(args.local_dataset_path)
    else:
        ds = datasets.load_dataset(data_source)

    # AceReason-Math is train-only on HF, but keep this generic.
    base = ds["train"] if "train" in ds else next(iter(ds.values()))

    def make_map_fn(split_name: str):
        def process_fn(example, idx):
            # HF columns per viewer: "problem", "answer"
            problem = example.get("problem", "")
            answer = example.get("answer", "")

            question = f"{problem} {instruction_following}".strip()
            solution = normalize_answer(answer)

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "answer": str(answer),
                    "question": str(problem),
                },
            }

        return process_fn

    # Optionally create a test split from train
    if args.make_test_split:
        split = base.train_test_split(test_size=args.test_ratio, seed=args.seed)
        train_dataset = split["train"].map(function=make_map_fn("train"), with_indices=True)
        test_dataset = split["test"].map(function=make_map_fn("test"), with_indices=True)
    else:
        train_dataset = base.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = None

    # Resolve save dir (keep same deprecated arg behavior as gsm8k.py)
    local_save_dir = args.local_dir if args.local_dir is not None else args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Write parquet(s)
    train_path = os.path.join(local_save_dir, "train.parquet")
    train_dataset.to_parquet(train_path)

    if test_dataset is not None:
        test_path = os.path.join(local_save_dir, "test.parquet")
        test_dataset.to_parquet(test_path)

    # Optional HDFS copy
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)

    print(f"Wrote: {train_path}")
    if test_dataset is not None:
        print(f"Wrote: {test_path}")
    if args.hdfs_dir is not None:
        print(f"Copied to HDFS: {args.hdfs_dir}")
