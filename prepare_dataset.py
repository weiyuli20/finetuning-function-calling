import argparse
import json
import os
import random

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Split xLAM dataset into train/eval/test")
    p.add_argument("--data_dir",    type=str,   default="data",
                   help="Directory to save the split JSONL files")
    p.add_argument("--input_file", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "xlam-function-calling-60k.json"),
                   help="Path to the local xLAM JSON dataset")
    p.add_argument("--eval_ratio",  type=float, default=0.05,
                   help="Fraction of data for eval (default: 1 %%)")
    p.add_argument("--test_ratio",  type=float, default=0.05,
                   help="Fraction of data for test (default: 5 %%)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--max_samples", type=int,   default=None,
                   help="Cap total dataset size (useful for quick tests)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    assert args.eval_ratio + args.test_ratio < 1.0, \
        "eval_ratio + test_ratio must be < 1.0"

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Dataset file not found: {args.input_file}")

    print(f"[prepare] Loading local dataset: {args.input_file}")
    raw = load_dataset("json", data_files=args.input_file, split="train")

    if args.max_samples:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    total = len(raw)
    print(f"[prepare] Total examples: {total:,}")

    indices = list(range(total))
    random.seed(args.seed)
    random.shuffle(indices)

    n_test  = int(total * args.test_ratio)
    n_eval  = int(total * args.eval_ratio)
    n_train = total - n_eval - n_test

    train_idx = indices[:n_train]
    eval_idx  = indices[n_train : n_train + n_eval]
    test_idx  = indices[n_train + n_eval :]

    splits = {
        "train": raw.select(train_idx),
        "eval":  raw.select(eval_idx),
        "test":  raw.select(test_idx),
    }

    for split_name, ds in splits.items():
        out_path = os.path.join(args.data_dir, f"xlam_{split_name}.jsonl")
        print(f"[prepare] Writing {len(ds):,} examples → {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "source":       "Salesforce/xlam-function-calling-60k",
        "seed":         args.seed,
        "total":        total,
        "n_train":      n_train,
        "n_eval":       n_eval,
        "n_test":       n_test,
        "eval_ratio":   args.eval_ratio,
        "test_ratio":   args.test_ratio,
        "train_file":   os.path.join(args.data_dir, "xlam_train.jsonl"),
        "eval_file":    os.path.join(args.data_dir, "xlam_eval.jsonl"),
        "test_file":    os.path.join(args.data_dir, "xlam_test.jsonl"),
    }
    meta_path = os.path.join(args.data_dir, "split_info.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] Split metadata saved → {meta_path}")

    print("\n[prepare] Done!")
    print(f"  Train : {n_train:>6,}  ({n_train/total*100:.1f} %)")
    print(f"  Eval  : {n_eval:>6,}  ({n_eval/total*100:.1f} %)")
    print(f"  Test  : {n_test:>6,}  ({n_test/total*100:.1f} %)")
    print(f"\n  Files saved under: {os.path.abspath(args.data_dir)}/")
    print("  Run training scripts with:")
    print(f"    --train_file {meta['train_file']}")
    print(f"    --eval_file  {meta['eval_file']}")


if __name__ == "__main__":
    main()

'''
python prepare_dataset.py --max_samples 3000
'''