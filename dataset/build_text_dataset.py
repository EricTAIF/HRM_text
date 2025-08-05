# dataset/build_text_dataset.py
from __future__ import annotations
import os
import json
from glob import glob
from typing import List, Tuple
import numpy as np
from utils.tokenizer_char import CharTokenizer, PAD_ID, EOS_ID  # EOS_ID kept for completeness

def _read_all_texts(input_dir: str) -> List[str]:
    paths = []
    for ext in ("*.txt",):
        paths.extend(glob(os.path.join(input_dir, ext)))
    texts = []
    for p in sorted(paths):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return texts

def _make_sequences(token_ids: np.ndarray, seq_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (inputs, labels) where:
      inputs  = tokens[t : t+seq_len]
      labels  = tokens[t+1 : t+seq_len+1]
    Drop the last window if shorter than seq_len+1.
    """
    T = token_ids.shape[0]
    window = seq_len + 1
    if T < window:
        return np.empty((0, seq_len), dtype=np.int32), np.empty((0, seq_len), dtype=np.int32)

    xs, ys = [], []
    for start in range(0, T - window + 1, stride):
        s = token_ids[start : start + window]
        x = s[:-1]                  # len = seq_len
        y = s[1:]                   # next-token targets
        xs.append(x.astype(np.int32))
        ys.append(y.astype(np.int32))
    return np.stack(xs, 0), np.stack(ys, 0)

def build_text_dataset(input_dir: str,
                       output_dir: str,
                       seq_len: int = 256,
                       stride: int = 256,
                       train_fraction: float = 0.9) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1) Read and fit tokenizer
    texts = _read_all_texts(input_dir)
    assert len(texts) > 0, f"No .txt files found in {input_dir}"
    tok = CharTokenizer()
    tok.fit(texts)
    tok.save(os.path.join(output_dir, "vocab.json"))

    # 2) Concatenate all texts into one stream per split
    full = "".join(texts)
    split = int(len(full) * train_fraction)
    train_text = full[:split]
    test_text  = full[split:]

    train_ids = np.array(tok.encode(train_text), dtype=np.int32)
    test_ids  = np.array(tok.encode(test_text),  dtype=np.int32)

    # 3) windows → inputs/labels
    tr_x, tr_y = _make_sequences(train_ids, seq_len=seq_len, stride=stride)
    te_x, te_y = _make_sequences(test_ids,   seq_len=seq_len, stride=stride)

    print(f"[builder] tokenizer vocab_size={tok.vocab_size} (PAD=0, EOS=1, chars start at 2)")
    print(f"[builder] train sequences: {tr_x.shape[0]}, test sequences: {te_x.shape[0]}, seq_len={seq_len}")

    def _write_split(split_name: str, x: np.ndarray, y: np.ndarray):
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        N = int(x.shape[0])
        T = int(x.shape[1])

        # === IMPORTANT: one sequence = one puzzle = one group ===
        # identifiers: 1..N (0 is reserved <blank>)
        puzzle_identifiers = np.arange(1, N + 1, dtype=np.int32)
        # puzzle_indices: cumulative example boundaries; 1 example per puzzle
        # [0,1,2,...,N]
        puzzle_indices = np.arange(0, N + 1, dtype=np.int32)
        # group_indices: one puzzle per group → same as puzzle_indices
        group_indices = np.arange(0, N + 1, dtype=np.int32)

        # Save arrays
        np.save(os.path.join(split_dir, "all__inputs.npy"),             x)
        np.save(os.path.join(split_dir, "all__labels.npy"),             y)
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"),     puzzle_indices)
        np.save(os.path.join(split_dir, "all__group_indices.npy"),      group_indices)

        # Metadata (ARC-compatible fields)
        metadata = dict(
            seq_len=T,                          # model sees T tokens, predicts next at every position
            vocab_size=int(tok.vocab_size),
            pad_id=PAD_ID,
            ignore_label_id=PAD_ID,             # labels equal to PAD will be ignored (none here)
            blank_identifier_id=0,
            num_puzzle_identifiers=N + 1,       # include <blank>=0
            total_groups=N,                     # <-- critical: drives training steps
            mean_puzzle_examples=1.0,
            sets=["all"],
        )
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f)

    _write_split("train", tr_x, tr_y)
    _write_split("test",  te_x, te_y)

    # identifiers.json: index 0 = <blank>, then 1..maxN are simple strings
    maxN = int(max(tr_x.shape[0], te_x.shape[0]))
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"] + [f"text_{i}" for i in range(1, maxN + 1)], f)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder with .txt files")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)   # set <seq_len for overlapping windows
    ap.add_argument("--train-fraction", type=float, default=0.9)
    args = ap.parse_args()
    build_text_dataset(args.input_dir, args.output_dir, args.seq_len, args.stride, args.train_fraction)
