"""
Quick script: download a small slice of Pile-uncopyrighted for pretraining.
Uses HuggingFace datasets streaming to avoid downloading the full 11GB file.
Outputs HDF5 format compatible with pretrain_base.py.
"""
import os, sys
import h5py
import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = "D:/MasterOfFreudsLLM/data"
os.makedirs(OUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding("r50k_base")
eot = enc._special_tokens["<|endoftext|>"]

N_TARGET = 100_000_000  # 100M tokens

def tokenize_and_write(split, out_path, n_target):
    ds = load_dataset("monology/pile-uncopyrighted", split=split, streaming=True)
    tokens = [eot]
    count = 0
    for example in ds:
        text = example["text"] + "<|endoftext|>"
        t = enc.encode(text)
        tokens.extend(t)
        count += len(t)
        if count >= n_target:
            break

    arr = np.array(tokens[:n_target], dtype=np.int32)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("tokens", data=arr, compression="gzip")
    print(f"Wrote {len(arr)} tokens -> {out_path}")

print("Downloading validation set...")
tokenize_and_write("val", os.path.join(OUT_DIR, "pile_dev.h5"), 10_000_000)

print("Downloading training set (100M tokens)...")
tokenize_and_write("train", os.path.join(OUT_DIR, "pile_train.h5"), N_TARGET)

print("Done!")
