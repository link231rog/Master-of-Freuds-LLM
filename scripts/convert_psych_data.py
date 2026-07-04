"""
Convert psychology SFT data from HDF5 (r50k_base) to plain text JSONL for Qwen.
"""

import os
import json, os
import h5py
import tiktoken

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

enc = tiktoken.get_encoding("r50k_base")
OUT_DIR = os.path.join(ROOT_DIR, "data_psychology")
os.makedirs(OUT_DIR, exist_ok=True)

for split, fname, out_name in [
    ("train", "sft_psych.h5", "psych_train.jsonl"),
    ("dev", "sft_psych_dev.h5", "psych_dev.jsonl"),
]:
    with h5py.File(f"{OUT_DIR}/{fname}", "r") as f:
        tokens = f["tokens"][:]

    count = 0
    with open(f"{OUT_DIR}/{out_name}", "w", encoding="utf-8") as out:
        for row in tokens:
            text = enc.decode(row.tolist())
            # Extract user and assistant parts
            parts = text.split("<|endoftext|>")
            messages = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if p.startswith("<|user|>"):
                    content = p.replace("<|user|>\n", "").replace("<|user|>", "").strip()
                    if content:
                        messages.append({"role": "user", "content": content})
                elif p.startswith("<|assistant|>"):
                    content = p.replace("<|assistant|>\n", "").replace("<|assistant|>", "").strip()
                    if content:
                        messages.append({"role": "assistant", "content": content})
            if len(messages) >= 2:
                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1

    print(f"{out_name}: {count} examples ({split})")
