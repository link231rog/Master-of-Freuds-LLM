"""
Manual LoRA fine-tuning for Qwen2.5-0.5B on psychology data.
Avoids Trainer segfault by using a manual training loop.
"""

import json
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "D:/MasterOfFreudsLLM/checkpoints/qwen_psych"
CACHE_DIR = "D:/MasterOfFreudsLLM/.cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR


class PsychDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msg = self.data[idx]["messages"]
        text = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        tok = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {
            "input_ids": tok["input_ids"][0],
            "attention_mask": tok["attention_mask"][0],
            "labels": tok["input_ids"][0].clone(),
        }


def main():
    print("🧠 Master of Freud's LLM — Qwen LoRA (manual training)")
    print("=" * 60)

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = PsychDataset("data_psychology/qwen_psych_train.jsonl", tokenizer, max_len=512)
    dev_ds = PsychDataset("data_psychology/qwen_psych_dev.jsonl", tokenizer, max_len=512)
    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=2, shuffle=False)

    # Model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR).to(device)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    n_epochs = 5
    accum_steps = 4
    log_every = 5
    eval_every = 50
    global_step = 0
    best_loss = float("inf")

    print("\n🚀 Training...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        accum_loss = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            inp = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=inp, attention_mask=attn, labels=labels)
            loss = out.loss / accum_steps
            loss.backward()
            accum_loss += out.loss.item()

            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    tok_s = accum_steps * batch["input_ids"].shape[0] * batch["input_ids"].shape[1] / (time.time() - t0)
                    print(f"  E{epoch+1} step {global_step} | loss {accum_loss/accum_steps:.4f} | {tok_s:.0f} tok/s")
                    total_loss += accum_loss
                    accum_loss = 0
                    t0 = time.time()

                if global_step % eval_every == 0:
                    # Eval
                    model.eval()
                    dev_loss = 0
                    n_dev = 0
                    with torch.no_grad():
                        for dev_batch in dev_loader:
                            inp = dev_batch["input_ids"].to(device)
                            attn = dev_batch["attention_mask"].to(device)
                            labels = dev_batch["labels"].to(device)
                            out = model(input_ids=inp, attention_mask=attn, labels=labels)
                            dev_loss += out.loss.item()
                            n_dev += 1
                    dev_loss /= n_dev
                    print(f"  [eval] dev_loss {dev_loss:.4f}")
                    if dev_loss < best_loss:
                        best_loss = dev_loss
                        model.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
                        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
                        print(f"  ✅ Saved best model (dev_loss {dev_loss:.4f})")
                    model.train()

            if global_step >= 500:
                break

        if global_step >= 500:
            break

    # Save final
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"\n✅ Training done! Saved to {OUTPUT_DIR}/final")

    # Test
    print("\n🧪 Test generation:")
    model.eval()
    model = model.to(device)
    for prompt in [
        "今天在会议上被领导当众批评了，好几个人都看着，尴尬得要死",
        "最近总是失眠，脑子里停不下来",
    ]:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, temperature=0.8, top_p=0.9, do_sample=True)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nQ: {prompt}")
        print(f"A: {reply.strip()}")


if __name__ == "__main__":
    main()
