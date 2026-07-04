"""
Minimal Qwen LoRA fine-tuning. Pre-tokenizes all data first.
"""

import os
import json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".cache")
device = "cuda"
OUT_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen_psych")
os.makedirs(OUT_DIR, exist_ok=True)

# Load tokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

# Pre-tokenize ALL data at once
print("Pre-tokenizing data...")
train_texts, dev_texts = [], []
with open("data_psychology/qwen_psych_train.jsonl") as f:
    for line in f:
        msg = json.loads(line)["messages"]
        train_texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=False))
with open("data_psychology/qwen_psych_dev.jsonl") as f:
    for line in f:
        msg = json.loads(line)["messages"]
        dev_texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=False))

train_enc = tok(train_texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
dev_enc = tok(dev_texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
print(f"Train: {len(train_texts)} | Dev: {len(dev_texts)} | Max len: {train_enc.input_ids.shape[1]}")

# Model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to(device)
model.gradient_checkpointing_enable()
lora = LoraConfig(r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
B = 2  # batch size
GA = 4  # grad accum steps
N_EPOCHS = 5
MAX_STEPS = 500

step = 0
n = len(train_enc.input_ids)
best_loss = 999

for epoch in range(N_EPOCHS):
    # Shuffle
    idx = torch.randperm(n)
    inp, attn, lbl = train_enc.input_ids[idx], train_enc.attention_mask[idx], train_enc.input_ids[idx].clone()
    inp, attn, lbl = inp.to(device), attn.to(device), lbl.to(device)

    for i in range(0, n - B + 1, B):
        if step >= MAX_STEPS: break
        out = model(inp[i:i+B], attention_mask=attn[i:i+B], labels=lbl[i:i+B])
        loss = out.loss / GA
        loss.backward()

        if (i // B + 1) % GA == 0:
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 5 == 0:
                print(f"E{epoch+1} s{step} loss {out.loss.item():.4f}")

        if step % 50 == 0 and step > 0:
            # Eval
            model.eval()
            with torch.no_grad():
                dev_loss = model(dev_enc.input_ids.to(device), attention_mask=dev_enc.attention_mask.to(device), labels=dev_enc.input_ids.to(device)).loss.item()
            print(f"  [eval] dev_loss {dev_loss:.4f}", end="")
            if dev_loss < best_loss:
                best_loss = dev_loss
                model.save_pretrained(os.path.join(OUT_DIR, "best"))
                print(f" 💾 saved!")
            else:
                print()
            model.train()

    if step >= MAX_STEPS: break

# Save final
model.save_pretrained(os.path.join(OUT_DIR, "final"))
tok.save_pretrained(os.path.join(OUT_DIR, "final"))
print(f"\n✅ Done! Saved to {OUT_DIR}")

# Test
print("\n🔮 Test:")
model.eval()
for prompt in [
    "今天在会议上被领导当众批评了，好几个人都看着，尴尬得要死",
    "最近总是失眠，脑子里停不下来",
]:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, temperature=0.8, top_p=0.9, do_sample=True)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {prompt}\nA: {reply.strip()}")
