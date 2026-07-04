"""
Stable Qwen LoRA training. No gradient checkpointing (causes bugs on this setup).
Uses use_cache=True and small batch.
"""
import os
import json, os, torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".cache")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

device = "cuda"
OUT_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen_psych_v2")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading tokenizer & pre-tokenizing...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

train_texts = []
with open("data_psychology/qwen_psych_train.jsonl") as f:
    for line in f:
        msg = json.loads(line)["messages"]
        train_texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=False))

train_enc = tok(train_texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
print(f"Train: {len(train_texts)}, max_len: {train_enc.input_ids.shape[1]}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to(device)
# NO gradient_checkpointing — it's buggy with this setup
lora = LoraConfig(r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
B = 1  # batch_size=1 to save memory
GA = 8  # grad_accum=8 for effective batch=8
MAX_STEPS = 500
step = 0
n = len(train_enc.input_ids)

for epoch in range(5):
    if step >= MAX_STEPS: break
    idx = torch.randperm(n)
    for i in range(0, n, B):
        if step >= MAX_STEPS: break
        ids = train_enc.input_ids[idx[i:i+B]].to(device)
        attn = train_enc.attention_mask[idx[i:i+B]].to(device)
        lbl = ids.clone()
        out = model(ids, attention_mask=attn, labels=lbl)
        loss = out.loss / GA
        loss.backward()
        if (i // B + 1) % GA == 0:
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"E{epoch+1} s{step} loss {out.loss.item():.4f}")
            if step % 100 == 0:
                model.save_pretrained(os.path.join(OUT_DIR, f"step_{step}"))

model.save_pretrained(os.path.join(OUT_DIR, "final"))
tok.save_pretrained(os.path.join(OUT_DIR, "final"))
print(f"\n✅ Saved to {OUT_DIR}")

# Test
print("\n🔮 Test:")
model.eval()
model.config.use_cache = True
tests = [
    "今天在会议上被领导当众批评了，好几个人都看着，尴尬得要死",
    "最近总是失眠，脑子里停不下来",
    "三十岁了还在做基础岗位，觉得人生没希望了",
]
for prompt in tests:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=100, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n😶 {prompt}\n🧠 {reply.strip()}")
