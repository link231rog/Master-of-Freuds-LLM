"""
Train v4 — user's style psychology model
"""
import json, os, torch
os.environ["HF_HOME"] = "D:/MasterOfFreudsLLM/.cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

device = "cuda"
OUT_DIR = "D:/MasterOfFreudsLLM/checkpoints/qwen_psych_v4"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

texts = []
with open("data_psychology/psych_v4_train.jsonl") as f:
    for line in f:
        msg = json.loads(line)["messages"]
        texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=False))
enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
print(f"Train: {len(texts)}, max_len: {enc.input_ids.shape[1]}")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to(device)
lora = LoraConfig(r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
MAX_STEPS = 800
step = 0
n = len(enc.input_ids)

for epoch in range(5):
    if step >= MAX_STEPS: break
    idx = torch.randperm(n)
    for i in range(0, n, 1):
        if step >= MAX_STEPS: break
        ids = enc.input_ids[idx[i:i+1]].to(device)
        attn = enc.attention_mask[idx[i:i+1]].to(device)
        out = model(ids, attention_mask=attn, labels=ids.clone())
        loss = out.loss / 8
        loss.backward()
        if (i + 1) % 8 == 0:
            opt.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                print(f"E{epoch+1} s{step} loss {out.loss.item():.4f}")
            if step % 200 == 0:
                model.save_pretrained(os.path.join(OUT_DIR, f"ckpt_{step}"))

model.save_pretrained(os.path.join(OUT_DIR, "final"))
tok.save_pretrained(os.path.join(OUT_DIR, "final"))
print(f"\n✅ v4 saved to {OUT_DIR}")

# Test
print("\n🔮 Testing v4...")
model.eval()
model.config.use_cache = True
tests = [
    "面试了好几家都失败了，怀疑自己是不是真的不行",
    "帮了朋友大忙，他连句谢谢都没有",
    "发了朋友圈没人点赞，觉得自己人缘差",
]
for prompt in tests:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=300, temperature=0.8, top_p=0.9, do_sample=True, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n😶 {prompt}\n🧠 {reply.strip()}\n")
