"""Train v5 with tag-matched psychology data"""
import json, os, torch
os.environ["HF_HOME"] = "D:/MasterOfFreudsLLM/.cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

OUT_DIR = "D:/MasterOfFreudsLLM/checkpoints/qwen_psych_v5"
os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token
texts = []
with open("data_psychology/psych_v5_train.jsonl") as f:
    for line in f:
        texts.append(tok.apply_chat_template(json.loads(line)["messages"], tokenize=False, add_generation_prompt=False))
enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
print(f"Train: {len(texts)}, max_len: {enc.input_ids.shape[1]}")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to("cuda")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
step, n = 0, len(enc.input_ids)
for epoch in range(5):
    if step >= 800: break
    idx = torch.randperm(n)
    for i in range(0, n):
        if step >= 800: break
        ids = enc.input_ids[idx[i:i+1]].to("cuda")
        attn = enc.attention_mask[idx[i:i+1]].to("cuda")
        out = model(ids, attention_mask=attn, labels=ids.clone())
        loss = out.loss / 8
        loss.backward()
        if (i + 1) % 8 == 0:
            opt.step(); opt.zero_grad(); step += 1
            if step % 20 == 0: print(f"E{epoch+1} s{step} loss {out.loss.item():.4f}")
            if step % 200 == 0: model.save_pretrained(os.path.join(OUT_DIR, f"ckpt_{step}"))

model.save_pretrained(os.path.join(OUT_DIR, "final"))
tok.save_pretrained(os.path.join(OUT_DIR, "final"))
print(f"\n✅ v5 saved to {OUT_DIR}")

# Test
from peft import PeftModel
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to("cuda"), os.path.join(OUT_DIR, "final"))
model.eval()

def clean(text):
    """Remove non-Chinese trailing characters (Thai etc.)"""
    import re
    # Keep only Chinese chars, common punctuation, and spaces
    clean = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u0020-\u007e\n，。！？、；：""''（）【】《》\-]', '', text)
    return clean.strip()

tests = [
    "面试了好几家都失败了，怀疑自己是不是真的不行",
    "帮了朋友大忙，他连句谢谢都没有",
    "部门来了个新领导总觉得他在针对我",
]
for q in tests:
    ids = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=200, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.1, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    reply = clean(reply)
    print(f"\n😶 {q}\n🧠 {reply}\n")
