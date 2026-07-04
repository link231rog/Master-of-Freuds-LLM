"""
Master of Freud's LLM — Interactive Chat (fast version)
"""

import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".cache")
device = "cuda"

print("🧠 Loading Master of Freud's LLM...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, os.path.join(ROOT_DIR, "checkpoints", "qwen_psych", "best"))
model.eval()
print(f"✅ Ready! {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

print("\n" + "="*50)
print("🧠 Master of Freud's LLM")
print("输入你的生活事件，按 Enter")
print("/quit 退出")
print("="*50)

with torch.no_grad():
    while True:
        user = input("\n😶 你 > ").strip()
        if not user: continue
        if user == "/quit":
            print("👋 再见！")
            break

        msgs = [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(device)

        out = model.generate(
            **ids,
            max_new_tokens=150,
            temperature=0.85,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
        )
        reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"🧠 弗洛伊德 > {reply.strip()}")
