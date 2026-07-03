"""Master of Freud's LLM v5 — 交互式聊天"""
import os, torch, re
os.environ["HF_HOME"] = "D:/MasterOfFreudsLLM/.cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to("cuda"),
    "D:/MasterOfFreudsLLM/checkpoints/qwen_psych_v5/final"
)
model.eval()

def clean(text):
    return re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u0020-\u007e\n，。！？、；：""''（）【】《》\-]', "", text).strip()

print("🧠 Master of Freud's LLM v5 已就绪！输入 /quit 退出\n")
while True:
    q = input("\n😶 你 > ").strip()
    if not q: continue
    if q == "/quit": print("👋 再见！"); break
    ids = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=250, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.1, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    reply = clean(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
    print(f"🧠 {reply}")
