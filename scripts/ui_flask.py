"""
Master of Freud's LLM — Flask Web UI
浏览器打开 http://localhost:5000
"""

import os
import torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_HOME"] = "D:/MasterOfFreudsLLM/.cache"

print("🧠 Loading Master of Freud's LLM...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "D:/MasterOfFreudsLLM/checkpoints/qwen_psych/best")
model.eval()
print(f"✅ Loaded! {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(".", "ui_index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message", "")
    msgs = [{"role": "user", "content": msg}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=200, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return jsonify({"reply": reply.strip()})


if __name__ == "__main__":
    print("\n🌐 打开浏览器访问 http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
