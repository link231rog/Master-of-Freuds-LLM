"""
Master of Freud's LLM — Web UI
浏览器打开 http://localhost:5000
模型升级时改 MODEL_PATH 即可
"""

import os, sys, torch, re

MODEL_PATH = "D:/MasterOfFreudsLLM/checkpoints/qwen_psych_v5/final"

os.environ["HF_HOME"] = "D:/MasterOfFreudsLLM/.cache"
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, "D:/MasterOfFreudsLLM/scripts")
from rag_psychology import build_prompt_with_theory

print("🧠 Loading...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to("cuda"),
    MODEL_PATH
)
model.eval()
print(f"✅ Loaded from {MODEL_PATH}")

def clean(text):
    return re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u0020-\u007e\n，。！？、；：""''（）【】《》-]', "", text).strip()

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Master of Freud's LLM</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f0eb;min-height:100vh;display:flex;justify-content:center}
.container{width:720px;max-width:100%;display:flex;flex-direction:column;min-height:100vh;padding:0 20px}
header{text-align:center;padding:28px 0 12px;border-bottom:1px solid #e0d6cc}
header h1{font-size:22px;color:#2c1810}
header p{font-size:13px;color:#8b7355;margin-top:4px}
.chat{flex:1;overflow-y:auto;padding:16px 0;display:flex;flex-direction:column;gap:14px}
.msg{max-width:88%;padding:12px 16px;border-radius:16px;line-height:1.7;font-size:14px;animation:fadeIn .3s;white-space:pre-wrap}
.user{background:#2c1810;color:#fff;align-self:flex-end;border-radius:16px 4px 16px 16px}
.bot{background:#fff;color:#2c1810;align-self:flex-start;border-radius:4px 16px 16px 16px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.bot .avatar{font-size:20px;margin-bottom:4px}
.think{color:#8b7355;font-size:13px;padding:4px 16px;font-style:italic;animation:pulse 1.5s infinite}
.tag{display:inline-block;font-size:11px;background:#e8e0d8;color:#6b5a4a;border-radius:4px;padding:1px 8px;margin:4px 0}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.input-area{border-top:1px solid #e0d6cc;padding:12px 0 24px;display:flex;gap:8px}
.input-area input{flex:1;padding:12px 16px;border:1px solid #ddd0c4;border-radius:24px;font-size:14px;outline:none;background:#fff;color:#2c1810}
.input-area input:focus{border-color:#8b7355}
.input-area button{padding:12px 24px;background:#2c1810;color:#fff;border:none;border-radius:24px;font-size:14px;cursor:pointer}
.input-area button:hover{background:#4a2c1a}
footer{text-align:center;font-size:11px;color:#c4b8a8;padding:8px 0 16px}
</style>
</head>
<body>
<div class="container">
<header><h1>🧠 Master of Freud's LLM</h1><p>描述你的生活事件 · 获得朋友般的心理学解读</p></header>
<div class="chat" id="chat"></div>
<div class="input-area"><input id="inp" placeholder="说说你的烦恼或开心事..." autofocus><button id="btn">发送</button></div>
<footer>不能替代专业心理咨询 · 如有严重困扰请寻求专业帮助</footer>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('inp'),btn=document.getElementById('btn');
function add(t,u){const d=document.createElement('div');d.className='msg '+(u?'user':'bot');if(!u)d.innerHTML='<div class="avatar">🧠</div>';d.innerHTML+=t.replace(/\\n/g,'<br>');chat.appendChild(d);chat.scrollTop=chat.scrollHeight}
async function send(){const t=inp.value.trim();if(!t)return;inp.value='';add(t,true)
const th=document.createElement('div');th.className='think';th.textContent='🧠 让我想想...';chat.appendChild(th);chat.scrollTop=chat.scrollHeight;btn.disabled=true
try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});const d=await r.json();th.remove()
let html='';if(d.theory)html+='<div class="tag">'+d.theory+'</div>';html+=d.reply;add(html,false)}
catch(e){th.remove();add('🙁 连接失败',false)}
btn.disabled=false;inp.focus()}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send()});btn.addEventListener('click',send);
add('你好！说说你遇到了什么事，我来帮你分析分析。',false);
</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "")
    theories, enhanced = build_prompt_with_theory(msg)
    theory_names = [t["name"] for t in theories] if theories else []

    ids = tok(tok.apply_chat_template([{"role": "user", "content": enhanced}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=250, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.1, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    reply = clean(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))

    return jsonify({"reply": reply, "theory": ", ".join(theory_names) if theory_names else ""})

if __name__ == "__main__":
    print("\n🌐 打开 http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
