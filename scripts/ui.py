"""
Master of Freud's LLM — Web UI (单文件版)
运行后访问 http://localhost:5000
"""

import os, json, torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".cache")
device = "cuda"

print("🧠 Master of Freud's LLM loading...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, os.path.join(ROOT_DIR, "checkpoints", "qwen_psych", "best"))
model.eval()
print(f"✅ Loaded! Model is ready.")

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
header{text-align:center;padding:30px 0 16px;border-bottom:1px solid #e0d6cc}
header h1{font-size:24px;color:#2c1810}
header p{font-size:14px;color:#8b7355;margin-top:4px}
.chat{flex:1;overflow-y:auto;padding:20px 0;display:flex;flex-direction:column;gap:16px}
.msg{max-width:88%;padding:14px 18px;border-radius:18px;line-height:1.7;font-size:15px;animation:fadeIn .3s;white-space:pre-wrap}
.user{background:#2c1810;color:#fff;align-self:flex-end;border-radius:18px 4px 18px 18px}
.bot{background:#fff;color:#2c1810;align-self:flex-start;border-radius:4px 18px 18px 18px;box-shadow:0 2px 6px rgba(0,0,0,.06)}
.bot .avatar{font-size:22px;margin-bottom:4px}
.thinking{color:#8b7355;padding:8px 16px;font-size:14px;animation:pulse 1.5s infinite}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.input-area{border-top:1px solid #e0d6cc;padding:16px 0 24px;display:flex;gap:8px}
.input-area input{flex:1;padding:14px 18px;border:1px solid #ddd0c4;border-radius:28px;font-size:15px;outline:none;background:#fff;color:#2c1810}
.input-area input:focus{border-color:#8b7355}
.input-area button{padding:14px 28px;background:#2c1810;color:#fff;border:none;border-radius:28px;font-size:15px;cursor:pointer}
.input-area button:hover{background:#4a2c1a}
.input-area button:disabled{opacity:.4;cursor:not-allowed}
.examples{display:flex;gap:8px;flex-wrap:wrap;padding:10px 0}
.examples button{font-size:12px;padding:6px 14px;border:1px solid #e0d6cc;border-radius:16px;background:#fff;color:#8b7355;cursor:pointer}
.examples button:hover{background:#f0ebe6;border-color:#8b7355}
footer{text-align:center;font-size:11px;color:#c4b8a8;padding:8px 0 16px}
</style>
</head>
<body>
<div class="container">
<header><h1>🧠 Master of Freud's LLM</h1><p>描述你的生活事件 · 获得朋友般的心理学解读</p></header>
<div class="examples" id="exs"></div>
<div class="chat" id="chat"></div>
<div class="input-area"><input id="inp" placeholder="说说你的烦恼..." autofocus><button id="btn">发送</button></div>
<footer>不能替代专业心理咨询 · 如有严重困扰请寻求专业帮助</footer>
</div>
<script>
const exs=["今天在会议上被领导当众批评了，尴尬得要死","最近总是失眠，脑子里停不下来","朋友创业成功了，我嘴上说恭喜心里不是滋味","三十岁了还在做基础岗位，觉得人生没希望了","总是忍不住跟别人比较，越比越焦虑"];
const ec=document.getElementById('exs');
exs.forEach(e=>{const b=document.createElement('button');b.textContent=e.length>14?e.slice(0,14)+'…':e;b.onclick=()=>send(e);ec.appendChild(b)});
const chat=document.getElementById('chat'),inp=document.getElementById('inp'),btn=document.getElementById('btn');
function add(t,u){const d=document.createElement('div');d.className='msg '+(u?'user':'bot');if(!u)d.innerHTML='<div class="avatar">🧠</div>';d.innerHTML+=t.replace(/\\n/g,'<br>');chat.appendChild(d);chat.scrollTop=chat.scrollHeight}
async function send(t){if(!t)t=inp.value.trim();if(!t)return;inp.value='';add(t,true)
const th=document.createElement('div');th.className='thinking';th.textContent='🧠 正在思考…';chat.appendChild(th);chat.scrollTop=chat.scrollHeight;btn.disabled=true
try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});const d=await r.json();th.remove();add(d.reply,false)}
catch(e){th.remove();add('🙁 连接失败，请确认服务器正在运行',false)}
btn.disabled=false;inp.focus()}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send()});btn.addEventListener('click',()=>send());
add('你好！我是弗洛伊德大师。说说你遇到了什么事，我来帮你分析分析。',false);
</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "")
    msgs = [{"role": "user", "content": msg}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=150, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return jsonify({"reply": reply.strip()})

if __name__ == "__main__":
    print("\n🌐 打开 http://localhost:5000 开始对话！")
    app.run(host="127.0.0.1", port=5000, debug=False)
