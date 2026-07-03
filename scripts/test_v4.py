"""
Test v4 model on 10 psychology questions
"""
import os, json, torch
os.environ["HF_HOME"] = "D:/MasterOfFreudsLLM/.cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to(device)
model = PeftModel.from_pretrained(model, "D:/MasterOfFreudsLLM/checkpoints/qwen_psych_v4/final")
model.eval()

questions = [
    "和好朋友吵架了，他已经三天没理我了，我要不要主动找他？",
    "辞职创业的同事发了条朋友圈说感谢当初勇敢的自己，我还在格子间加班，心里很不是滋味。",
    "每次跟妈妈打电话她都要问有对象了吗，我说没有她就开始叹气，搞得我现在都不敢打电话回家了。",
    "部门来了个新领导，新官上任三把火全烧在我头上，总觉得他在针对我。",
    "我把一个很重要的项目搞砸了，虽然领导说没事，但我自己过不去这个坎。",
    "和女朋友在一起五年了，她家里开始催婚，但我还没准备好，又不敢跟她说。",
    "我好像不管怎么努力，都只能做到还行，永远做不到最好。",
    "上次在群里提了个建议被同事怼了，现在在群里说话都要犹豫半天。",
    "最近总是莫名其妙想哭，明明没什么特别的事发生，就是觉得心里空落落的。",
    "朋友拉我一起搞副业，我挺心动的，但又怕赔钱又怕没时间。",
]

for i, q in enumerate(questions, 1):
    msgs = [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=250, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*50}")
    print(f"😶 Q{i}: {q}")
    print(f"{'='*50}")
    print(f"🧠 {reply.strip()}")
