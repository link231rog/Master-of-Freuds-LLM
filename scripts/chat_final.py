"""
Master of Freud's LLM — Final Chat with RAG + Filter + Thinking
"""

import os, torch, re
os.environ["HF_HOME"] = "D:/MasterOfFreudsLLM/.cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scripts.rag_psychology import build_prompt_with_theory

# ── Load model ──
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

# Try v5 first, fall back to v4
model_path = "D:/MasterOfFreudsLLM/checkpoints/qwen_psych_v5/final"
if not os.path.exists(model_path):
    model_path = "D:/MasterOfFreudsLLM/checkpoints/qwen_psych_v4/final"
    print("⚠️ v5 not ready yet, using v4")

model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to("cuda"),
    model_path
)
model.eval()

def clean(text):
    return re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u0020-\u007e\n，。！？、；：""''（）【】《》\-]', "", text).strip()

# ── Psychology-related keywords ──
PSYCH_KEYWORDS = [
    "心情", "难过", "焦虑", "压力", "失眠", "烦恼", "抑郁", "生气", "害怕",
    "紧张", "伤心", "委屈", "迷茫", "烦躁", "崩溃", "累", "疲惫", "孤独",
    "无助", "绝望", "自卑", "尴尬", "丢脸", "后悔", "内疚", "担心",
    "想多", "不开心", "没意思", "没劲", "提不起劲", "不想动",
    "朋友", "同事", "家人", "父母", "对象", "女朋友", "男朋友",
    "分手", "吵架", "沟通", "关系", "社交", "聚会", "群",
    "工作", "考试", "面试", "考研", "期末", "作业", "毕业",
    "比较", "别人", "比不上", "加班", "升职", "工资", "辞职",
    "结婚", "催婚", "买房", "年龄", "三十", "未来",
    "性格", "习惯", "拖延", "懒", "放弃", "坚持",
    "梦", "想法", "感受", "心态", "情绪", "心理", "咨询",
]

def is_psychology_question(text):
    """Check if the question is psychology/emotion related."""
    text_lower = text.lower()
    for kw in PSYCH_KEYWORDS:
        if kw in text:
            return True
    return False

# ── Thinking phrases ──
THINKING_PHRASES = [
    "嗯……让我想想该怎么回答你这个问题。",
    "我理解你的意思，让我想一想……",
    "好的，我大概明白你的情况了，我在想该怎么跟你说。",
    "嗯，我在思考你的问题……",
]

# ── Main loop ──
import random

print("🧠 Master of Freud's LLM v5 — 已就绪！")
print("📌 我只回答心理和情绪类问题。输入 /quit 退出\n")

while True:
    q = input("\n😶 你 > ").strip()
    if not q: continue
    if q == "/quit": print("👋 再见！"); break
    
    # ── Filter ──
    if not is_psychology_question(q):
        print("🧠 嗯？这个问题好像不属于心理或情绪类的范畴哦。")
        print("   我只擅长分析生活中的心理现象和情绪问题。")
        print("   你可以跟我说说最近遇到了什么烦恼，或者什么事情让你心情不好？")
        continue
    
    # ── Thinking visualization ──
    thinking = random.choice(THINKING_PHRASES)
    print(f"🧠 {thinking}")
    
    # ── Generate with RAG ──
    theories, enhanced = build_prompt_with_theory(q)
    
    ids = tok(tok.apply_chat_template([{"role": "user", "content": enhanced}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=250, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.1, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    reply = clean(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
    
    # Try to remove theory instruction from output if model repeats it
    if "请用" in reply[:20]:
        reply = reply.split("：", 1)[-1] if "：" in reply else reply
    
    print(f"   {reply}")
