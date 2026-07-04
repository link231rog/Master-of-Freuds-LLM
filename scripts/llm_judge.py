"""
LLM-as-a-Judge 批量评估脚本
- 先让所有模型回答10个问题
- 再用 GPT-4 打分
- 输出对比矩阵
"""

import os, json, torch, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))
sys.path.insert(0, ROOT_DIR)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".cache")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rag_psychology import build_prompt_with_theory, retrieve_theory

device = "cuda"

# ── 加载训练好的模型 ──
MODELS = {
    "Qwen原版": None,  # 不需要LoRA
    "v3": os.path.join(ROOT_DIR, "checkpoints", "qwen_psych_v3", "final"),
    "v4": os.path.join(ROOT_DIR, "checkpoints", "qwen_psych_v4", "final"),
    "v5": os.path.join(ROOT_DIR, "checkpoints", "qwen_psych_v5", "final"),
}

# ── 10 个测试问题 ──
TEST_QUESTIONS = [
    "面试了好几家都失败了，怀疑自己是不是真的不行",
    "帮了朋友大忙，他连句谢谢都没有",
    "期末周作业和考试堆在一起，压力大到喘不过气",
    "今天面试通过了！好开心",
    "今天心情好好，什么都不想说只想笑",
    "和好朋友吵架了，他已经三天没理我了",
    "总是忍不住跟别人比较，越比越焦虑",
    "部门来了个新领导总感觉他在针对我",
    "总觉得自己不够好，别人夸我是客气",
    "三十岁了还在做基础岗位，觉得人生没希望了",
]

# ── 生成所有模型的回答 ──
def load_model(name, path):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    if name == "Qwen原版":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to(device)
    else:
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to(device),
            path
        )
    model.eval()
    return tok, model

def generate(tok, model, question, use_rag=False):
    if use_rag:
        theories, prompt = build_prompt_with_theory(question)
    else:
        prompt = question
    ids = tok(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=200, temperature=0.85, top_p=0.9, do_sample=True, repetition_penalty=1.1, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return reply

# ── 主流程 ──
results = {}

for name, path in MODELS.items():
    print(f"Generating {name}...")
    tok, model = load_model(name, path)
    results[name] = []
    for q in TEST_QUESTIONS:
        # 非RAG版本
        r1 = generate(tok, model, q, use_rag=False)
        results[name].append({"question": q, "response": r1, "mode": "base"})
        # RAG版本（只有v5有RAG）
        if name == "v5":
            r2 = generate(tok, model, q, use_rag=True)
            results["v5+RAG"] = results.get("v5+RAG", []) + [{"question": q, "response": r2, "mode": "rag"}]

# 保存所有回答
with open(os.path.join(ROOT_DIR, "data", "eval_responses.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"✅ 已保存 {sum(len(v) for v in results.values())} 条回答")

# ── GPT-4 评分部分 ──
# 需要你提供 OpenAI API Key 才能跑
print("""
下一步：用 GPT-4 评分

你需要：
1. 在 https://platform.openai.com 注册，充值 $5
2. 拿到 API Key（sk-...）
3. 运行以下 Python 脚本：

import openai
openai.api_key = "sk-你的key"

for model_name, responses in results.items():
    for r in responses:
        prompt = f'''
你是一位心理学对话评估专家。
请对以下 AI 回答进行评分（1-5分）。

评分维度：
1. 共情度：是否理解用户感受
2. 理论匹配：心理学理论是否合适
3. 实用性：建议是否具体可操作
4. 自然度：像朋友还是像机器人

用户问题：{r["question"]}
AI回答：{r["response"]}

请输出：
共情度：
理论匹配：
实用性：
自然度：
'''
        # 调用 GPT-4 API
        ...
""")
