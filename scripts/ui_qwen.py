"""
Master of Freud's LLM — Web UI
浏览器打开 http://localhost:7860 即可对话
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_HOME"] = "D:/MasterOfFreudsLLM/.cache"
device = "cuda"

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


def chat(message, history):
    """Generate response."""
    msgs = [{"role": "user", "content": message}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=200,
            temperature=0.85,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
        )
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return reply.strip()


# Custom CSS for a nice look
css = """
#chatbot { height: 500px; overflow-y: auto; }
h1 { text-align: center; }
footer { display: none !important; }
"""

with gr.Blocks(title="Master of Freud's LLM", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Master of Freud's LLM
    ### 弗洛伊德大师 · 生活心理学分析
    描述你的生活事件或心理困惑，获得朋友般的心理学解读
    """)

    chatbot = gr.ChatInterface(
        chat,
        title="",
        description="",
        theme=gr.themes.Soft(),
        examples=[
            ["今天在会议上被领导当众批评了，好几个人都看着，尴尬得要死"],
            ["最近总是失眠，脑子里停不下来"],
            ["朋友创业成功了，我嘴上说恭喜心里不是滋味"],
            ["跟女朋友吵架了，明明是她不对但最后哄人的是我"],
            ["三十岁了还在做基础岗位，觉得人生没希望了"],
            ["总是忍不住跟别人比较，越比越焦虑"],
        ],
    )

    gr.Markdown("""
    ---
    ⚠️ **免责声明**：本模型不能替代专业心理咨询。如有严重心理困扰，请寻求专业帮助。
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
