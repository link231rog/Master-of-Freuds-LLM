#!/usr/bin/env python3
"""
Master of Freud's LLM — OpenAI-compatible model server.
Exposes fine-tuned Qwen2.5-0.5B + LoRA with RAG psychology theory as
OpenAI Chat Completion API for integration with new-api gateway.

Endpoints:
  GET  /v1/models              → model list
  POST /v1/chat/completions    → OpenAI-compatible chat
  GET  /health                 → health check

Env vars:
  MODEL_NAME        base model name             (default: Qwen/Qwen2.5-0.5B)
  MODEL_CHECKPOINT  path to LoRA checkpoint     (default: checkpoints/qwen_psych_v5/final)
  MODEL_ID          model name exposed via API   (default: qwen-psych-v5)
  DEVICE            cuda / cpu                   (default: cuda if available)
  HOST              bind address                 (default: 0.0.0.0)
  MODEL_PORT        listen port                  (default: 8000)
  HF_HOME           huggingface cache dir        (default: .cache)
  MAX_NEW_TOKENS    max generation tokens        (default: 250)
"""
import os, sys, json, time, re, torch
from flask import Flask, request, jsonify

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))
from rag_psychology import build_prompt_with_theory

# ── config ──────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
CHECKPOINT_DIR = os.getenv("MODEL_CHECKPOINT", os.path.join(ROOT_DIR, "checkpoints", "qwen_psych_v5", "final"))
MODEL_ID = os.getenv("MODEL_ID", "qwen-psych-v5")
HF_HOME = os.getenv("HF_HOME", os.path.join(ROOT_DIR, ".cache"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("MODEL_PORT", "8000"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "250"))

os.environ["HF_HOME"] = HF_HOME

# ── load model ──────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print(f"[model-server] Loading {MODEL_NAME} on {DEVICE}...", flush=True)
try:
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(base, CHECKPOINT_DIR)
    model = model.to(DEVICE)
    model.eval()
    print(f"[model-server] ✅ {sum(p.numel() for p in model.parameters())/1e6:.0f}M params | device={DEVICE}", flush=True)
except Exception as e:
    print(f"[model-server] ❌ Failed to load model: {e}", flush=True)
    sys.exit(1)

app = Flask(__name__)

# ── helpers ─────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Keep CJK, common punct, basic ASCII. Drop control chars."""
    return re.sub(
        # keep CJK, fullwidth, basic ASCII, common punct
        r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u0020-\u007e\n，。！？、；：\u201c\u201d\u2018\u2019（）【】《》\-]",
        "", text
    ).strip()


def _build_usage(input_ids, output_ids) -> dict:
    return {
        "prompt_tokens": input_ids.shape[1],
        "completion_tokens": output_ids.shape[1] - input_ids.shape[1],
        "total_tokens": output_ids.shape[1],
    }


# ── endpoints ───────────────────────────────────────────────────────

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "master-of-freuds",
        }]
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"error": "messages is required"}), 400

    # ── RAG theory enhancement on last user message ──
    last_uidx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_uidx = i
            break

    theory_names = []
    if last_uidx is not None:
        theories, enhanced = build_prompt_with_theory(messages[last_uidx]["content"])
        if theories:
            theory_names = [t["name"] for t in theories]
            messages[last_uidx] = {"role": "user", "content": enhanced}

    # ── build prompt & tokenize ──
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)

    # generation params
    max_tokens = min(body.get("max_tokens", MAX_NEW_TOKENS), 1024)
    temperature = body.get("temperature", 0.85)
    top_p = body.get("top_p", 0.9)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    reply = clean(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

    # prepend matched theory info
    if theory_names:
        reply = f"[理论: {', '.join(theory_names)}]\n{reply}"

    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": _build_usage(inputs.input_ids, out),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_ID, "device": DEVICE})


# ── main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[model-server] 🌐 http://{HOST}:{PORT}  model_id={MODEL_ID}", flush=True)
    app.run(host=HOST, port=PORT, debug=False)
