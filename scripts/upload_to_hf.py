"""
Upload v5 model to HuggingFace and create a Space.
Run this script after setting your HF_TOKEN.

Usage:
    set HF_TOKEN=hf_your_token_here
    python upload_to_hf.py
"""
import os, shutil, json
from huggingface_hub import HfApi, create_repo, logout

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen_psych_v5", "final")
SPACE_DIR = os.path.join(ROOT_DIR, "space")
USERNAME = "MasterFreudLLM"  # 你可以改成自己的用户名
MODEL_REPO = f"{USERNAME}/qwen_psych_v5"
SPACE_REPO = f"{USERNAME}/freud-psych-analysis"

api = HfApi()

# 1. Upload model
print("Creating model repo...")
create_repo(MODEL_REPO, exist_ok=True, private=False)
api.upload_folder(folder_path=MODEL_DIR, repo_id=MODEL_REPO)
print(f"✅ Model uploaded: https://huggingface.co/{MODEL_REPO}")

# 2. Create Space
print("Creating Space...")
create_repo(SPACE_REPO, repo_type="space", exist_ok=True, private=False, space_sdk="gradio")
api.upload_folder(folder_path=SPACE_DIR, repo_id=SPACE_REPO, repo_type="space")
print(f"\n🎉 Done! Your app is live at: https://huggingface.co/spaces/{SPACE_REPO}")
