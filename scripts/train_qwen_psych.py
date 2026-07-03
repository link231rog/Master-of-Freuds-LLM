"""
Fine-tune Qwen2.5-0.5B with LoRA on psychology conversation data.

Usage:
    python scripts/train_qwen_psych.py
"""

import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "D:/MasterOfFreudsLLM/checkpoints/qwen_psych"
DATA_DIR = "D:/MasterOfFreudsLLM/data_psychology"
CACHE_DIR = "D:/MasterOfFreudsLLM/.cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)


def main():
    print("🧠 Master of Freud's LLM — Qwen2.5-0.5B LoRA Fine-tuning")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    dataset = load_jsonl(f"{DATA_DIR}/qwen_psych_train.jsonl")
    dev_dataset = load_jsonl(f"{DATA_DIR}/qwen_psych_dev.jsonl")
    print(f"Train: {len(dataset)} | Dev: {len(dev_dataset)}")

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tok = tokenizer(text, truncation=True, max_length=512, padding=False)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    dataset = dataset.map(tokenize, remove_columns=["messages"])
    dev_dataset = dev_dataset.map(tokenize, remove_columns=["messages"])
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
    dev_dataset = dev_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    # Model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = model.to("cuda")
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        eval_steps=20,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("\n🚀 Training...")
    trainer.train()

    # Save
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"✅ Saved to {OUTPUT_DIR}/final")

    # Quick test
    print("\n🧪 Test:")
    model.eval()
    for prompt in [
        "今天在会议上被领导当众批评了，好几个人都看着，尴尬得要死",
        "最近总是失眠，脑子里停不下来",
    ]:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, temperature=0.8, top_p=0.9, do_sample=True)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nU: {prompt}")
        print(f"M: {reply[:200]}")


if __name__ == "__main__":
    main()
