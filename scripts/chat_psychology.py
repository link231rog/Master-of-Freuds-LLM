"""
Master of Freud's LLM — Interactive Psychology Analysis Interface

A command-line chat interface that analyzes everyday life events
from a psychological perspective.

Usage:
    # With a trained checkpoint
    PYTHONPATH=. python scripts/chat_psychology.py \
        --ckpt /ephemeral/ckpts/freud_sft.pt

    # With the base model (raw completion mode)
    PYTHONPATH=. python scripts/chat_psychology.py \
        --ckpt /ephemeral/ckpts/freud_base_pretrained.pt --raw

    # CPU mode (slow but works without GPU)
    PYTHONPATH=. python scripts/chat_psychology.py \
        --ckpt /ephemeral/ckpts/freud_sft.pt --device cpu

Controls:
    /help      Show this help
    /clear     Clear screen
    /examples  Show example life events to analyze
    /save      Save current conversation to file
    /quit      Exit
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.transformer import Transformer
from src.post_training.chat_template import decode, get_tokenizer
from src.post_training.inference import generate_reply

# ── Color helpers ──
if os.name == "nt":  # Windows
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    _ENABLE_COLOR = True
else:
    _ENABLE_COLOR = True


def _color(text: str, code: str) -> str:
    if not _ENABLE_COLOR:
        return text
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return f"{colors.get(code, '')}{text}{colors['reset']}"


# ── Example events for inspiration ──
EXAMPLE_EVENTS = [
    "我今天在会议上提了一个建议，被领导当众否决了，感觉很没面子，以后都不想发言了。",
    "我最近总是失眠，躺在床上脑子停不下来，反复想白天说过的话做过的事。",
    "女朋友三天没主动找我了，我一直在想是不是我做错了什么。",
    "看到朋友圈里大家都在旅游、升职、结婚，而我什么都没有，感到很焦虑。",
    "我今天和一个陌生人聊得很开心，但我很奇怪为什么我总能对陌生人敞开心扉，却对亲近的人有所保留。",
    "我每次考试前都会肚子疼，但一考完就好了，这是怎么回事？",
    "我总是不自觉地拖延，明明知道 deadline 要到了还是不想开始。",
    "做了一个奇怪的梦，梦到自己在一个陌生的城市里迷路了，怎么都找不到回家的路。",
    "今天不小心把同事的名字叫错了，叫成了前任的名字，场面非常尴尬。",
    "我发现自己总是被同一类人吸引，虽然知道和这种人不合适但就是控制不住。",
    "妈妈总是拿我和别人家的孩子比较，我很烦但又不知道怎么和她说。",
    "我最近换了一份新工作，觉得自己什么都不会，每天都在怀疑自己是不是能力不行。",
]

# ── System prompts ──
SYSTEM_PROMPT = (
    "你是一位精通心理学各个流派的大师。你的专长是分析日常生活中的心理现象，"
    "用通俗易懂的语言解释背后的心理学原理，并给出实用的改善建议。\n\n"
    "你的分析风格：\n"
    "1. 先共情：理解对方的感受\n"
    "2. 再分析：指出背后的心理学概念和机制\n"
    "3. 给建议：提供具体可操作的方法\n"
    "4. 联系经典：适当引用心理学理论和研究\n\n"
    "注意：不用过于学术化，要用生活化的语言让普通人也能听懂。"
    "对于严重心理问题的描述，要提醒寻求专业帮助。"
)


def load_model(ckpt_path: str, device: str) -> Transformer:
    """Load model from checkpoint, supporting our custom 85M config."""
    print(f"  Loading checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {}) or {}

    # Extract model dimensions from cfg or fallback to Freud 85M defaults
    n_head = cfg.get("n_head", 12)
    n_embed = cfg.get("n_embed", 768)
    context_length = cfg.get("context_length", 512)
    vocab_size = cfg.get("vocab_size", 50304)
    n_blocks = cfg.get("n_blocks", 12)

    model = Transformer(
        n_head=n_head,
        n_embed=n_embed,
        context_length=context_length,
        vocab_size=vocab_size,
        N_BLOCKS=n_blocks,
    )

    state = ck.get("model_state_dict", ck)
    state = {k.removeprefix("module.").removeprefix("transformer."): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print(f"  Architecture: {n_blocks} layers, {n_embed} embed, {n_head} heads")
    print(f"  Context length: {context_length}")
    return model


def print_header():
    """Print the welcome banner."""
    header = f"""
{_color('╔══════════════════════════════════════════════════════╗', 'cyan')}
{_color('║', 'cyan')}  {_color('🧠 Master of Freud\'s LLM', 'bold')}               {_color('║', 'cyan')}
{_color('║', 'cyan')}  {_color('心理学大师 · 生活事件分析', 'yellow')}         {_color('║', 'cyan')}
{_color('║', 'cyan')}  {_color('描述你的生活事件，获得专业的心理学解读', 'dim')}  {_color('║', 'cyan')}
{_color('╚══════════════════════════════════════════════════════╝', 'cyan')}
"""
    print(header)
    print(f"  {_color('输入 /help 查看命令  |  /examples 看示例事件', 'dim')}")
    print()


def print_analysis(text: str):
    """Print analysis with nice formatting."""
    print()
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            print()
        elif line.startswith("【") and "】" in line:
            print(f"  {_color(line, 'bold')}")
        elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
            print(f"    {_color(line, 'cyan')}")
        elif line.startswith(("**",)):
            print(f"    {_color(line, 'yellow')}")
        else:
            print(f"  {line}")
    print()


def print_examples():
    """Print example events."""
    print(f"\n  {_color('📋 示例生活事件（复制粘贴到输入框）:', 'bold')}")
    print()
    for i, ev in enumerate(EXAMPLE_EVENTS, 1):
        print(f"  {_color(f'{i}.', 'dim')} {ev}")
    print()


def run_interactive(model, device: str, raw_mode: bool, temperature: float):
    """Main interactive loop."""
    history: list[dict] = []
    print_header()

    while True:
        try:
            user_input = input(f"\n{_color('你', 'green')} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_color('再见！', 'magenta')}")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input == "/quit":
            print(f"\n{_color('再见！愿你的心灵之旅继续。', 'magenta')}")
            break
        elif user_input == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_header()
            continue
        elif user_input == "/help":
            print(f"""
  {_color('命令列表:', 'bold')}
    {_color('/help', 'yellow')}     显示此帮助
    {_color('/clear', 'yellow')}    清屏
    {_color('/examples', 'yellow')} 显示示例生活事件
    {_color('/save', 'yellow')}     保存对话记录到文件
    {_color('/quit', 'yellow')}     退出

  {_color('使用说明:', 'bold')}
    直接输入你遇到的生活事件或心理困惑，
    模型会从心理学角度给你分析和建议。

  {_color('示例:','bold')}
    「我今天在会上被批评了，很难受」
    「最近总是失眠，脑子里停不下来」
    「做了一个奇怪的梦...」
""")
            continue
        elif user_input == "/examples":
            print_examples()
            continue
        elif user_input == "/save":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"freud_conversation_{timestamp}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# Master of Freud's LLM - Conversation\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for h in history:
                    role = "👤 User" if h["role"] == "user" else "🧠 Freud"
                    f.write(f"### {role}\n\n{h['content']}\n\n")
            print(f"  {_color(f'✓ 对话已保存到 {filename}', 'green')}")
            continue

        # ── Generate analysis ──
        print(f"\n  {_color('🧠 弗洛伊德大师正在分析...', 'yellow')}")

        # Store in history
        history.append({"role": "user", "content": user_input})

        try:
            response = generate_reply(
                model,
                user_input,
                device=device,
                system=SYSTEM_PROMPT if not raw_mode else None,
                raw=raw_mode,
                max_new_tokens=300,
                temperature=temperature,
                top_p=0.95,
                top_k=None,
            )

            print()
            print_analysis(response)
            history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"\n  {_color(f'✗ 生成出错: {e}', 'red')}")
            print(f"  {_color('提示: 如果是显存不足，可以重启程序。', 'dim')}")


def main():
    parser = argparse.ArgumentParser(description="Master of Freud's LLM - Interactive Psychology Analysis")
    parser.add_argument("--ckpt", default="/ephemeral/ckpts/freud_sft.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run on")
    parser.add_argument("--raw", action="store_true",
                        help="Raw completion mode (for base model, not SFT)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Generation temperature (0.0-1.5)")
    parser.add_argument("--cpu-fallback", action="store_true",
                        help="Fall back to CPU if CUDA is unavailable")
    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if args.cpu_fallback:
            print(f"  {_color('CUDA not available, falling back to CPU...', 'yellow')}")
            device = "cpu"
        else:
            print(f"  {_color('CUDA not available! Use --device cpu or --cpu-fallback', 'red')}")
            sys.exit(1)

    # Check checkpoint exists
    if not os.path.exists(args.ckpt):
        print(f"  {_color(f'Checkpoint not found: {args.ckpt}', 'red')}")
        print(f"  You need to train the model first, or download a pretrained checkpoint.")
        print(f"  Run with --help for options.")
        sys.exit(1)

    # Load model
    print(f"\n  {_color('🔮 Loading Master of Freud\'s LLM...', 'bold')}")
    model = load_model(args.ckpt, device)
    print(f"  {_color('✓ Ready! Describe a life event and I shall analyze.', 'green')}")
    print()

    # Run
    run_interactive(model, device, args.raw, args.temperature)


if __name__ == "__main__":
    main()
