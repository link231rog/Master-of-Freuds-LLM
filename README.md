# 🧠 Master of Freud's LLM

**让 0.5B 小模型学会有「人味儿」的心理学分析**

> 基于 Qwen2.5-0.5B + LoRA + RAG 的人机协同迭代微调项目
> 
> 全部工作在 **RTX 4060 (8GB)** 上 **~2 小时** 内完成
> 
> 综合评分从基座 1.3 → **v5+RAG 3.5（+169%）**，理论匹配超过 DeepSeek

---

## 📖 项目故事

大多数人做心理咨询 AI 的方法是：收集一堆心理咨询对话数据，直接丢进去微调，训完就完了。出来的模型满口教科书味儿——

*"我理解你的感受，根据认知行为理论来看……"*

我不这样做。我迭代了 **五轮**，从最机械的模板一路磨到能说人话。

### 迭代历程

| 版本 | 数据量 | 核心改进 | 问题 | 效果 |
|:---:|:---:|:---|:---|:---:|
| **v1** | 538 条 | 程序生成模板对话 | 🤖 太机械，列表式输出 | — |
| **v2** | 4,341 条 | 口语化 + 共情开头 | 📖 像背课文 | — |
| **v3** | 12,000 条 | 引入 15 个心理学理论 | 🎓 像在讲课，没个人风格 | 综合 2.5 |
| **v4** | 4,900 条 | 5 条手标样本做风格迁移 | ✅ 终于像人话了 | 综合 2.8 |
| **v5** | 4,900 条 | 标签匹配系统 + 指令跟随 | 🎯 理论匹配更准 | 综合 3.3 |
| **v5+RAG** | — | 检索增强生成 | 🏆 **理论匹配 3.5 超过 DeepSeek** | **综合 3.5** |

---

## 🏗️ 项目结构

```
MasterOfFreudsLLM/
├── scripts/                    # 🔥 核心脚本
│   ├── train_v5.py             # v5 LoRA 微调 Qwen2.5-0.5B
│   ├── generate_v5_data.py     # v5 带标签数据生成
│   ├── rag_psychology.py       # RAG 心理学理论数据库 + 检索
│   ├── chat_v5_rag.py          # v5+RAG 交互式聊天
│   ├── chat_psychology.py      # 完整心理学分析聊天界面
│   ├── chat_final.py           # 最终版推理（过滤+RAG+思考）
│   ├── llm_judge.py            # LLM-as-a-Judge 批量评测
│   └── ...                     # v1-v4 训练脚本
├── data_psychology/            # 心理学训练数据（v3/v4/v5）
├── checkpoints/                # 🚫 LoRA 权重（gitignored）
├── configs/                    # 训练配置文件
├── data/                       # 评测结果与评分数据
├── paper/                      # 学术论文
├── docs/                       # 文档
└── ui/                         # Web UI 资产
```

---

## 🔬 训练过程详解

### 数据格式演变

**v1 — 模板式（太机械）**
```
【心理学分析】
1. 你的问题属于社交焦虑范畴
2. 建议你尝试认知重构
3. 推荐相关书籍
```

**v2 — 口语化（像背书）**
```
说得对，这种感觉真的挺让人难受的。
你是不是觉得每次在人群里都特别不自在？
其实很多人在这种场合都会紧张，你不是一个人。
我建议你可以先试试……
```

**v3 — 理论增强（像讲课）**
```
根据聚光灯效应（康奈尔大学基洛维奇提出），
人们总觉得自己站在舞台中央，每一个细节都被放大检视。
但实际上每个人最关注的是他们自己，不是你。
所以……
```

**v4 — 风格迁移 ✅（人味儿）**
```
啧，刷朋友圈刷出来的焦虑是吧？
我跟你说个好玩的理论——聚光灯效应，
其实是心理学家说我们总觉得自己是人群的焦点，
但实际上别人根本没那么关注你。
你看到的朋友圈都是别人想让你看的精选集，
不是他们的真实人生。
```

**v5 — 标签匹配 + 指令跟随**
```json
{"messages": [
  {"role": "user", "content": "面试了好几家都失败了，怀疑自己是不是真的不行"},
  {"role": "assistant", "content": "请用习得性无助来分析……"}
]}
```

### 标签匹配系统

每个问题场景和每个心理学理论都打上情绪标签，只有标签配对才使用：

```
问题: "面试失败" → 标签: [work, self, achievement]
理论: "习得性无助" → 标签: [work, self, achievement]  ✅ 匹配
理论: "达克效应"   → 标签: [social, achievement]       ❌ 不匹配
```

### RAG 检索增强

模型不靠自己硬记理论，而是实时去数据库查：

```python
# rag_psychology.py 核心逻辑
theories, enhanced_prompt = build_prompt_with_theory(user_input)
# 将匹配的理论注入 prompt
# → "请用【习得性无助】理论来分析：面试失败了……"
```

---

## 📊 实验结果

### 综合评分（LLM-as-a-Judge，满分 5）

| 模型 | 共情度 | 理论匹配 | 实用性 | 自然度 | **综合** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Qwen 原版 | 1.4 | 1.0 | 1.4 | 1.2 | **1.3** |
| v3 | 3.0 | 2.3 | 2.1 | 2.4 | **2.5** |
| v4 | 3.6 | 3.1 | 2.0 | 2.3 | **2.8** |
| v5 | 3.9 | 2.9 | 3.0 | 3.3 | **3.3** |
| **v5+RAG** | **3.8** | **3.5** 🏆 | **3.3** | **3.4** | **3.5** |
| DeepSeek | 5.0 | 3.4 | 3.8 | 5.0 | 4.3 |

### RAG 效果对比

| 用户问题 | 无 RAG | 有 RAG |
|:---|:---|:---|
| 面试失败 | 达克效应 ❌ | 习得性无助 ✅ |
| 朋友不道谢 | 基本归因错误 ❌ | 互惠规范 ✅ |
| 期末压力大 | 自我实现预言 ❌ | 自我损耗 ✅ |
| 领导针对我 | 聚光灯效应 ❌ | 基本归因错误 ✅ |

---

## 🚀 快速开始

### 环境要求

- GPU: RTX 3060+ (8GB+ VRAM)
- Python 3.9+
- PyTorch 2.0+ (CUDA)

### 安装

```bash
# 克隆
git clone https://github.com/<你的用户名>/MasterOfFreudsLLM.git
cd MasterOfFreudsLLM

# 安装依赖
pip install torch transformers peft flask
pip install -r requirements.txt
```

### 训练（从头训 v5）

```bash
# 1. 生成带标签的训练数据
python scripts/generate_v5_data.py

# 2. LoRA 微调 Qwen2.5-0.5B（~2 小时）
python scripts/train_v5.py
```

### 运行

```bash
# 命令行聊天
python scripts/chat_v5_rag.py

# 启动 Web UI（浏览器打开 http://localhost:5000）
python scripts/webui.py
```

### 评测

```bash
python scripts/llm_judge.py
```

---

## 📋 核心脚本说明

| 脚本 | 功能 |
|:---|:---|
| `generate_v5_data.py` | 标签匹配数据生成器，50+ 场景，15 个心理学理论 |
| `train_v5.py` | LoRA 微调（r=16, α=32, 800 steps, ~2h on RTX 4060） |
| `rag_psychology.py` | 心理学理论数据库 + 关键词检索 |
| `chat_v5_rag.py` | 交互式命令行聊天（推荐） |
| `chat_psychology.py` | 完整版聊天（含系统提示词、对话历史、保存功能） |
| `chat_final.py` | 最终版推理（输入过滤 + RAG + 思考可视化） |
| `webui.py` | Flask Web 界面 |
| `llm_judge.py` | LLM-as-a-Judge 四维度自动评测 |

---

## 📄 引用

如果你在研究中使用了本项目：

```bibtex
@misc{masteroffreudsllm2025,
  author = {晁辰熙},
  title = {Master of Freud's LLM: 基于人机协同迭代微调与检索增强的个性化生活心理学分析系统},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/<你的用户名>/MasterOfFreudsLLM}
}
```

---

## ⚠️ 免责声明

本模型 **不能替代专业心理咨询或治疗**。如果你或身边的人正在经历严重的心理困扰，请寻求专业帮助。

---

## 📝 许可

- **代码**：Apache 2.0
- **模型权重**：Apache 2.0
- **训练数据**：合成数据，公共领域

---

*"The mind is like an iceberg, it floats with one-seventh of its bulk above water."* — Sigmund Freud
