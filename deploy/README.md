# gpu2 部署指南

## 架构

```
Client (OpenAI SDK)  ──▶  new-api (:3000)  ──▶  Model Server (:8000)
                               │                        │
                          API 管理/分发            Qwen2.5-0.5B + LoRA
                          内网认证/计费            + RAG 心理学理论
```

## 前置条件

- **gpu2** 服务器，已安装 Docker 和 NVIDIA Container Toolkit
- 微调后的 LoRA checkpoint 已存在服务器上（默认路径：`/data/checkpoints/qwen_psych_v5/final`）
- 网络环境可访问 HuggingFace（或已配置镜像）

## 部署步骤

### 1. 在 gpu2 上拉取代码

```bash
git clone https://github.com/link231rog/Master-of-Freuds-LLM.git
cd Master-of-Freuds-LLM
```

### 2. 确认 checkpoint 路径

微调后的 LoRA 权重需在服务器上：

```bash
# 检查目录结构
ls -la /data/checkpoints/qwen_psych_v5/final/
# 应包含: adapter_config.json, adapter_model.safetensors, ...
```

如果路径不同，编辑 `deploy/docker-compose.yml` 修改 volumes：

```yaml
volumes:
  - /你的/checkpoints/路径:/checkpoints
```

### 3. 启动服务

```bash
cd deploy
docker compose up -d
```

首次启动会自动：
1. 构建模型服务器镜像（安装 PyTorch、transformers、peft 等）
2. 拉取 new-api 镜像
3. 启动两个容器

检查日志：
```bash
docker compose logs -f model-server  # 模型加载进度
docker compose logs -f new-api        # 网关日志
```

模型加载成功标志：
```
[model-server] ✅ 546M params | device=cuda
[model-server] 🌐 http://0.0.0.0:8000  model_id=qwen-psych-v5
```

### 4. 配置 new-api

服务启动后，访问 **http://gpu2-ip:3000**（默认账号 `root` / 密码 `123456`）

**添加通道（Channel）：**

1. 登录 → 左侧菜单 **「通道」** → **「添加通道」**
2. 填写：
   - **名称**：`弗洛伊德模型`
   - **类型**：`Custom`（自定义）
   - **地址**：`http://model-server:8000`
   - **模型**：`qwen-psych-v5`
   - **分组**：`default`
3. 点击 **「提交」**

**添加令牌（Token）：**

1. 左侧菜单 **「令牌」** → **「添加令牌」**
2. 填写名称，选择模型 `qwen-psych-v5`
3. 提交后复制 `sk-xxx` 格式的密钥

### 5. 客户端使用

任何支持 OpenAI API 的工具/客户端，配置：

```bash
API_BASE_URL=http://gpu2-ip:3000/v1
API_KEY=sk-xxx   # 上一步获取的密钥
MODEL=qwen-psych-v5
```

**示例（Python）：**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://gpu2-ip:3000/v1",
    api_key="sk-xxx"
)

response = client.chat.completions.create(
    model="qwen-psych-v5",
    messages=[{"role": "user", "content": "最近工作压力很大，总是睡不着"}]
)
print(response.choices[0].message.content)
```

**示例（curl）：**

```bash
curl http://gpu2-ip:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-psych-v5",
    "messages": [{"role": "user", "content": "最近工作压力很大，总是睡不着"}]
  }'
```

## 环境变量参考

### Model Server

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B` | 基座模型名称 |
| `MODEL_CHECKPOINT` | `/checkpoints/qwen_psych_v5/final` | LoRA adapter 路径（容器内） |
| `MODEL_ID` | `qwen-psych-v5` | API 中暴露的模型名称 |
| `DEVICE` | `cuda` | 计算设备 |
| `MODEL_PORT` | `8000` | 监听端口 |
| `HF_HOME` | `/cache` | HuggingFace 缓存目录 |
| `MAX_NEW_TOKENS` | `250` | 最大生成长度 |

### new-api

详见：[官方文档](https://docs.newapi.pro/en/docs/installation/config-maintenance/environment-variables)

## 常见问题

### Q: 模型加载失败（OOM）

降低精度或限制 GPU 数量：编辑 `docker-compose.yml` 添加环境变量：

```yaml
environment:
  - DEVICE=cuda
  # 或改用 CPU 测试连通性
  # - DEVICE=cpu
```

### Q: new-api 提示 "channel not found"

检查通道配置中模型名是否为 `qwen-psych-v5`，与 `MODEL_ID` 一致。

### Q: 需要 nvidia-container-toolkit

```bash
# Ubuntu/Debian
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

验证：
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```
