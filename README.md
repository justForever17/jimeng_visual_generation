# Jimeng 视觉生成 MCP 服务器

基于 Volcengine（火山引擎）Jimeng 视觉生成 API 的 MCP 服务器。通过模型上下文协议（MCP）提供图像和视频生成功能。

## 功能

- **文本生成图像（T2I）**：根据文本提示生成图像
- **图像到图像（I2I）**：基于提示和参考图像对图像进行变换
- **多图融合**：合成多张参考图像
- **文本生成视频（T2V）**：根据文本提示生成视频
- **图像到视频（I2V）**：从首帧或首帧+末帧图像生成视频
- **自动本地文件支持**：可传入图像公网URL，同时支持本地文件路径，程序会自动将其转换为 Base64

## 安装

### 选项 1：通过 pip 安装

```bash
pip install jimeng_visual_generation
```

### 选项 2：直接使用 `uvx` 运行（推荐）

无需安装。`uvx` 会自动下载并运行该包：

```bash
uvx jimeng_visual_generation
```

## 在 VS Code / Cursor / Claude Desktop 中的配置

将下列内容添加到你的 MCP 配置文件：

- **VS Code**：`~/.vscode/mcp.json` 或 工作区设置
- **Cursor**：Settings -> MCP Servers
- **Claude Desktop**：`%APPDATA%\Claude\claude_desktop_config.json`

### 示例配置（使用环境变量）

```json
{
  "mcpServers": {
    "jimeng_visual_generation": {
      "command": "uvx",
      "args": ["jimeng_visual_generation"],
      "env": {
        "VOLC_API_KEY": "your_volcengine_api_key_here",
        "VOLC_IMAGE_MODEL": "doubao-seedream-4-5-251128",
        "VOLC_VIDEO_MODEL": "doubao-seedance-1-0-pro-fast-251015"
      }
    }
  }
}
```

### 环境变量

| 变量 | 是否必需 | 描述 |
|------|----------|------|
| `VOLC_API_KEY` | ✅ 必需 | 你的 Volcengine API Key |
| `VOLC_IMAGE_MODEL` | 可选 | 图像模型 ID（默认：`doubao-seedream-4-5-251128`） |
| `VOLC_VIDEO_MODEL` | 可选 | 视频模型 ID（默认：`doubao-seedance-2.0`） |

## 可用工具

### `generate_image`

使用文本提示和可选参考图像生成图像。

**参数：**

- `prompt`（必需）：描述目标图像的文本
- `image_urls`（可选）：参考图像列表（支持 URL、Base64 或本地文件路径）
- `model`（可选）：使用的模型 ID 或 Endpoint ID
- `size`（可选）：图像尺寸比例（支持 "1:1", "16:9", "2K", "4K" 等，禁止使用 `ratio`）

### `generate_video`

创建视频生成任务。支持多种生成模式（包括最新的 Seedance 2.0 多模态输入）：

- **文本生成视频 (T2V)**：不提供图像、视频、音频输入，只提供 `prompt`。
- **首帧/尾帧生视频 (I2V)**：提供 1-2 张图像。
- **多模态参考生视频 (Seedance 2.0)**：可混合提供图像、参考视频和参考音频（支持本地文件自动转换）。

**参数：**

- `prompt`（可选）：视频描述的文本提示词
- `image_urls`（可选）：输入参考图像列表（支持 URL、Base64 或本地文件路径，最多 9 张）
- `video_urls`（可选）：参考视频列表（支持 URL 或本地文件路径，最多 3 个，总时长 ≤ 15s）
- `audio_urls`（可选）：参考音频列表（支持 URL 或本地文件路径，最多 3 个）
- `image_roles`（可选）：为 `image_urls` 显式指定的角色列表（例如 `["reference_image", "first_frame"]`）
- `model`（可选）：使用的模型 ID 或 Endpoint ID
- `ratio`（可选）：宽高比（例如："16:9"、"9:16"，禁止在图片生成里使用该参数）
- `resolution`（可选）：分辨率（"720p" 或 "1080p"）
- `duration`（可选）：视频时长（秒，支持 4-15s）
- `return_last_frame`（可选）：是否返回生成的视频最后一帧图像 URL，适用于连续生成

### `get_video_task_result`

查询视频生成任务的状态和结果。

**参数：**

- `task_id`（必需）：由 `generate_video` 返回的任务 ID

## 📄 开源协议 (License)

本项目基于 [MIT License](LICENSE) 开源。

## 💖 赞助 (Sponsorship)

维护开源项目不易，如果您觉得 jimeng_visual_generation 对您有帮助，欢迎请作者喝杯咖啡！

<div align="center">

| 平台 | 链接 | 支付方式 |
| :--- | :--- | :--- |
| **爱发电 (Afdian)** | [![Afdian](https://img.shields.io/badge/Afdian-支持我-946ce6?logo=afdian)](https://afdian.com/a/justforever17) | 微信, 支付宝 |

</div>
