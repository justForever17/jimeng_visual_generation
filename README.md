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
        "VOLC_IMAGE_MODEL": "doubao-seedream-4.5",
        "VOLC_VIDEO_MODEL": "doubao-seedance-1.5-pro-251215"
      }
    }
  }
}
```

### 环境变量

| 变量 | 是否必需 | 描述 |
|------|----------|------|
| `VOLC_API_KEY` | ✅ 必需 | 你的 Volcengine API Key |
| `VOLC_IMAGE_MODEL` | 可选 | 图像模型 ID（默认：doubao-seedream-4.5） |
| `VOLC_VIDEO_MODEL` | 可选 | 视频模型 ID（默认：doubao-seedance-1.5-pro） |

## 可用工具

### `generate_image`

使用文本提示和可选参考图像生成图像。

**参数：**

- `prompt`（必需）：描述目标图像的文本
- `image_urls`（可选）：参考图像列表（支持 URL、Base64 或本地文件路径）
- `model`（可选）：使用的模型 ID
- `size`（可选）：图像尺寸（例如："2048x2048"、"2K"、"4K"）

### `generate_video`

创建视频生成任务。根据输入自动判断模式：

- 无图像 → 文本生成视频（T2V）
- 1 张图像 → 首帧 I2V
- 2 张图像 → 首帧 & 末帧 I2V

**参数：**

- `prompt`（可选）：视频描述的文本
- `image_urls`（可选）：输入图像（支持 URL、Base64 或本地文件路径）
- `model`（可选）：使用的模型 ID
- `ratio`（可选）：宽高比（例如："16:9"、"9:16"）
- `duration`（可选）：视频时长（秒）

### `get_video_task_result`

查询视频生成任务的状态和结果。

**参数：**

- `task_id`（必需）：由 `generate_video` 返回的任务 ID

## 许可证

MIT
