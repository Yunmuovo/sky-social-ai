# 🌟 光遇社交AI

让AI在光遇中社交：识别椅子、自动坐下、AI对话、添加好友。

## 项目结构

```
sky_social_ai/
├── main.py                  # 主控程序 (状态机)
├── gui.py                   # 悬浮窗控制面板
├── 启动悬浮窗.bat            # 双击启动
├── config.yaml              # 配置文件 (API Key、坐标等)
├── calibrate.py             # 坐标标定工具
├── screen_capture.py        # 截图模块 (PC/Android)
├── chair_detector.py        # 椅子检测 (模板匹配)
├── chat_button_detector.py  # 聊天按钮检测 (左下角气泡)
├── speech_bubble_detector.py# 头顶气泡检测
├── chat_ocr.py              # 聊天文字识别 (OCR)
├── llm_dialogue.py          # AI对话引擎 (DeepSeek)
├── friend_manager.py        # 好友操作管理
├── input_simulator.py       # 输入模拟 (PC/Android)
├── requirements.txt         # 依赖清单
└── templates/               # 模板图片 (需自行准备)
    ├── chair_tea_table.png  # 椅子模板
    ├── chat_button.png      # 聊天按钮模板
    └── speech_bubble.png    # 头顶气泡模板
```

## 快速开始

### 方式一：悬浮窗启动（推荐）

双击 `启动悬浮窗.bat` 或运行：

```bash
python gui.py
```

一个精致的悬浮控制面板会弹出，包含：
- ▶ 启动/停止 按钮
- 实时状态显示（空闲/已坐下/思考中/回复中）
- 日志面板
- 📐 标定 / ⚙ 配置 快捷按钮
- 置顶模式（始终悬浮在其他窗口上方）

### 方式二：命令行启动

```bash
python main.py
```

### 1. 安装依赖

```bash
pip install -r requirements.txt
```
> 注意：PaddleOCR 安装较大（~1GB），如果暂不需要聊天OCR功能，可先跳过。

### 2. 配置 DeepSeek API

编辑 `config.yaml`，填入你的 API Key：

```yaml
deepseek:
  api_key: "sk-xxxxxxxxxxxxx"  # 你的API Key
```

获取 API Key: https://platform.deepseek.com/

### 3. 准备椅子模板

在光遇中截图各种椅子 → 裁剪特征区域 → 放入 `templates/` 目录

### 4. 标定坐标

打开光遇游戏，运行标定工具：

```bash
python calibrate.py
```

按提示框选聊天区域、好友按钮等UI位置。

### 5. 运行

```bash
python main.py
```

```yaml
# 根据实际分辨率调整（3200x2000参考值）
chat_area:
  region: [50, 300, 650, 1200]       # 左侧对话框区域
chat_input:
  region: [450, 1650, 1470, 1780]    # 底部输入条区域
```

## 状态机流程

```
IDLE ──(发现椅子)──> APPROACHING ──(坐下动画)──> SEATED
                                                    │
                          ┌─────────────────────────┤
                          │                  (未打招呼)
                          │                         │
                          │              ┌─ OPENING_CHAT → TYPING → SENDING
                          │              │
                   (扫描头顶气泡) ←──────┘
                          │
                   发现气泡？── 否 ──> 继续等待
                      │ 是
                      ▼
              OPENING_BUBBLE → READING_BUBBLE
                                      │
                                OCR 读取消息
                                      │
                                      ▼
                                 THINKING (AI生成回复)
                                      │
                                      ▼
                              OPENING_CHAT → TYPING → SENDING
                                      │
                                      └──> SEATED (循环)
```

## 需要准备的模板截图

| 模板 | 文件名 | 说明 |
|------|--------|------|
| 椅子 | `templates/chair_tea_table.png` | 茶桌/长桌截图 |
| 聊天按钮 | `templates/chat_button.png` | 坐下后左下角气泡图标 |
| 头顶气泡 | `templates/speech_bubble.png` | 别人头上白色对话气泡 |

在光遇中截好图 → 裁剪 → 放入 `templates/` 目录

## 移植到 Android

1. 确保 adb 已连接手机
2. 修改 `config.yaml` 中 `platform: android`
3. 重新运行 `calibrate.py` 标定
4. 运行 `python main.py`

## 安全提醒

- 本工具仅供学习研究
- 自动化操作可能违反游戏服务条款
- 请勿用于破坏其他玩家的游戏体验
