# 10 — Nexus（WebUI 与外部接口）

Nexus 是**唯一**的外部接口。内部子系统从不将自身暴露给网络
——Nexus 负责翻译。

## WebSocket 通道

- **音频上行**：浏览器麦克风的原始 PCM → Auditus 管线。
- **音频下行**：Vox 的 PCM 输出 → 浏览器播放。
- **视频上行**：摄像头帧（MJPEG 或通过 MediaStream API 的原始帧）
  → Visus 管线。带元数据头的二进制帧（时间戳、分辨率、格式）。
  也支持本地 V4L2 捕获绕过 WS。
- **状态流**：JSON 帧——意识状态、Decode 分支、唤醒等级、活跃
  说话人、注意力热图。
- **控制**：启动/停止、配置变更、人格切换。

## HTTP 端点

- `GET  /health`     — 就绪探针
- `GET  /api/state`  — 意识状态快照
- `GET  /api/memory` — 查询长期记忆
- `POST /api/config` — 更新运行时配置
- `POST /api/input`  — 注入文本输入

## 可观察仪表盘

WebUI 可视化：
- 当前意识流内容
- 所有 Decode 分支状态（thinking / speech / action / daydream）
- 唤醒等级仪表
- 说话人识别面板
- GPU 利用率、内存、KV Cache 占用率、驱逐率
- KV 块重要性热图（热 vs 驱逐候选）
- 视觉：当前摄像头帧 + 检测到的特征
- 长期记忆：情节存储大小、语义图统计、最近检索
- 整合活动：梦中记忆维护进度

## 前端规则

语义 HTML、样式-功能解耦、组件模型、文件结构规则见
`.github/instructions/webui.instructions.md`。

## 实现面

```
src/nexus/
├── ws_server.{h,cpp}
├── http_server.{h,cpp}
└── webui/
    ├── index.html
    ├── css/
    ├── js/
    └── assets/
```

## 哲学注记

- Nexus 是内部拉丁世界与外部世界相遇的*唯一*场所。网络侧的命名
  （WS 消息类型、HTTP 路由）允许务实——它们必须对外部集成者可理解。
  但这种翻译只发生在 `src/nexus/`。
