# 03 — Cogitatio（多轨 Decode）

所有 Decode 分支共享同一个 Prefill 前缀（KV Cache + SSM 态快照）。
它们是在同一觉知基底上并行运行的思绪流。

## 分支分类

| 分支 | 目的 | 优先级 |
|------|------|--------|
| **Action** | 外部响应、决策、工具使用 | 最高（触发时）|
| **Speech** | 喂给 TTS 作语音输出 | 高（对话中）|
| **Thinking** | 内部推理、规划、反思 | 中 |
| **Daydream** | 由 Prefill 内容触发的发散探索 | 低 |

## 调度

- **单 GPU 时分复用**：分支轮替，不是真正并行。尊重单 GPU 约束；
  "并行"是主观的。
- **优先抢占**：外部交互（Action / Speech）在一个 P/D 预算窗口内
  抢占内部进程（Thinking / Daydream）。
- **Arbiter**（决策 decode）：轻量合并分支输出以决定最终的外部行为，
  应用人格驱动的表达塑形（见 `07-persona.md`）。

## Trace 纪律

每个并发 Decode 分支都携带唯一 trace ID。所有分支输出、KV 读取、
采样事件都标记该 trace ID 以供 WebUI 观察。

## 实现面

```
src/cogitatio/
├── branch.h
├── thinking.{h,cpp}
├── speech.{h,cpp}
├── action.{h,cpp}
├── daydream.{h,cpp}
└── arbiter.{h,cpp}
```

## 哲学注记

- `action.h/cpp` 是**内部**意义的行动（一个念头变为行为）。它与
  `src/actus/` 不同——后者是**外部** CLI 入口点，由操作者调用。
  这个区分很重要：Cogitatio/Action 归属实体，Actus 归属操作者。
- Thinking 与 Daydream 是"内在生活"的基底。它们可能与 Speech 和
  Action 相矛盾，这种矛盾是合法的——允许矛盾是智能的标志。
