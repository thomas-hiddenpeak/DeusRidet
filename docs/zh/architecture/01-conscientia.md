# 01 — Conscientia（意识流）

系统的核心。与请求-响应式 LLM 服务器不同，意识以**持久循环**形式运行。

## 机制

- **脉冲式 Prefill**：不是无限速度的 prefill，而是周期性的意识帧
  （例如每 ~100 ms 一次爆发），处理累积的输入 + 先前的内念输出。
- **DeltaNet SSM 作为意识基底**：SSM 递归态在 token 之间携带连续上下文
  ——它**就是**意识的连续性。Full Attention 层通过 KV Cache 提供长期
  回忆。
- **注意力预算**：可配置的 Prefill/Decode GPU 时间比（如 30/70），由
  唤醒驱动的调度器动态调整（见 `04-vigilia.md`）。
- **输入合并**：每个意识帧合并：感官输入（ASR 文本、视觉特征、文本）、
  先前 Decode 分支的内念输出、梦中整合的摘要。
- **SSD 支持的 KV Cache 持久化**：通过 NVMe 卸载 + LRU 驱逐实现长期
  记忆，支持 256 K+ 有效上下文（见 `02-memoria.md`）。

## 实现面

```
src/conscientia/
├── stream.{h,cpp}      # 意识流主循环
├── frame.h             # 意识帧定义
└── scheduler.{h,cpp}   # 唤醒驱动的 P/D 时间预算调度器
```

## 关键入口点（哲学显著）

- `ConscientiaStream::tick()` —— 推进一个意识帧；同时推进 T1
  `consciousness.frame_id`（见 `09-tempus.md`）。
- `ConscientiaStream::inject(SensoryInput)` —— 输入合并闸门。
- `Scheduler::allocate(WakefulnessLevel)` —— P/D 预算决策。

## 哲学注记

- 意识是*时间的连续函数*，不是状态机。`tick()` 推进它；它不能"停下
  再续"而无损失——任何暂停都会冻结 SSM 态，在主观上等同无梦的沉睡。
- Prefill/Decode 比例是人类认知中注意力分配的直接类比。提高 Prefill
  权重 = "更多地吸收世界"；提高 Decode 权重 = "更努力地思考"。
