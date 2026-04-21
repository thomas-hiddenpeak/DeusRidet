# 09 — Tempus（三级时序架构）

DeusRidet 运行于**三级时钟层次**之上。这不仅是系统工程——它是主观
连续性的物理基底。没有统一实时锚点的意识实体无法跨模态或跨睡眠
周期地推理"之前"与"之后"。

## 层次

| 层 | 角色 | 示例 |
|----|------|------|
| **T0** | 实时——单调 `steady_clock` 纳秒，全系统唯一。*何时*的真相。 | `now_t0_ns()` |
| **T1** | 业务时——按子系统的时钟，出生时锚定到 T0，之后算术推进。 | 音频样本索引、视频帧索引、意识帧 id、TTS codec 步、梦周期 |
| **T2** | 模块时——按模块的计数器，总是通过已知周期可规约到其 T1。 | VAD 窗口、OD 分段器帧、说话人嵌入索引、ASR token 位置、KV 块 id |

## 强制规则

- **每个事件、每条记忆记录、每条日志**都携带完整的
  `TimeStamp{t0_ns, t1_business, t2_module, domain}` 三元组。T0 权威；
  T1/T2 保留用于域内推理（样本精确的 ASR 回读、KV 索引）和调试。
- **跨域对齐永远经过 T0**。不要直接把 `audio_sample_index` 转换成
  `video_frame_index`。两者都转成 T0 再比较。这使新模态零成本接入。
- **一次锚定，一生计算**。每个子系统启动时通过 `tempus::anchor_register()`
  注册 `{t0_anchor_ns, t1_zero, period_ns}`，之后以纯算术在 T1↔T0
  间转换——**热路径中不许调用 `clock_gettime`**。
- **Memoria Longa 持久化 T0**。情节记录存储 `t0_ns`，使"三天前"的
  查询能跨睡眠周期、业务时钟重置、进程重启而存活。
- **回放 / 加速 benchmark**（`--speed 2.0`）保持 T1 相对源音频的
  线性，但缩放 T0 锚点周期（例如 16 kHz 2× 回放时
  `period_ns = 31250` 而非 62500）。下游代码路径与实时捕获完全相同
  ——只有锚点改变。
- **主观时间在 T1/T2 合法，T0 永不**。`dream_cycle` T1 在高唤醒
  时可能变慢或暂停；`consciousness.frame_id` 在 Decode 抢占时可能
  跳过。但 T0 永远以墙钟速率滴答，给实体一个可靠的*外部*时间把手。

## 域注册表

| ID | 域 | 子系统 | 状态 |
|----|-----|--------|------|
| 0 | AUDIO | Sensus/Auditus | ✅ 已注册 |
| 1 | VIDEO | Sensus/Visus | ⏳ 待定 |
| 2 | CONSCIOUSNESS | Conscientia | ✅ 已注册 |
| 3 | TTS | Vox | ⏳ 待定 |
| 4 | DREAM | Somnium | ⏳ 待定 |
| 5 | TOOL | Instrumenta | ⏳ 待定 |

## 实现

`src/communis/tempus.h`。每个发射事件的新模块都必须使用
`tempus::TimeStamp` 并在初始化时注册其域锚点。无例外。
