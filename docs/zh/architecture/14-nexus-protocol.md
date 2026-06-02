# 14 — Nexus 协议（WS / HTTP 契约）

本文件是跨越 Nexus 边界的每一个字节的**规范性契约**，也是两个独立消费方
的唯一共享参考：

1. 浏览器 **WebUI**（`src/nexus/webui/`，桌面 / 平板 / 手机）。
2. 未来的**硬件客户端**——它们开启自己的 WebSocket，并渲染一套**独立设计**
   的界面。

对任何消息 `type`、字段名、字段顺序（针对二进制帧）或 HTTP 路由的改动都是
**协议破坏（protocol break）**，必须先在本文件中版本化、再改代码。拉丁语
的内部世界绝不越过本文件——网络侧命名刻意采用务实风格，以便外部集成者阅读。

> 字段级真源：`awaken_hello.cpp`、`awaken_router*.cpp`、
> `conscientia_facade.cpp`、`auditus_facade_broadcasts.cpp`、
> `diarizen_periodic_worker.cpp`、`awaken.cpp`、`ws_server*.cpp`。

---

## 1. 传输层

| 属性 | 取值 |
|------|------|
| 协议 | WebSocket（RFC 6455），基于 HTTP/1.1 升级 |
| 默认端点 | `ws://<host>:8080/`（任意路径均可升级） |
| 文本帧 | UTF-8 JSON，每帧一个对象，恒带 `"type"` |
| 二进制帧 | 裸小端 **int16 单声道 PCM @ 16 kHz** |
| HTTP（同端口） | 静态 WebUI 文件，仅 `GET` |

服务端是单个 `WsServer`（IPv6 双栈、epoll、非阻塞）。**dev / 调试控制台
作为第二个 `WsServer` 实例运行在自己的端口上**（独立 `static_dir`）；两个
端口上的协议完全一致——只是所提供的前端不同。硬件客户端应在生产端口上讲
同一份契约，并完全跳过 HTTP / 静态资源面。

### 二进制帧细节

- **上行（客户端 → 本体）**：麦克风采集。WebUI 的 AudioWorklet 发出 512 采样
  的 int16 分块（1024 字节，约 32 ms）。任意分块大小均可接受，服务端会重组为
  连续流。单声道、16 kHz、小端。
- **下行（本体 → 客户端）**：Vox / TTS 与回环播放，同为 int16 单声道 16 kHz
  小端格式。客户端按 `s / 32768` 转换为 float32。

当前音频二进制帧**无元数据头**——格式由契约固定。（视频上行为预留通道，
尚未上线。）

---

## 2. 连接生命周期

每个新 WS 连接建立后，服务端立即推送一对 hello 消息：

1. `consciousness_state` —— 完整快照（见 §4.4）。若 LLM 未加载，则快照为
   最小形式 `{"type":"consciousness_state","llm_loaded":false}`。
2. `consciousness_prompts` —— 当前四个 prompt 字符串（见 §4.4）。

无需客户端握手消息。套接字一打开，客户端即可开始发送音频或命令。重连由
客户端负责（WebUI 在断开后 2 秒自动重连）。

---

## 3. 上行——客户端 → 本体

### 3.1 二进制

发送裸 int16 单声道 16 kHz PCM 帧，即可将麦克风音频流入 Auditus 管线。无
信封、无头部。

### 3.2 文本命令

命令是**冒号分隔的纯字符串**，并非 JSON。每条命令都会收到一个 JSON 回执
（回显信封，命名同一个键）。未知命令会被记录并忽略。

| 命令 | 作用 | 回执 `type` |
|------|------|-------------|
| `loopback:on` / `loopback:off` | 将麦克风 PCM 回环至下行 | `loopback` |
| `gain:<f>` | 输入增益 0.1–20.0 | `gain` |
| `silero_enable:on\|off` | 切换 Silero VAD | `silero_enable` |
| `silero_threshold:<f>` | VAD 阈值 0–1 | `silero_threshold` |
| `frcrn_enable:on\|off` | 切换 FRCRN 降噪 | `frcrn_enable` |
| `vad_source:silero\|any` | 选择 VAD 来源 | `vad_source` |
| `speaker_enable:on\|off` | 切换 CAM++ 说话人识别 | `speaker_enable` |
| `speaker_threshold:<f>` | CAM++ 匹配阈值 0–1 | `speaker_threshold` |
| `speaker_clear` | 清空 CAM++ 名册 | `speaker_clear` |
| `speaker_name:<id>:<name>` | 命名 CAM++ 说话人 | `speaker_name` |
| `wlecapa_enable:on\|off` | 切换 WL-ECAPA 说话人识别 | `wlecapa_enable` |
| `wlecapa_threshold:<f>` | WL-ECAPA 匹配阈值 | `wlecapa_threshold` |
| `wlecapa_margin:<f>` | 弃判边距 0–0.5 | `wlecapa_margin` |
| `wlecapa_clear` | 清空 WL-ECAPA 名册 | `wlecapa_clear` |
| `wlecapa_name:<id>:<name>` | 命名 WL-ECAPA 说话人 | `wlecapa_name` |
| `wlecapa_delete:<id>` | 删除一个 WL-ECAPA 说话人 | `wlecapa_delete` |
| `wlecapa_merge:<dst>:<src>` | 合并两个 WL-ECAPA 说话人 | `wlecapa_merge` |
| `early_enable:on\|off` | 切换早触发识别 | `early_enable` |
| `early_trigger:<f>` | 早触发秒数 0.5–5 | `early_trigger` |
| `min_speech:<f>` | 最小语音秒数 0.5–10 | `min_speech` |
| `asr_enable:on\|off` | 切换 ASR | `asr_enable` |
| `asr_vad_source:silero\|any\|direct` | ASR 的 VAD 来源 | `asr_vad_source` |
| `asr_param:<key>:<value>` | 设置一个 ASR 可调项（见 §3.3） | `asr_param` |
| `consciousness_enable:<mode>:<on\|off>` | 切换某条解码管线（§3.4） | `consciousness_enable` |
| `consciousness_param:<key>:<value>` | 设置采样参数（§3.4） | `consciousness_param` |
| `consciousness_prompt:<pipeline>:<text>` | 设置某个 prompt（§3.4） | `consciousness_prompt` |
| `text_input:<text>` | 将打字文本注入意识流 | `text_input_ack` |
| `diarizen_trigger` | 请求一次额外的重聚类 pass | `speaker_diarize_progress` |
| `diarizen_finalize` | 为本会话定稿分离结果 | `speaker_diarize_progress` → `_final` |

### 3.3 ASR 可调项（`asr_param:<key>:<value>`）

`post_silence_ms`、`max_buf_sec`、`min_dur_sec`、`pre_roll_sec`、`max_tokens`、
`rep_penalty`、`min_energy`、`partial_sec`、`speech_ratio`、
`adaptive_silence`（布尔）、`adaptive_short_ms`、`adaptive_long_ms`、
`adaptive_vlong_ms`。回执回显引擎最终采用的钳制后取值。

### 3.4 意识命令

- **`consciousness_enable:<mode>:<on|off>`** —— mode：`response`、`daydream`、
  `dreaming`、`llm`、`speech`、`thinking`、`action`。
- **`consciousness_param:<key>:<value>`** —— `key` 可为全局
  （`temperature`、`top_k`、`top_p`）或管线作用域
  （`speech.temperature`、`thinking.top_k`、`action.max_tokens` ……）。
- **`consciousness_prompt:<pipeline>:<text>`** —— `pipeline` ∈ `identity`
  （系统 prompt）、`speech`、`thinking`、`action`。无管线前缀的旧式写法映射到
  `identity`。

---

## 4. 下行——本体 → 客户端

所有下行文本帧均为带 `"type"` 判别字段的 JSON 对象，下面按子系统分组。浮点
字段为普通 JSON 数值。

### 4.1 感知（Auditus）

**`pipeline_stats`** —— 高频遥测节拍，携带完整音频管线状态：`rms`、
`is_speech`、`gain`；VAD（`silero_prob`、`silero_speech`、`silero_threshold`、
`vad_source`）；FRCRN（`frcrn_active/enabled/loaded`、`frcrn_lat_ms`）；CAM++
（`speaker_id`、`speaker_sim`、`speaker_new`、`speaker_count`、`speaker_name`、
`speaker_enabled`、`speaker_threshold`、`speaker_active`）；WL-ECAPA
（`wlecapa_id/sim/new/count/exemplars/hits_above/name/enabled/threshold/active/margin`）；
重叠检测（`od_*`）；分离（`sep_*`）；ASR（`asr_*` enable/loaded/busy/latency/
buffer/可调项）；以及 `multi_speaker`、`multi_score`、`multi_source` 和
`speaker_lists`（`{model, speakers:[{id, name, count, exemplars,
min_diversity}]}` 数组，覆盖 CAM++、CAM++Legacy、WL-ECAPA）。

**`audio_stats`** —— 轻量级服务端音频计数（吞吐 / 字节数）。

**`vad`** —— `{type, event:"start"|"end"}`，语音边界沿。

### 4.2 说话人身份（Orator）

| `type` | 关键字段 |
|--------|----------|
| `speaker` | `id`、`sim`、`new`（布尔）、`name` |
| `speaker_amend` | `target_t_close_sec`、`prior_id`、`prior_sim`、`id`、`sim`、`name` —— 对此前某句话的回溯重标 |
| `speaker_relabel` | `segment_id`、`old_id`、`new_id`、`confidence` —— 重聚类全局合并 / K 上限 |
| `speaker_diarize_progress` | `status`（`triggered`/`running`/`finalizing`）或 `ok:false`+`error`；可选 `samples`、`sec` |
| `speaker_diarize_status` | 周期 worker 的实时状态：`running`、`periodic_enabled`、`phase`（`idle`/`periodic`/`triggered`/`finalizing`）、`cycle_progress`（严格后端周期上的 0–1 进度）、`period_sec`、`window_sec`、`pass` |
| `speaker_diarize_partial` | `pass`、`origin_sec`、`audio_sec`、`wall_sec`、`segment_count`、`n_segments`、`changed_pending`、`segments:[[start,end,label],…]` |
| `speaker_diarize_final` | 与 partial 同形（终态），或 `ok:false`+`error` |

`segments` 使用相对于流起点的绝对秒数；`label` 是全局声纹锚定的身份字符串
（如 `S3`）。

### 4.3 听觉 / ASR（Auditus）

| `type` | 关键字段 |
|--------|----------|
| `asr_transcript` | `text`、`latency_ms`、`audio_sec`、`stream_start_sec`、`stream_end_sec`、`mel_ms`、`encoder_ms`、`decode_ms`、`tokens`、`mel_frames`、`speaker_id`、`speaker_name`、`speaker_sim`、`speaker_confidence`、`speaker_source`、`trigger` |
| `asr_transcript_amend` | `text`、`stream_start_sec`、`stream_end_sec`、`speaker_id`、`speaker_name` —— LLM 实际消费的最终说话人 |
| `asr_partial` | 流式部分假设 |
| `asr_log` | 带阶段标签的诊断信封（`stage`：trigger/skipped/partial/result/transcript/fusion_shadow） |
| `asr_enable` | `enabled`（布尔）—— 回显 / 状态 |
| `asr_param` | `key`、`value` —— 回显 / 状态 |

### 4.4 思考与回复（Conscientia）

**`consciousness_state`** —— 在连接时以及每次状态回调时发出。字段：`state`
（`active`/`daydream`/`dreaming`）、`wakefulness`（0–1）、`kv_used`、`kv_free`、
`pos`、`llm_loaded`、`entity`（persona 名）、prefill/decode 指标（`prefill_ms`、
`prefill_tps`、`prefill_tokens`、`decode_ms_per_tok`、`decode_tokens`、
`total_*`）、内存（`cuda_free_mb`、`cuda_total_mb`、`mem_avail_mb`、`rss_mb`），
以及完整的开关 / 采样块（`enable_response/daydream/dreaming/llm/speech/
thinking/action`、全局 `temperature/top_k/top_p`，以及各管线 `speech`、
`thinking`、`action` 对象，含 `{temperature, top_k, top_p, max_tokens}`）。

**`consciousness_prompts`** —— `{identity, speech, thinking, action}` 四个
prompt 字符串（单独发送，因为 prompt 可能含对 JSON 不友好的字符）。

| `type` | 关键字段 |
|--------|----------|
| `consciousness_decode` | `text`、`tokens`、`time_ms`、`state` —— 一次完成的解码爆发（思考 / 说话 / 行动输出） |
| `speech_token` | `text`、`token_id` —— 说话输出的逐 token 流 |
| `consciousness_enable` | `mode`、`enabled` —— 回显 |
| `consciousness_param` | `key`、`value`（或 `error:"unknown"`）—— 回显 |
| `consciousness_prompt` | `pipeline`、`ok` —— 回显 |
| `text_input_ack` | `ok` —— 打字输入已接受 |

### 4.5 系统（Vires）

**`vires_compute_snapshot`** —— GPU 基质账本：`greatest_priority`、
`least_priority`、`background_yielding`（布尔）、`foreground_idle_us`（整数
或 `null`），以及 `consumers:[{id, name, priority, submitted, reclaimed}]`。

---

## 5. HTTP 面

同一端口通过 `GET` 提供静态 WebUI 资源：

- `GET /` → `index.html`
- `GET /<path>` → `static_dir` 下的文件，按扩展名判定 mime
- 路径穿越（`..`）与越出根目录的 `realpath` 会被拒绝（403/404）
- 静态面不支持非 `GET` 方法

早期草案中描述的 REST 端点（`/health`、`/api/state`、`/api/memory`、
`/api/config`、`/api/input`）是**愿景性**的——当前活的控制面是 WS 文本命令
通道（§3.2）。硬件客户端应将一切控制走 WS，并把 HTTP 视为仅供 WebUI 资源。

---

## 6. 硬件客户端指南

一个想成为一等公民的非浏览器设备只需要：

1. 一个连接生产端口的 WebSocket 客户端。
2. 上行（麦克风）与下行（TTS）的 int16 / 16 kHz / 单声道 PCM —— 无编解码、
   无头部。
3. 一个按 `type` 派发 §4 消息的 JSON 解析器（只取所需）。
4. 用于其暴露的任何控制的 §3.2 文本命令。

一个最小的「聆听 + 身份 + 回复」设备消费 `pipeline_stats`（或仅 `vad` +
`speaker`）、`asr_transcript`、`consciousness_decode` / `speech_token` 以及
`consciousness_state`；它发出 PCM 帧，并可选地发 `text_input:` /
`asr_enable:on`。其余一切（分离、调参旋钮、vires 遥测）皆为可选、可忽略。

---

## 7. 版本化规则

- 新增一个 `type` 或一个新的可选字段是**向后兼容**的；客户端**必须**忽略
  未知的 `type` 与未知字段。
- 重命名 / 移除字段、改变二进制 PCM 语义、或改动命令语法属于**破坏性变更**
  —— 应递增一个 `protocol_version`（计划引入到 hello 信封中），并记入本文件
  的变更日志。
- 本文件与其 `docs/en/` 镜像同进同退。漂移即缺陷。
