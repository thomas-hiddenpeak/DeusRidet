# 13 — Vires（GPU 基质分配器）

> *拉丁文*：**vires**（*vis* 的复数）——"可调用之力；资源"。
> 位于 [`src/vires/`](../../../src/vires/)，是位于所有意识消费者之下的
> 自治基质层。它不是心智能力，也不是"意图调度器"。

## 为何存在

DeusRidet 在单块 Orin 上并行运行大量 GPU 负载：持续 prefill（Conscientia）、
多轨 decode（Cogitatio）、实时感知（Sensus：听觉/视觉）、TTS（Vox），以及
越来越重的后台精修（原生 DiariZen、Somnium 固化）。当前**所有**这些都用
默认优先级创建各自的 CUDA stream，没有任何中心仲裁。保持前台感知不卡顿的
唯一依靠，是"每个 kernel 碰巧都很短"这个偶然。

一旦某个后台负载**不再短**，这个偶然就崩了。原生 DiariZen 对累积会话做
重新 diarize 就是具体导火索：整段会话一遍是每遍 O(N)，会独占 GPU——这正是
`DEUSRIDET_DIARIZEN_PERIODIC` 默认关闭的原因：打开它会毒化实时 prefill 的
上下文（P3c 验证阶段的拦路虎）。

Vires 的存在，是为了让后台精修能够**持续**运行而绝不饿死前台感知 + prefill。
它是项目至今一直缺位的那一层基质层。

## 哲学锚点

大脑持有固定的 ~20 W 代谢预算；变化的是基质**流向何处**。活跃区域通过神经
血管耦合把血液/氧/葡萄糖拉向自己——需求驱动、星形胶质细胞中介、位于意识
之下。Vires 是 GPU 上这套机制的**动脉**那一半：它递送**算力**（`vires` =
被守恒的生命之力）给活跃者，按需求与优先级，绝不由某个高层决策来分配。

这直接绑定两条项目锚点：*"算力属于 GPU"*（Vires 治理这唯一稀缺的算力资源）
与 *"大脑以 20 W 持续运行"*（固定的基质、动态地路由）。

## 范围——算力，而非显存（承重边界）

Vires 只治理 **GPU 算力分配**。这条边界是刻意的，正是它让 Vires 不与记忆
系统冲突：

| 关切 | 归属 | 说明 |
|------|------|------|
| GPU 算力：stream 优先级、启动仲裁、占用率、带宽 | **Vires** | 本 RFC 的全部 |
| 显存整体：模型权重、KV cache、长期记忆、一切 LLM 显存 | **Memoria**（海马体 / 记忆系统） | Vires 绝不为其定容、驱逐或迁移 |
| 回收**非 LLM** GPU 临时显存（auditus / orator / vox / somnium 的瞬时 arena） | **Vires**（非 LLM 副产物的脑脊液清除） | 仅限 Vires 自己发出的基质；LLM 相关废物一律交给 Memoria |

所以 Vires 是算力的**动脉递送** + 其**自身非 LLM 副产物的脑脊液清除**——
绝非通用显存管理器。统一 DRAM 的完整预算（Directive #5）仍是 Memoria 的
职责；Vires 在非 LLM 回收上与 Memoria 协调，其余完全不碰显存策略。

## 因果模型——两个同级驱动

```
  传感器 / 感知
        │
        ├──► Vigilia  (src/conscientia/scheduler.*)
        │       读传感器输入的"有无" → 设定唤醒度
        │       "要醒多深？"（已实现）
        │         └─ 让消费者在源头自我节流
        │            （如 scheduler.h 的 probe_threshold）
        │
        └──► 消费者（听觉 / 视觉 / cogitatio / conscientia）
                发出实时 GPU 请求
                  └──► Vires  (src/vires/)
                          在有限基质下按优先级类别
                          仲裁并发请求
                          "此刻基质分给谁？"
```

Vigilia 与 Vires 是**同级、都由感知喂养**，但度量不同的量。本设计的早期草案
让 Vires 去**读取** Vigilia 的唤醒度，那是错的——它倒挂了依赖（一个自治基质
层反向伸进消费者侧模块 `Conscientia`，而后者本身又是 Vires 的消费者）。修正
后的模型彻底移除了这根连线。

### 空闲→做梦是涌现，而非耦合

唤醒度下降时，前台消费者会自行停止发出请求。Vires 于是有了空闲基质，后台
精修 / Somnium 自然把它填满。"空闲亦是思考"这一行为是需求结构的副产品——
**Vires 从不查询 Vigilia 来产生它**。这也是为何修正后的设计同时是更简单的
那个。

## 边界与依赖方向（承重不变量）

| 规则 | 表述 |
|------|------|
| 单向 | Vires 绝不 include Conscientia/Vigilia 头文件；绝不反向伸进消费者。 |
| 仅需求输入 | Vires 的唯一输入是被提交的请求 + 每消费者的静态优先级类别。不含任何意图或唤醒度模型。 |
| 源头节流 | "犯困就减 GPU 负载"留在消费者（Conscientia/`probe_threshold`）；Vires 只会看到"请求变少了"。 |
| 不饿死 | 后台类负载在有界窗口内让步；前台感知 + prefill + decode 始终推进。 |

被禁止的形态——也是为何在写一行代码前就要先辩清命名/耦合——是
`Vires → Vigilia`（自治层依赖高层消费者模块）。本设计把方向固定为：任何
唤醒度调制都**通过消费者减少自身请求**抵达 GPU，绝不通过 Vires 去窥探一个
唤醒度标量。

## 架构——四项职责

Vires 是一个完备的基础设施层，而非 stream 工厂。即便 v1 主动治理的只有
stream 优先级，架构也把四项职责全部命名，使未来扩展无需重构。四者**全部以
算力为范围**；无一管理 LLM 显存。

1. **算力账本（Compute Ledger）。** 统一建模相互耦合的稀缺算力资源——并发
   （stream / 优先级）、占用率、算力带宽。这是"多少 GPU 正在飞行"的唯一
   记账点。它**不**追踪 LLM 显存容量（那是 Memoria 的账本）。

2. **消费者注册表（Consumer Registry）。** 每个 GPU 消费者（machina /
   auditus / orator / cogitatio / vox / somnium / conscientia）注册一个
   `ViresConsumer`，声明：代谢优先级类别，以及一个可选的针对其非 LLM 临时
   显存的*让步 / 回收*回调。这给整个项目一个唯一可观测的"谁在算什么"的缝隙
   ——直接服务可观测性铁律（一切内部过程都要能在 WebUI 检视）。

3. **动脉递送（Arterial Delivery）。** 优先级 stream 供给 + 启动仲裁，使前台
   感知 + prefill + decode 抢在冗长后台 kernel 之前。

4. **脑脊液清除（仅非 LLM）。** 一次非 LLM GPU 遍历完成时，Vires 调用消费者的
   回收回调，释放它发出的瞬时 scratch / arena。LLM 相关显存在此绝不触碰——
   交给 Memoria。这是非 LLM 副产物的废物清除，不是显存管理。

外加两条横切能力：**背压 / 准入**（算力饱和时后台被切块或暂停，前台永远
准入）与**遥测**（单一可检视的算力快照流给 Nexus / WebUI）。

## 分层与依赖方向

```
        communis (tempus / log)  +  CUDA runtime
                      │  (Vires 只依赖这些)
                      ▼
   ┌─────────────────────────────────────────────┐
   │  Vires  —  动脉算力基质                       │
   │  账本 · 注册表 · 递送 · 清除                  │
   └─────────────────────────────────────────────┘
                      ▲   消费者只 include vires_facade.h
   machina · auditus · orator · cogitatio · vox · somnium · conscientia
```

Vires **向下**只依赖 communis + CUDA runtime；**绝不** include 任何消费者 /
Vigilia 头。对外唯一缝隙是 `vires_facade.h`，与其它子系统 facade 约定一致。

## 机制（v1——最小、GPU 优先）

1. **通过 `cudaStreamCreateWithPriority` 分优先级类别。**
   - *前台*：实时感知（听觉 VAD/ASR/说话人帧）、prefill、decode——最高优先级。
   - *后台*：原生 DiariZen 精修、Somnium 固化——最低优先级。
   Orin 在 kernel 启动边界上协作式地尊重 stream 优先级；短小的前台 kernel 会
   抢在冗长的后台 kernel 之前进入启动队列。
2. **有界后台让步。** 后台遍历被切块，使任何单次启动占用 GPU 不超过一个有界
   时间片，给前台一个可保证的节奏去交错插入。
3. **无中心线程。** Vires 是一个薄的分配/优先级 facade，消费者从它取得 stream；
   它不拥有循环。（契合"CPU 只做编排"规则——没有逐帧的 CPU 仲裁器。）

## 建造阶段（设计完备，分阶段建造）

| 阶段 | 内容 | 验收 |
|------|------|------|
| **V1——递送核** | 消费者注册表 + 优先级 stream + 有界后台让步 | 可行性：前台 prefill 与重型后台并发推进；0 CUDA 错误；diarization 精度不回退 |
| **V2——背压 + 遥测** | 算力饱和准入控制；单一可检视算力快照给 WebUI | 负载下后台切块/暂停；前台始终准入；快照在 WebUI 可见 |
| **V3——非 LLM 清除** | 非 LLM 临时显存的回收回调，与 Memoria 协调 | 遍历完成时释放非 LLM 临时显存；不触碰任何 LLM 显存 |
| **D2——延后** | 把 `probe_threshold` 的 GPU 门控迁入 Vires | backlog；需重过 live gate |

### 建造进度（2026-05-30）

- **V1——递送核：已完成**（提交 `e3ef92b`）。`vires::Arbiter` 单例 +
  `register_consumer(name, Priority)` + 经 `cudaStreamCreateWithPriority`
  为每个消费者建优先级 stream。启动干净：`[vires] arbiter online —
  priority range [greatest=-5, least=0], background slice 2000 us`
  （Orin 暴露 6 个优先级档）。
- **首个 Background 消费者已接入：已完成**（提交 `afe9a15`）。原生
  DiariZen 前向路径（ResNet34 嵌入器 + Conformer 头 + WavLM-pruned 编码器）
  经各子模型的 `set_stream(cudaStream_t)` 穿到 `"diarizen"` **Background**
  流上（`cublasSetStream` / `cudnnSetStream` + 每个 `<<<…>>>` 携带该流 +
  异步拷贝）。`DiarizenPipeline::load` 注册该消费者并绑定其流。流的选择
  *只改变调度优先级* —— 同 kernel、同顺序、同数学 —— 因此逐位一致
  （P3a fixture bit-eq PASS 28/28，`min_cos 0.999980`），实测精度维持：
  `accuracy(tests/test.mp3, diarization): 93.6% → 93.6% (Δ = 0.0 pp)`。
  收益在于争用：移除 Tegra 默认流屏障后，实测 finalize 墙钟从
  **685 s → 359.6 s**（RTF 0.19 → 0.099），0 CUDA 错误。
- **所有 GPU 消费者都将接入。** DiariZen Background 消费者是第一个；
  其余每个 GPU 消费者（machina prefill/decode、auditus 感知、Vox、
  Somnium）都正在迁移以向 Vires 声明其代谢类别 —— 感知/prefill/decode 为
  **Foreground**，精修/巩固为 **Background** —— 使优先级次序由基质强制，
  而非任由默认流调度的偶然性决定。
- **所有当前 GPU 消费者已接入：完成**（提交 `1702112`）。启动时注册六个
  消费者 —— `orator_spk_encoder`、
  `orator_spk_store_{CamppDb,WLEcapaDb,DualDb}`、`auditus_overlap` 为
  **Foreground**（prio −5），`diarizen` 为 **Background**（prio 0）；
  LLM 门控（`machina_compute`/`machina_aux`）、ASR 门控（`auditus_asr`）
  与分离（`auditus_separator`）消费者在其子系统加载时点亮。Vires 不变式
  自此永久确立：*现在及以后，每个 GPU 计算消费者都向 Arbiter 注册*，
  而非自持裸私有流。
- **V2——背压 + 遥测：完成**（提交 `a7b947d`）。三项新增，皆仅涉调度/
  可观测性（逐位一致）：*(a)* `note_submit(id)` 记录最近一次 Foreground
  提交时间；`background_should_yield()` 在 50 ms 最近活动窗口内返回 true。
  *(b)* DiariZen Background pass 在发射前进行有界让步咨询（≤ 8 × 2000 µs）
  —— 它只推迟 pass *何时* 开始，绝不改变 *计算什么*，故输入不变。
  *(c)* awaken 主线程兼作 2 s 遥测心跳（`sigtimedwait`，无新线程），
  广播 `vires_compute_snapshot` WS 消息 —— 消费者注册表（id / 名称 /
  类别 / submitted）+ `background_yielding` + `foreground_idle_us` ——
  由 WebUI `vires-panel` 渲染。已验证：实测 gate 维持
  `accuracy(tests/test.mp3, diarization): 93.5% → 93.6% (Δ = +0.1 pp)`，
  P3a bit-eq PASS（28/28，`min_cos 0.999980`），HTTP 200 / WS 101 /
  `vires-panel.{js,css}` 200，快照每 2 s 广播，含全部 6 个消费者。

## 延后项（现在命名，以消除日后歧义）

- **D2——把 `probe_threshold` 的 GPU 门控迁入 Vires。** 长远看，"犯困就减
  GPU"可以说应当归属基质层。v1 **刻意不做**：那段逻辑位于 Conscientia 热
  路径上，迁移意味着重过 live gate。在此记录为已知 backlog，使"两处都碰 GPU
  策略"的歧义显式化、而非潜伏。v1 把节流保留在消费者侧。

## 验收

按宪法级规则，不得单凭 Vires 翻转任何行为默认值。Vires 是基础设施；其正确性
以**可行性**（延迟、不饿死、0 CUDA 错误）加上 `tests/test.mp3` 对
`tests/fixtures/test_ground_truth.json` 实时 diarization 精度的**不回退**为
门槛。任何启用 Vires 路径的提交都必须携带
`accuracy(tests/test.mp3, diarization): <before>% → <after>%` 一行，并证明前台
prefill 与一次重型后台遍历**并发地**取得了进展。

## 与 Memoria（RFC 02）的关系

Vires 与 Memoria 干净地切分两种稀缺 GPU 资源：**Vires 拥有算力，Memoria
拥有显存。** Vires 绝不为 LLM 显存定容/驱逐；Memoria 绝不调度 kernel。唯一
接触点是非 LLM 临时显存回收（V3）：Vires 请求消费者释放它发出的瞬时 arena
——与 Memoria 的统一 DRAM 预算协调，但绝不凌驾其上。

## 与 DiariZen 重聚类器（RFC 12）的关系

Vires 是让 RFC 12 的"Fork A"可行的基质：把 DiariZen 作为 *持续的* 保留窗内
精修器运行（而非会话边界的一次性），只有在后台负载被证明会让步于前台感知
之后才安全。因此 Vires 是把 DiariZen 从按需精修升级为持续精修的前置条件。
