# DeusRidet 开发日志

按日条目（最新在前）. 点击日期查看当日完整记录。

新增条目：在 `devlog/YYYY-MM-DD.md` 中创建文件，并在下方列表顶端追加链接。

## 开发日志

- [2026-04-30](devlog/2026-04-30.md) — Step 18n-18w：Auditus 分离源 identity fusion selector、runtime fusion shadow ASR/speaker evidence、arbitrium + ledger telemetry、deterministic enrollment replay，以及 Mel rolling compaction；r2 run1/run2 复现 4/4 个 birth、10 个 stable hit、0 wrong、0 authority violation
- [2026-04-29](devlog/2026-04-29.md) — Step 18l-18m：在隔离 py313_audio_labs 中完成上游 ClearerVoice + 3D-Speaker + FunASR smoke，并跑通 top30 Auditus 上游-vs-C++ 探针；生产 py313 CUDA torch 保持不变
- [2026-04-28](devlog/2026-04-28.md) — Step 18g-18k：Auditus 分离源 ASR + speaker-ID 对 GT 时间线验证；identity monitor 保持 raw_o16000 为生产主状态并完成失败归因
- [2026-04-27](devlog/2026-04-27.md) — Step 18：Auditus 同质性审计、ASR split shadow metadata、真实 clip MossFormer2 分离、FRCRN A/B，以及固定窗口 separator 适配
- [2026-04-26](devlog/2026-04-26.md) — Step 17b-A：RetroFullRing 实时修正（speaker_amend 帧；3 次 1x 回放，decided_macro 0.924 / 0.930 / 0.949；无说话人整建制坍缩）
- [2026-04-24](devlog/2026-04-24.md) — 第 11 步收官：函数级分解（A3 spectral_cluster、A2 SpeakerTracker::check、A1 AudioPipeline::process_loop 1574 → 353）；第 13 步：R1 残余清账（spectral_cluster_stages 636 → 3×≤235；audio_pipeline.h 511 → 489）—— R1 全线合规
- [2026-04-23](devlog/2026-04-23.md) — 第 10 步收官：Actus 宪章复位（5 个原子提交；去除 `cmd_` 前缀，bench/profile 迁往 tools/，引擎探针迁往 tests/integration/，`test-ws` → `awaken`）
- [2026-04-22](devlog/2026-04-22.md) — 第 8a 步：Conscientia 外观抽取（cmd_test_ws.cpp 458 → 392 行；JSON 工具上提到 Communis）
- [2026-04-21](devlog/2026-04-21.md) — 第 7 步收官：Auditus 外观 + Actus WS  路由（cmd_test_ws.cpp 1543 → 458 行）；第 9 步收官：CUDA/音频 R1 拆分大行动（20 个原子提交，12 个超限文件全部解决）；第 11 步开启：函数级分解
- [2026-04-20](devlog/2026-04-20.md) — S1–S4 循环：语义指标 + S3 绕过 FRCRN + Seg3 阈值扫描
- [2026-04-19](devlog/2026-04-19.md) — v24d/v24e 说话人识别：发现阶段 + 大规模参数搜索 ; GPTQ GEMM 优化轮次：4 个实验，1 个成功
- [2026-04-18](devlog/2026-04-18.md) — MossFormer2 原生 CUDA 重写：架构拆解与实现准备 ; 说话人识别准确率达成 90%+ ; Prefill 优化分析：接近硬件极限（88 ms, M=11）
- [2026-04-17](devlog/2026-04-17.md) — Decode 算子融合 + INT4 Marlin Attention：113→89 ms/tok（提速 21%）
- [2026-04-16](devlog/2026-04-16.md) — Marlin 优化：SMEM 精确分配 + 融合残差加 + Tile 配置调优
- [2026-04-15](devlog/2026-04-15.md) — 音频增强 P1 + P2：重叠检测与语音分离
- [2026-04-07](devlog/2026-04-07.md) — Phase 3.8（实验）：PTX MMA 替代 WMMA（已回退）
- [2026-04-06](devlog/2026-04-06.md) — Phase 3.6：子层剖析、内核融合、内联反量化 ; Phase 3.7：合并投影权重
- [2026-04-05](devlog/2026-04-05.md) — Phase 2.3：带宽优化 — SMEM、向量化、核融合 ; Phase 3.3：DeltaNet 融合内核 + 寄存器缓存状态 ; Phase 3.4：GPTQ + INT8 内核深度优化 ; Phase 3.5：SM87 结构性分析 & INT8 寄存器预取
- [2026-04-04](devlog/2026-04-04.md) — Phase 2.1: 解码速度优化 ; 阶段 2.2：CUDA Graph + 核优化 ; Phase 2.4：FP16 投影层 INT8 量化 ; Phase 2.5：GPTQ Scale 延迟乘法 ; 阶段 3.1：Tensor Core WMMA GEMM ; 阶段 3.2：SMEM Bank Conflict 消除
- [2026-04-03](devlog/2026-04-03.md) — 阶段 3.0：批量 Prefill
- [2026-04-02](devlog/2026-04-02.md) — Phase 1：GPTQ-Int4 内核 ; Phase 0 完成：基础设施 ; 开发计划 ; Phase 2.6：内核融合 + INT8 lm_head
- [2025-11-20](devlog/2025-11-20.md) — 重叠检测三阶段优化：无净收益
- [2025-07-26](devlog/2025-07-26.md) — 说话人识别优化：v14–v20b 实验
- [2025-07-02](devlog/2025-07-02.md) — FP16 GEMM 内核优化：达到 cuBLAS 同等性能

