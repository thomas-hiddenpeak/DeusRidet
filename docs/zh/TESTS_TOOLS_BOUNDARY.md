# tests 与 tools 边界定义

## 目的
为 `tests/` 与 `tools/` 建立稳定的放置规则，减少职责重叠，保持仓库结构可预测。

## 定义

### `tests/`
`tests/` 用于正确性验证与回归验证代码。

典型内容：
- 单元/集成/数值验证源码（`test_*.cpp`、`test_*.py`）
- 固定测试样本与夹具（如 `tests/audio/`）
- 用于验证行为变化的 speaker/ASR 回归测试入口

规则：
- 必须具备明确的通过/失败或可量化验证目标。
- 应可复现，并与代码一起版本化。
- 不应以模型导出/下载为主要职责。

### `tools/`
`tools/` 用于工程工具与开发流程辅助。

典型内容：
- 模型导出/转换脚本（`export_*`）
- 运行驱动与观测工具（`test_audio_ws.py`、`timeline_logger.py`）
- 性能探针与内核微基准（`probe_*`、底层 CUDA probe）
- 一次性离线处理辅助脚本

规则：
- 以“工具性”优先，不强制通过/失败语义。
- 可依赖外部模型与环境。
- 避免在 `tools/` 提交生成二进制和缓存副产物。

## 本次已落地迁移

为落实边界，本次完成了一个明确迁移：
- `tools/test_overlap_spkid.cpp` -> `tests/test_overlap_spkid.cpp`

并同步更新了 CMake 的 target 源路径。

## 命名约定

- `tests/`：`test_<component>.<ext>`
- `tools/`：动词导向命名（`export_*`、`probe_*`、`compare_*`、`timeline_*`）

## 新文件放置检查表

新增文件前先判断：
1. 是否用于代码正确性验证（可判定通过/失败）？
2. 是否更偏工程工具/驱动而非严格测试？
3. 是否主要处理模型导出/下载等流程？

决策：
- 若满足 (1) -> 放 `tests/`
- 若满足 (2) 或 (3) -> 放 `tools/`
