# 12 档 PMU 方案(`--counters 12`,默认 / 推荐)

= **`cpu_cycles`(专用周期计数器)+ 12 个通用事件 = 13 个并发信号**(全集)。在 6 档
基础上补齐 **L1 访问总量、L2/LLC 缓存层级、load、分支**,把判别力从“能抓但误报多”
提升到“能区分恶意加密与合法加密”。

> ARMv8-A 通用计数器(PMEVCNTR)通常只有 6 个,本档要并发数 12 个通用事件,**需内核
> 多路复用**(分时轮采再外推),会引入少量采样误差,换来显著更高的精确率。
> `cpu_cycles` 仍走专用 PMCCNTR,不占通用名额。

## 一、采集的 PMU 计数器(13 个信号)

每行 = 某执行流(pid)在一个采样区间内的事件增量计数。

| 计数器 | ARMv8 编号 | 通道 | 物理含义 | 6 档有? |
|--------|:---:|:---:|---------|:---:|
| `cpu_cycles` | 0x11 | **专用 PMCCNTR** | CPU 周期数 | ✓ |
| `inst_retired` | 0x08 | 通用 | 完成的指令数 | ✓ |
| `inst_spec` | 0x1B | 通用 | 投机执行指令数 | ✓ |
| `l1d_cache` | 0x04 | 通用 | L1 数据缓存访问总数 | ✗ 新增 |
| `l1d_cache_refill` | 0x03 | 通用 | L1 缓存重填(未命中) | ✓ |
| `l2d_cache_refill` | 0x17 | 通用 | L2 缓存重填 | ✗ 新增 |
| `ll_cache_miss` | 0x37 | 通用 | 末级缓存未命中 | ✗ 新增 |
| `ld_spec` | 0x70 | 通用 | load 指令数 | ✗ 新增 |
| `st_spec` | 0x71 | 通用 | store 指令数 | ✓ |
| `br_retired` | 0x21 | 通用 | 退休分支数 | ✗ 新增 |
| `br_mis_pred` | 0x10 | 通用 | 分支误预测数 | ✗ 新增 |
| `crypto_spec` | 0x77 | 通用 | **加密扩展指令** | ✓ |
| `ase_spec` | 0x74 | 通用 | **SIMD/向量指令** | ✓ |

相对 6 档新增 6 个通用事件:`l1d_cache`、`l2d_cache_refill`、`ll_cache_miss`、
`ld_spec`、`br_retired`、`br_mis_pred`(补全缓存层级 + 分支 + load)。

## 二、构建出的 23 个特征(代码实跑)

**微架构效率(8)**:`ipc`、`l1d_miss_rate`、`l2d_refill_per_l1miss`、`llc_miss_per_l2`、
`l1d_mpki`、`llc_mpki`、`br_mispredict_rate`、`br_per_inst`

**指令构成 / 加密指纹(7)**:`crypto_ratio`、`ase_ratio`、`crypto_plus_ase`、
`ld_ratio`、`st_ratio`、`st_ld_ratio`(写读比)、`spec_to_retired`

**活动强度(2)**:`log_cycles`、`log_inst`

**时间一阶差分(6)**:`d_crypto_ratio`、`d_ase_ratio`、`d_st_ratio`、
`d_l1d_miss_rate`、`d_llc_mpki`、`d_ipc`

合计 8+7+2+6 = **23 维**(对比 6 档 13 维)。

相对 6 档新增的关键特征及其依赖:
| 特征 | 公式 | 6 档为何没有 |
|------|------|------|
| `l1d_miss_rate` | `l1d_cache_refill / l1d_cache` | 6 档无 `l1d_cache` |
| `l2d_refill_per_l1miss` | `l2d_cache_refill / l1d_cache_refill` | 6 档无 `l2d_cache_refill` |
| `llc_miss_per_l2` | `ll_cache_miss / l2d_cache_refill` | 6 档无 L2/LLC |
| `llc_mpki` | `1000 × ll_cache_miss / inst_retired` | 6 档无 `ll_cache_miss` |
| `br_mispredict_rate` | `br_mis_pred / br_retired` | 6 档无分支计数器 |
| `ld_ratio` / `st_ld_ratio` | 含 `ld_spec` | 6 档无 `ld_spec` |

## 三、处理流程(与 6 档一致)

1. **清洗**:`fillna(0)` → `clip(lower=0)` → 分母列 `+1` 平滑。
2. **比值化**:转上面 17 个无量纲特征。
3. **差分**:对 6 个关键特征求同 pid 内一阶差分,抓“进入加密期”跃变。
4. **滑窗**:`win=32 / stride=16`;短序列尾部 padding + masking。
   - LSTM 输入 `X_seq = (窗口数, 32, 23)`
   - XGBoost 输入 `X_tab = mean/std/min/max/last × 23 = 115` 维
5. **标准化**:只在训练集 fit(防泄漏)。
6. **切分**:按 pid 分组 + 标签分层 ≈ 64/16/20,同一执行流不跨集合。

## 四、原型表现(Hybrid 测试集,241 窗口含 22 正例)

| 指标 | 12 档 | (对比 6 档) |
|------|---:|---:|
| ROC-AUC | **0.9996** | 0.989 |
| PR-AUC | **0.996** | 0.889 |
| F1 | **0.933** | 0.772 |
| 精确率 | **0.913** | 0.629 |
| 召回率 | 0.955 | 1.000 |

**解读**:PR-AUC 从 0.889 跳到 0.996,精确率 0.629→0.913(误报大幅下降)。新增的
`ll_cache_miss`(LLC)、`br_mis_pred`(分支)、`l2d_cache_refill`(L2)、`ld_spec`(load)
提供了区分**“恶意全盘加密”与“正常合法加密”**的上下文——全盘加密的缓存遍历模式、
分支可预测性、读写比与备份/TLS 不同。

## 五、适用场景

- ✅ **推荐的默认检测配置**:精确率与 PR-AUC 全面优于 6 档。
- ✅ 平台支持多路复用、可接受少量计数外推误差时。
- ⚠️ 同时采集 12 个通用事件需多路复用,采样有轻微误差;若名额极紧用 6 档初筛。

> ⚠️ 指标基于合成数据,证明的是流水线随计数器增减合理涨跌,**非真实检测率**。
> 真实场景建议用真实数据做计数器选择(按特征重要性/前向选择挑最优子集)。
