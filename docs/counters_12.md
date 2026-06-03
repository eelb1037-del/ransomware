# 12 档 PMU 方案(`--counters 12`,默认 / 推荐)

在 6 档基础上**补齐 SIMD、LLC、分支、load、L1 访问总量**,把判别力从“能抓但误报多”
提升到“能区分恶意加密与合法加密”。需要内核多路复用(同时数 >6 个计数器,有少量外推
误差),但换来显著更高的精确率。**本工程默认档,推荐配置。**

## 一、12 个原始 PMU 计数器

每行 = 某执行流(pid)在一个采样区间内的事件增量计数。

| # | 计数器 | ARMv8 编号 | 物理含义 | 作用 | 6 档有? |
|---|--------|:---:|---------|------|:---:|
| 1 | `cpu_cycles` | 0x11 | CPU 周期数 | IPC 分母 | ✓ |
| 2 | `inst_retired` | 0x08 | 完成的指令数 | IPC 分子 / MPKI 分母 | ✓ |
| 3 | `inst_spec` | 0x1B | 投机执行指令数 | 占比分母 | ✓ |
| 4 | `l1d_cache` | 0x04 | L1 数据缓存访问总数 | 算 L1 未命中**率** | ✗ 新增 |
| 5 | `l1d_cache_refill` | 0x03 | L1 缓存重填(未命中) | 访存压力 | ✓ |
| 6 | `ll_cache_miss` | 0x37 | 末级缓存未命中 | **全盘流式遍历强信号** | ✗ 新增 |
| 7 | `ld_spec` | 0x70 | load 指令数 | 读明文占比 / 读写比 | ✗ 新增 |
| 8 | `st_spec` | 0x71 | store 指令数 | 密文写回占比 | ✓ |
| 9 | `br_retired` | 0x21 | 退休分支数 | 分支密度 | ✗ 新增 |
| 10 | `br_mis_pred` | 0x10 | 分支误预测数 | 加密紧循环可预测性高→误预测低 | ✗ 新增 |
| 11 | `crypto_spec` | 0x77 | **加密扩展指令** | ⭐ 加密指纹核心 | ✓ |
| 12 | `ase_spec` | 0x74 | **SIMD/向量指令** | ⭐ 加密常走向量化 | ✗ 新增 |

相对 6 档新增 6 个:`l1d_cache`、`ll_cache_miss`、`ld_spec`、`br_retired`、
`br_mis_pred`、`ase_spec`。

## 二、构建出的 21 个特征(代码实跑)

**微架构效率(6)**:`ipc`、`l1d_miss_rate`(=refill/cache)、`l1d_mpki`、`llc_mpki`、
`br_mispredict_rate`、`br_per_inst`

**指令构成 / 加密指纹(7)**:`crypto_ratio`、`ase_ratio`、`crypto_plus_ase`、
`ld_ratio`、`st_ratio`、`st_ld_ratio`(=store/load 写读比)、`spec_to_retired`

**活动强度(2)**:`log_cycles`、`log_inst`

**时间一阶差分(6)**:`d_crypto_ratio`、`d_ase_ratio`、`d_st_ratio`、
`d_l1d_miss_rate`、`d_llc_mpki`、`d_ipc`

合计 6+7+2+6 = **21 维**(对比 6 档的 10 维)。

关键公式:
| 特征 | 公式 | 6 档为何没有 |
|------|------|------|
| `l1d_miss_rate` | `l1d_cache_refill / l1d_cache` | 6 档无 `l1d_cache` |
| `llc_mpki` | `1000 × ll_cache_miss / inst_retired` | 6 档无 `ll_cache_miss` |
| `ase_ratio` | `ase_spec / inst_spec` | 6 档无 `ase_spec` |
| `st_ld_ratio` | `st_spec / ld_spec` | 6 档无 `ld_spec` |
| `br_mispredict_rate` | `br_mis_pred / br_retired` | 6 档无分支计数器 |
| `crypto_plus_ase` | `(crypto_spec + ase_spec) / inst_spec` | 6 档无 `ase_spec` |

## 三、处理流程(与 6 档一致)

1. **清洗**:`fillna(0)` → `clip(lower=0)` → 分母列 `+1` 平滑。
2. **比值化**:转上面 15 个无量纲特征。
3. **差分**:对 6 个关键特征求同 pid 内一阶差分,抓“进入加密期”跃变。
4. **滑窗**:`win=32 / stride=16`;短序列尾部 padding + masking。
   - LSTM 输入 `X_seq = (窗口数, 32, 21)`
   - XGBoost 输入 `X_tab = mean/std/min/max/last × 21 = 105` 维
5. **标准化**:只在训练集 fit(防泄漏)。
6. **切分**:按 pid 分组 + 标签分层 ≈ 64/16/20,同一执行流不跨集合。

## 四、原型表现(Hybrid 测试集,241 窗口含 22 正例)

| 指标 | 12 档 | (对比 6 档) |
|------|---:|---:|
| ROC-AUC | 0.995 | 0.987 |
| PR-AUC | **0.973** | 0.865 |
| F1 | 0.840 | 0.800 |
| 精确率 | **0.750** | 0.667 |
| 召回率 | 0.955 | 1.000 |

**解读**:PR-AUC 从 0.865 跳到 0.973,精确率 0.667→0.750。新增的 `ase_spec`(SIMD)、
`ll_cache_miss`(LLC)、`br_mis_pred`(分支)提供了区分**“恶意全盘加密”与“正常合法
加密”**的上下文——全盘加密的缓存遍历模式和分支可预测性与备份/TLS 不同。

## 五、适用场景

- ✅ **推荐的默认检测配置**:精确率与 PR-AUC 全面优于 6 档。
- ✅ 平台支持多路复用、可接受少量计数外推误差时。
- ⚠️ 同时采集 >6 个计数器需内核多路复用,采样会有轻微误差;若名额极紧用 6 档初筛。

> ⚠️ 指标基于合成数据,证明的是流水线随计数器增减合理涨跌,**非真实检测率**。
> 真实场景建议用真实数据做计数器选择(按特征重要性/前向选择挑最优子集)。
