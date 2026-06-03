# 6 档 PMU 方案(`--counters 6`)

面向**硬件最严格约束**的方案:ARM 核心通常只有 6 个通用 PMU 计数器,本档恰好占满,
**无需内核多路复用**(time-multiplexing),采集无外推误差、开销最低。定位为**高召回
初筛器**。

## 一、6 个原始 PMU 计数器

每行 = 某执行流(pid)在一个采样区间内的事件增量计数。

| # | 计数器 | ARMv8 编号 | 物理含义 | 作用 |
|---|--------|:---:|---------|------|
| 1 | `cpu_cycles` | 0x11 | CPU 周期数 | IPC 分母、活动强度 |
| 2 | `inst_retired` | 0x08 | 完成的指令数 | IPC 分子、MPKI 分母 |
| 3 | `inst_spec` | 0x1B | 投机执行指令数 | 指令占比的分母 |
| 4 | `crypto_spec` | 0x77 | **加密扩展指令**(AES 等) | ⭐ 加密指纹核心 |
| 5 | `l1d_cache_refill` | 0x03 | L1 数据缓存重填(未命中) | 访存压力 |
| 6 | `st_spec` | 0x71 | store 指令数 | 密文写回占比 |

设计取舍:在 6 个名额里,2 个给 IPC(`cpu_cycles`+`inst_retired`)、1 个给占比分母
(`inst_spec`)、3 个给判别信号(`crypto_spec` 加密 / `l1d_cache_refill` 访存 /
`st_spec` 写回)。**没有名额给 SIMD、LLC、分支、load**,这是 6 档判别力受限的根因。

## 二、构建出的 10 个特征(代码实跑)

依赖感知:只构建“依赖计数器全部可用”的特征。6 档下实际构建:

| 特征 | 公式 | 含义 |
|------|------|------|
| `ipc` | `inst_retired / cpu_cycles` | 每周期指令数 |
| `l1d_mpki` | `1000 × l1d_cache_refill / inst_retired` | 每千指令 L1 未命中 |
| `crypto_ratio` | `crypto_spec / inst_spec` | ⭐ 加密指令占比 |
| `st_ratio` | `st_spec / inst_spec` | store 占比 |
| `spec_to_retired` | `inst_spec / inst_retired` | 投机浪费度 |
| `log_cycles` | `log1p(cpu_cycles)` | 活动强度(压缩) |
| `log_inst` | `log1p(inst_retired)` | 指令量(压缩) |
| `d_crypto_ratio` | `crypto_ratio` 的相邻区间差分 | 加密占比**跃变** |
| `d_st_ratio` | `st_ratio` 的差分 | 写回占比跃变 |
| `d_ipc` | `ipc` 的差分 | 效率跃变 |

> ⚠️ 注意:6 档**采不到** `l1d_cache`,所以 `l1d_miss_rate`(=refill/cache)无法构建,
> 只有 `l1d_mpki`(=refill/inst)。同理 `ase_ratio`、分支类、load 类特征全部缺失。

## 三、处理流程(与 12 档一致)

1. **清洗**:`fillna(0)` → `clip(lower=0)` → 分母列 `+1` 平滑(防除零)。
2. **比值化**:转上面 10 个无量纲特征(原始计数受时长/频率影响,不能直接用)。
3. **差分**:对 `crypto_ratio / st_ratio / ipc` 求同 pid 内一阶差分,抓“进入加密期”跃变。
4. **滑窗**:`win=32 / stride=16`;短序列尾部 padding + masking。
   - LSTM 输入 `X_seq = (窗口数, 32, 10)`
   - XGBoost 输入 `X_tab = mean/std/min/max/last × 10 = 50` 维
5. **标准化**:只在训练集 fit(防泄漏)。
6. **切分**:按 pid 分组 + 标签分层 ≈ 64/16/20,同一执行流不跨集合。

## 四、原型表现(Hybrid 测试集,241 窗口含 22 正例)

| 指标 | 值 |
|------|---:|
| ROC-AUC | 0.987 |
| PR-AUC | 0.865 |
| F1 | 0.800 |
| 精确率 | 0.667 |
| 召回率 | **1.000** |

**解读**:召回 1.0(漏报为零)很好,但精确率仅 0.67——每抓 3 个有 1 个误报。原因是
全档只有 `crypto_spec` 一个加密指纹,**分不开“勒索加密”与“备份/TLS 合法加密”**
(合成数据中约 25% 良性进程也做合法加密)。

## 五、适用场景

- ✅ **告警初筛 / 第一道闸**:不漏报优先,误报交给二级确认。
- ✅ 计数器名额极紧、要求零多路复用误差、最低采集开销的嵌入式/高频场景。
- ❌ 不适合直接当最终判定(误报率高)。需要精确率时用 12 档。

> ⚠️ 指标基于合成数据,证明的是流水线随计数器增减合理涨跌,**非真实检测率**。
