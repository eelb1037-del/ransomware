# 6 档 PMU 方案(`--counters 6`)

面向**硬件最严格约束**的方案。ARMv8-A 的 PMU 分两类寄存器:

- **PMCCNTR**:专用周期计数器,**只数 `cpu_cycles`**,独立通道,不占通用名额;
- **PMEVCNTR0–5**:通常 **6 个**通用可编程计数器,数任意可选事件。

本档 = **`cpu_cycles`(专用)+ 6 个通用事件 = 7 个并发信号**,恰好占满硬件预算,
**无需内核多路复用**(time-multiplexing),采集无外推误差、开销最低。定位为**高召回
初筛器**。

## 一、采集的 PMU 计数器(7 个信号)

每行 = 某执行流(pid)在一个采样区间内的事件增量计数。

| 计数器 | ARMv8 编号 | 通道 | 物理含义 | 作用 |
|--------|:---:|:---:|---------|------|
| `cpu_cycles` | 0x11 | **专用 PMCCNTR** | CPU 周期数 | IPC 分母、活动强度 |
| `inst_retired` | 0x08 | 通用 1 | 完成的指令数 | IPC 分子、MPKI 分母 |
| `inst_spec` | 0x1B | 通用 2 | 投机执行指令数 | 指令占比的分母 |
| `crypto_spec` | 0x77 | 通用 3 | **加密扩展指令**(AES 等) | ⭐ 加密指纹 |
| `ase_spec` | 0x74 | 通用 4 | **SIMD/向量指令** | ⭐ 加密常走向量化 |
| `l1d_cache_refill` | 0x03 | 通用 5 | L1 数据缓存重填(未命中) | 访存压力 |
| `st_spec` | 0x71 | 通用 6 | store 指令数 | 密文写回占比 |

设计取舍:周期计数器免费(专用),6 个通用名额分给 2 个 IPC/占比分母
(`inst_retired`、`inst_spec`)、2 个加密指纹(`crypto_spec`、`ase_spec`)、
1 个访存(`l1d_cache_refill`)、1 个写回(`st_spec`)。**没有名额给 LLC、分支、load**。

## 二、构建出的 13 个特征(代码实跑)

依赖感知:只构建“依赖计数器全部可用”的特征。7 信号下实际构建:

| 特征 | 公式 | 含义 |
|------|------|------|
| `ipc` | `inst_retired / cpu_cycles` | 每周期指令数 |
| `l1d_mpki` | `1000 × l1d_cache_refill / inst_retired` | 每千指令 L1 未命中 |
| `crypto_ratio` | `crypto_spec / inst_spec` | ⭐ 加密指令占比 |
| `ase_ratio` | `ase_spec / inst_spec` | ⭐ SIMD 占比 |
| `crypto_plus_ase` | `(crypto_spec + ase_spec) / inst_spec` | ⭐ 加密相关总占比 |
| `st_ratio` | `st_spec / inst_spec` | store 占比 |
| `spec_to_retired` | `inst_spec / inst_retired` | 投机浪费度 |
| `log_cycles` | `log1p(cpu_cycles)` | 活动强度(压缩) |
| `log_inst` | `log1p(inst_retired)` | 指令量(压缩) |
| `d_crypto_ratio` | `crypto_ratio` 相邻区间差分 | 加密占比**跃变** |
| `d_ase_ratio` | `ase_ratio` 差分 | SIMD 占比跃变 |
| `d_st_ratio` | `st_ratio` 差分 | 写回占比跃变 |
| `d_ipc` | `ipc` 差分 | 效率跃变 |

> ⚠️ 6 档**采不到** `l1d_cache`,故 `l1d_miss_rate`(=refill/cache)无法构建,只有
> `l1d_mpki`。LLC、分支、load 类特征也全部缺失(对应计数器未采集)。

## 三、处理流程(与 12 档一致)

1. **清洗**:`fillna(0)` → `clip(lower=0)` → 分母列 `+1` 平滑(防除零)。
2. **比值化**:转上面 9 个无量纲特征(原始计数受时长/频率影响,不能直接用)。
3. **差分**:对 `crypto_ratio / ase_ratio / st_ratio / ipc` 求同 pid 内一阶差分。
4. **滑窗**:`win=32 / stride=16`;短序列尾部 padding + masking。
   - LSTM 输入 `X_seq = (窗口数, 32, 13)`
   - XGBoost 输入 `X_tab = mean/std/min/max/last × 13 = 65` 维
5. **标准化**:只在训练集 fit(防泄漏)。
6. **切分**:按 pid 分组 + 标签分层 ≈ 64/16/20,同一执行流不跨集合。

## 四、原型表现(Hybrid 测试集,241 窗口含 23 正例)

良性分**四类**干扰项:普通 / 合法加密 / 合法压缩 / **加密备份 encbackup**(加密+全盘
遍历都高,最难)。

| 指标 | 值 |
|------|---:|
| ROC-AUC | 1.000 |
| PR-AUC | 0.998 |
| F1 | 0.955 |
| 精确率 | **1.000** |
| 召回率 | 0.913 |

**解读**:精确率 1.0(零误报),召回 0.91。一个反直觉但重要的现象:6 档**缺
`ll_cache_miss`/`l1d_cache`,根本看不到“全盘遍历”信号**,因此最难的 encbackup
(合法加密备份)在它眼里只是“crypto 略高的普通进程”,**反而不会被误判成勒索** →
零误报。代价是同样因为缺缓存信号,部分真勒索的“全盘扫描”特征不全而漏检(召回 < 12 档)。

## 五、适用场景

- ✅ **告警初筛 / 第一道闸**:零误报,命中即高置信;漏掉的交给 12 档全量复核。
- ✅ 计数器名额极紧、要求零多路复用误差、最低采集开销的嵌入式/高频场景。
- ❌ 召回有限(~0.91),不适合作唯一防线。需要不漏报时用 12 档。
- 💡 注意:6 档“零误报”部分源于它看不到全盘遍历、躲过了 encbackup 陷阱,并非判别力
  更强——这是“瞎而稳”,12 档是“明而易受扰”。

> ⚠️ 指标基于合成数据,证明的是流水线随计数器增减合理涨跌,**非真实检测率**。
