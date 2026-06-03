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

## 四、原型表现(Hybrid 测试集,241 窗口含 22 正例)

| 指标 | 值 |
|------|---:|
| ROC-AUC | 0.989 |
| PR-AUC | 0.889 |
| F1 | 0.772 |
| 精确率 | 0.629 |
| 召回率 | **1.000** |

**解读**:召回 1.0(漏报为零),但精确率仅 0.63——误报偏高。虽然本档已含 `crypto`+
`ase` 两个加密指纹,但缺 LLC 未命中、分支可预测性等上下文,**仍难完全分开“勒索全盘
加密”与“备份/TLS 合法加密”**(合成数据中约 25% 良性进程也做合法加密)。

## 五、适用场景

- ✅ **告警初筛 / 第一道闸**:不漏报优先,误报交给二级确认或 12 档复核。
- ✅ 计数器名额极紧、要求零多路复用误差、最低采集开销的嵌入式/高频场景。
- ❌ 不适合直接当最终判定(误报率高)。需要精确率时用 12 档。

> ⚠️ 指标基于合成数据,证明的是流水线随计数器增减合理涨跌,**非真实检测率**。
