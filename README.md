# 勒索软件检测:ARM64 PMU 硬件事件 + XGBoost + LSTM 混合模型

基于 **ARM64(ARMv8-A)CPU 性能监控单元(PMU)硬件事件**的勒索软件检测器。
特征**仅来自 CPU 硬件计数器**——不依赖文件系统、句柄、进程名等操作系统软件信息,
因此难以被规避、且跨软件栈一致。把每个执行流(pid)的 PMU 事件时序切成滑动窗口,
用 **LSTM** 学时序模式、**XGBoost** 学窗口截面统计,再用 **Logistic 元学习器**做
**stacking 融合**。

> 数据为 `src/data_gen.py` 生成的**合成 PMU 时序**(按勒索软件的硬件指纹建模);
> 可替换为真实采集数据(`perf stat -e ... -I` 等),列名对齐即可。

## 为什么用 PMU 事件

勒索软件的本质是**大批量加密文件**,这在 CPU 硬件层留下三类指纹:

| 行为 | 硬件后果 | PMU 事件 |
|------|---------|----------|
| 跑 AES/ChaCha 加密 | 加密扩展 / SIMD 指令暴增 | `crypto_spec`(0x77)、`ase_spec`(0x74) |
| 流式遍历海量文件 | 各级缓存未命中升高 | `l1d_cache`、`l1d_cache_refill`、`l2d_cache_refill`、`ll_cache_miss` |
| 读明文 / 写密文 | load/store 构成偏移 | `ld_spec`(0x70)、`st_spec`(0x71) |
| (归一化基准) | IPC / 占比分母 / 分支 | `cpu_cycles` `inst_retired` `inst_spec` `br_retired` `br_mis_pred` |

共 **13 个信号 = `cpu_cycles`(专用周期计数器 PMCCNTR)+ 12 个通用事件**(`--counters 12`
全集)。`cpu_cycles` 走专用通道,不占 ARM 核心通常仅 6 个的通用计数器名额。
⚠️ `0x77` 等是 ARM 架构标准编号;Apple M 系列用私有 PMU 事件、编号语义不同且
`crypto_spec` 不一定暴露,真实采集前需用 `perf list` 等确认芯片实际支持的事件。

## 数据 schema

每行 = 某执行流(pid)在一个采样区间内的 **PMU 事件增量计数**(13 列):

```
timestamp, pid,
cpu_cycles,                                          # 专用周期计数器
inst_retired, inst_spec, l1d_cache, l1d_cache_refill,
l2d_cache_refill, ll_cache_miss, ld_spec, st_spec,
br_retired, br_mis_pred, crypto_spec, ase_spec,      # 12 个通用事件
ransomware            # 标签 0 良性 / 1 勒索
```

> 数据集**整体构成(良性 4 类 + 勒索)与训练/验证/测试切分**详见
> [docs/dataset.md](docs/dataset.md)。
> 三大平台(x86 Windows / ARM / Apple Silicon)勒索软件现状与 PMU 检测可迁移性
> 见 [docs/ransomware_platforms.md](docs/ransomware_platforms.md)。

## 方法

1. **特征工程**(`src/data.py`,**依赖感知**):每个衍生特征声明所需的原始计数器,
   只构建依赖全部可用的特征。全部为**与计数尺度无关的比值/归一化**:
   - 微架构效率:`ipc`、L1/LLC 的 `mpki`、缓存未命中率、分支误预测率、后端停顿率;
   - 指令构成占比:`crypto_ratio`、`ase_ratio`、`ld_ratio`、`st_ratio`、`st_ld_ratio`…(加密指纹核心);
   - 强度时间一阶差分(`d_crypto_ratio` 等):捕捉“进入加密期”的跃变。
2. **窗口化**:每个 pid 按时间切 `win=32 / stride=16`(短序列 padding + masking)。
   每窗产出 LSTM 时序张量 `(32, F)` 与 XGBoost 聚合特征 `mean/std/min/max/last × F`。
3. **切分**:按 pid 分组 + 标签分层,train/val/test ≈ 64/16/20%,**同一执行流不跨集合**(防泄漏)。
4. **模型**:XGBoost(`scale_pos_weight`,验证集 aucpr 早停)+ 双向 LSTM(掩码池化 +
   末步隐状态,验证集 AUC 早停)+ Logistic 元学习器堆叠 `[P_xgb, P_lstm]`。
5. **评估**:验证集选最大化 F1 的阈值,测试集报告 ROC-AUC / PR-AUC / F1 / 精确率 / 召回率。

## 计数器子集(6 / 12)

`cpu_cycles` 走专用周期计数器(PMCCNTR),不占通用名额;ARM 核心通常有 **6 个**通用
计数器,同时数更多需内核多路复用(有误差)。`data.COUNTER_SUBSETS` 预设两档,用
`--counters` 切换,模拟探针可并发采集的预算:

| 子集 | = cpu_cycles(专用)+ 通用事件 | 详细说明 |
|------|------|------|
| **6**(免多路复用,7 信号) | + `inst_retired` `inst_spec` `crypto_spec` `ase_spec` `l1d_cache_refill` `st_spec` | [docs/counters_6.md](docs/counters_6.md) |
| **12**(多路复用,13 信号,全集) | 6 档 + `l1d_cache` `l2d_cache_refill` `ll_cache_miss` `ld_spec` `br_retired` `br_mis_pred` | [docs/counters_12.md](docs/counters_12.md) |

> 每档的**完整计数器清单、构建出的特征、处理流程、表现**分别见上表链接的独立文档。

## 运行

```bash
pip install -r requirements.txt
# macOS 若 xgboost 报 "libomp.dylib could not be loaded":
#   brew install libomp   # 或把 torch 自带的 libomp.dylib 拷到 /opt/homebrew/opt/libomp/lib/

cd src
python3 data_gen.py                 # 生成合成 PMU 数据 -> data/processed/dataset.csv
python3 train.py                    # 训练 + 验证(默认 12 计数器全集)
python3 train.py --counters 6       # 只用 6 个计数器
python3 compare_counters.py         # 6/12 两档原型对比
```

## 结果:6 / 12 计数器对比(测试集 241 窗口,含 23 正例)

合成数据 7798 行 / 1200 执行流。良性进程分**四类真实负载**作为干扰项:
- 普通(~55%)、合法加密(~18%,TLS/磁盘加密,crypto 高但不全盘遍历)、
  合法压缩(~12%,zip/媒体,SIMD 高但 crypto 低);
- **加密备份 encbackup(~15%,最难)**:合法的加密备份 / 全盘加密初始化,**加密 +
  全盘遍历都高**,在两个核心维度与勒索重叠,**仅靠节奏区分**(平稳长流 vs 突发脉冲)。

勒索软件 = 加密 + 全盘遍历 + **突发节奏**(后者是与 encbackup 的关键差异)。**Hybrid 测试集:**

| 档位(信号数) | 衍生特征 | ROC-AUC | PR-AUC | F1 | 精确率 | 召回率 |
|:---:|:---:|---:|---:|---:|---:|---:|
| **6**(7 信号) | 13 | 1.000 | 0.998 | 0.955 | 1.000 | 0.913 |
| **12**(13 信号) | 23 | 0.999 | 0.993 | 0.936 | 0.917 | 0.957 |

- **6 档**:精确率 1.00、召回 0.91。因缺缓存计数器,**看不到 encbackup 的全盘遍历**,
  反而不会把它误判成勒索 → 零误报;代价是漏掉部分真勒索(召回 < 12 档)。
- **12 档**:召回 0.96(更全),但能看到缓存,故**会被 encbackup 迷惑产生 ~8% 误报**
  (精确率 0.92)。看得多 → 也更易被最难干扰项骗,这是直面真实难度后的真实代价。

> **关键发现**:加入 encbackup 后满分消失(更接近真实)。模型区分勒索与 encbackup 靠的
> 是**节奏特征**——勒索是突发脉冲(`d_` 差分大、窗口方差大),encbackup 是平稳长流
> (差分≈0)。这验证了差分/方差特征的真正价值:在绝对水平重叠时,**节奏**才是判据。
>
> ⚠️ 纯 PMU 对“加密备份”这类混合负载仍有残留误报;生产环境需补充白名单/签名/文件熵
> 等 PMU 之外的信号。

> ⚠️ 指标偏高源于**合成数据**,证明的是“特征+模型流水线随计数器增减合理涨跌”,
> **不是真实检测率**。6 个具体选哪几个为先验挑选;真实场景建议用真实数据做计数器
> 选择(按特征重要性/前向选择挑最优子集)。

## 产物

- `reports/metrics.json` — 默认 12 计数器的完整测试集指标
- `reports/counter_comparison.json` — 6/12 两档对比
- `reports/roc.png` / `pr.png` / `confusion_hybrid.png` / `xgb_importance.png`
- `models/xgb.json` / `lstm.pt` / `preproc.pkl`(标准化器、元学习器、阈值)

## 目录

```
src/data_gen.py          合成 PMU 事件数据生成
src/data.py              依赖感知特征工程 + 窗口化 + 计数器子集
src/model.py             LSTM 定义 + 训练/推理
src/train.py             端到端训练、验证、评估、保存(--counters 开关)
src/compare_counters.py  6/12 计数器子集原型对比
requirements.txt
```
