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

## 结果:6 / 12 计数器对比(测试集 241 窗口,含 26 正例)

合成数据 8241 行 / 1200 执行流。良性进程分**三类真实负载**作为干扰项:普通(~65%)、
合法加密(~20%,TLS/磁盘加密/备份,crypto+ase 高但不全盘遍历)、合法压缩(~15%,
zip/媒体编码,SIMD 高但 crypto 低)。勒索软件 = 加密 + 全盘流式遍历(LLC 未命中飙升)。
唯一变量是可用计数器数。**Hybrid 测试集:**

| 档位(信号数) | 衍生特征 | ROC-AUC | PR-AUC | F1 | 精确率 | 召回率 |
|:---:|:---:|---:|---:|---:|---:|---:|
| **6**(7 信号) | 13 | 0.996 | 0.972 | 0.894 | 1.000 | 0.808 |
| **12**(13 信号) | 23 | 1.000 | 1.000 | 0.981 | 0.963 | 1.000 |

- **6 档**(cpu_cycles + 6 通用):精确率 1.00(零误报)但召回 0.81(有漏报)——缺
  `ll_cache_miss`/`l1d_cache`,丢失了最强的“全盘遍历”信号,部分勒索漏检,适合初筛。
- **12 档**(cpu_cycles + 12 通用):全指标接近满分;缓存差分(`d_llc_mpki`/
  `d_l1d_miss_rate`,窗口级 Cohen's d 高达 2.0~2.5)+ 加密差分(`d_crypto_ratio`)
  两个正交维度,分别挡住“合法加密”与“合法压缩”,是推荐配置。

> **特征校验(窗口级可分性)**:最强判别特征是**时间差分**类——`d_l1d_miss_rate`
> (d≈2.5)、`d_llc_mpki`(d≈2.0)、`d_crypto_ratio`(d≈1.5)。这印证核心设计:勒索
> 指纹不是“缓存未命中高”,而是“进入加密期时缓存未命中**突然飙升**”。

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
