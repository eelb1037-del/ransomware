# 勒索软件检测:XGBoost + LSTM 混合模型(Stacking)

基于**进程行为时序快照**的勒索软件(ransomware)检测器。把每个进程(pid)随时间
变化的行为快照切成滑动窗口,用 **LSTM** 学习时序模式、**XGBoost** 学习窗口截面统计
特征,再用 **Logistic 元学习器**做 **stacking 融合**。

> 数据集为合成的“进程行为”数据(`src/data_gen.py`),模拟真实采集到的进程 IO/资源
> 时序;可直接替换为真实数据(列名对齐即可,见下)。

## 数据 schema

每行 = 某进程在某时刻的行为快照,关键列:

| 列 | 含义 |
|----|------|
| `timestamp` | 采样时间(秒) |
| `pid` | 进程号(序列分组键) |
| `processName` | 进程名(**训练时丢弃,防标签泄漏**) |
| `usage` | CPU 占用 % |
| `readCount/writeCount` | 读/写操作累积次数 |
| `readBytes/writeBytes` | 读/写累积字节数 |
| `otherCount/otherBytes` | 其它 IO 累积 |
| `handleCount/numThreads` | 句柄数 / 线程数 |
| `workingSetSize/pagefileUsage` | 工作集 / 页面文件占用 |
| `priorityBase/signed` | 基础优先级 / 是否数字签名 |
| `ransomware` | 标签(0 良性 / 1 勒索) |

替换真实数据时,保证上述列名一致即可复用全套特征工程。

## 方法

1. **特征工程**(`src/data.py`):重尾计数 `log1p`;按时间对 IO 累积计数器求**差分速率**
   (勒索软件的写突发信号);写读字节比、每次操作字节数等比值 → 每个时间步 23 维。
2. **窗口化**:每个 pid 按时间切 `win=32 / stride=16` 的窗口(短序列 padding + masking)。
   每窗产出:LSTM 时序张量 `(32, F)`,与 XGBoost 聚合特征 `mean/std/min/max/last × F = 115` 维。
3. **切分**:按 pid 分组 + 标签分层,train/val/test ≈ 64/16/20%,**同一进程不跨集合**(防泄漏)。
4. **模型**:
   - XGBoost:`scale_pos_weight` 处理不均衡,验证集 `aucpr` 早停。
   - 双向 LSTM:掩码均值池化 + 末步隐状态,`pos_weight` 加权,验证集 AUC 早停。
   - 元学习器:Logistic 回归堆叠 `[P_xgb, P_lstm]`。
5. **评估**:验证集选最大化 F1 的阈值,测试集报告 ROC-AUC / PR-AUC / F1 / 精确率 / 召回率。

## 运行

```bash
pip install -r requirements.txt
# macOS 若 xgboost 报 "libomp.dylib could not be loaded":
#   brew install libomp   # 或把 torch 自带的 libomp.dylib 拷到 /opt/homebrew/opt/libomp/lib/

cd src
python3 data_gen.py        # 生成合成数据 -> data/processed/dataset.csv
python3 train.py           # 训练 + 验证

cat ../reports/metrics.json
python3 train.py --epochs 60 --win 48 --stride 24   # 可选超参
```

## 测试集结果(默认配置,按 pid 分组切分,无标签泄漏)

数据:7213 行 / 1200 进程 → 1207 个窗口(正例 9.7%);test=241 窗口。

| 模型 | ROC-AUC | PR-AUC | F1 | 精确率 | 召回率 |
|------|--------:|-------:|---:|------:|------:|
| XGBoost | 0.9996 | 0.9965 | 0.978 | 1.000 | 0.957 |
| LSTM | 1.000 | 1.000 | 0.978 | 1.000 | 0.957 |
| **Hybrid** | **1.000** | **1.000** | **0.978** | **1.000** | **0.957** |

混合模型混淆矩阵 `[[TN,FP],[FN,TP]] = [[218,0],[1,22]]`(0 误报、1 漏报)。
元学习器权重 `[xgb=3.59, lstm=4.34]`(两路都被采用,LSTM 略占主导)。

> 说明:指标接近满分是因为**合成数据的写突发信号清晰可分**——目的是验证整条
> 训练/评估流水线正确跑通。换成真实采集数据后分数会自然下降,届时混合模型相对
> 单模型的增益会更明显。直接替换 `data/processed/dataset.csv` 重跑即可。

## 产物

- `reports/metrics.json` — 三个模型(XGBoost / LSTM / Hybrid)的完整测试集指标
- `reports/roc.png` / `pr.png` / `confusion_hybrid.png` / `xgb_importance.png`
- `models/xgb.json` / `lstm.pt` / `preproc.pkl`(标准化器、元学习器、阈值)

## 目录

```
src/data_gen.py  合成数据生成
src/data.py      特征工程 + 窗口化
src/model.py     LSTM 定义 + 训练/推理
src/train.py     端到端训练、验证、评估、保存
requirements.txt
```

