"""数据加载、特征工程与窗口化(ARM64 PMU 硬件事件)。

每行是某执行流(pid)在一个采样区间内的 PMU 事件增量计数。特征全部由这些硬件
计数器**比值/归一化**而来,不含任何操作系统软件信息。

**依赖感知**:每个衍生特征声明它需要哪些原始计数器(REQUIRES)。给定一组“可用
计数器”(available),只构建那些其依赖全部可用的特征。这样可在真实硬件只能并发
采集少量计数器(ARM 通常 6 个)的约束下,公平评估不同子集的检测能力。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PMU_COUNTERS = [
    "cpu_cycles", "inst_retired", "inst_spec", "l1d_cache", "l1d_cache_refill",
    "ll_cache_miss", "ld_spec", "st_spec", "br_retired", "br_mis_pred",
    "crypto_spec", "ase_spec",
]

# 预设子集:模拟硬件可并发采集的计数器数量
COUNTER_SUBSETS = {
    # 6 个:ARM 核心常见的通用计数器上限,无需内核多路复用
    #   cycles+inst -> IPC; inst_spec -> 占比分母; crypto -> 加密指纹;
    #   l1d_refill -> 访存压力(MPKI); st_spec -> 写回占比
    "6": ["cpu_cycles", "inst_retired", "inst_spec", "crypto_spec",
          "l1d_cache_refill", "st_spec"],
    # 12 个:多路复用可达,补齐 SIMD、load、分支、LLC、L1 总量(即全集)
    "12": list(PMU_COUNTERS),
}

# 每个衍生特征 -> (构建函数, 依赖的原始计数器)。构建函数入参为列字典 col。
# col[name] 已是 +1 平滑后的分母安全版本(见 engineer_features)。


def _feature_specs():
    """返回 [(name, requires:list[str], fn(col)->Series), ...]"""
    F = []
    def reg(name, req, fn):
        F.append((name, req, fn))

    # 微架构效率
    reg("ipc", ["inst_retired", "cpu_cycles"], lambda c: c["inst_retired"] / c["cpu_cycles"])
    reg("backend_stall_rate", ["stall_backend", "cpu_cycles"], lambda c: c["stall_backend"] / c["cpu_cycles"])
    reg("l1d_miss_rate", ["l1d_cache_refill", "l1d_cache"], lambda c: c["l1d_cache_refill"] / c["l1d_cache"])
    reg("l2d_refill_per_l1miss", ["l2d_cache_refill", "l1d_cache_refill"], lambda c: c["l2d_cache_refill"] / c["l1d_cache_refill"])
    reg("llc_miss_per_l2", ["ll_cache_miss", "l2d_cache_refill"], lambda c: c["ll_cache_miss"] / c["l2d_cache_refill"])
    reg("l1d_mpki", ["l1d_cache_refill", "inst_retired"], lambda c: 1000.0 * c["l1d_cache_refill"] / c["inst_retired"])
    reg("llc_mpki", ["ll_cache_miss", "inst_retired"], lambda c: 1000.0 * c["ll_cache_miss"] / c["inst_retired"])
    reg("mem_per_inst", ["mem_access", "inst_retired"], lambda c: c["mem_access"] / c["inst_retired"])
    reg("br_mispredict_rate", ["br_mis_pred", "br_retired"], lambda c: c["br_mis_pred"] / c["br_retired"])
    reg("br_per_inst", ["br_retired", "inst_retired"], lambda c: c["br_retired"] / c["inst_retired"])

    # 指令构成(占投机指令比例)—— 加密指纹核心
    reg("crypto_ratio", ["crypto_spec", "inst_spec"], lambda c: c["crypto_spec"] / c["inst_spec"])
    reg("ase_ratio", ["ase_spec", "inst_spec"], lambda c: c["ase_spec"] / c["inst_spec"])
    reg("ld_ratio", ["ld_spec", "inst_spec"], lambda c: c["ld_spec"] / c["inst_spec"])
    reg("st_ratio", ["st_spec", "inst_spec"], lambda c: c["st_spec"] / c["inst_spec"])
    reg("dp_ratio", ["dp_spec", "inst_spec"], lambda c: c["dp_spec"] / c["inst_spec"])
    reg("st_ld_ratio", ["st_spec", "ld_spec"], lambda c: c["st_spec"] / c["ld_spec"])
    reg("crypto_plus_ase", ["crypto_spec", "ase_spec", "inst_spec"], lambda c: (c["crypto_spec"] + c["ase_spec"]) / c["inst_spec"])
    reg("spec_to_retired", ["inst_spec", "inst_retired"], lambda c: c["inst_spec"] / c["inst_retired"])

    # 活动强度(log 压缩)
    reg("log_cycles", ["cpu_cycles"], lambda c: np.log1p(c["cpu_cycles"]))
    reg("log_inst", ["inst_retired"], lambda c: np.log1p(c["inst_retired"]))
    return F


# 对哪些特征追加时间一阶差分(若该特征存在)
_DIFF_TARGETS = ["crypto_ratio", "ase_ratio", "st_ratio", "l1d_miss_rate",
                 "llc_mpki", "ipc"]


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = df["timestamp"].astype(float)
    return df.sort_values(["pid", "timestamp"]).reset_index(drop=True)


def engineer_features(df: pd.DataFrame, available=None) -> tuple[pd.DataFrame, list[str]]:
    """available: 可用原始计数器集合(None=全部)。"""
    if available is None:
        available = set(PMU_COUNTERS)
    else:
        available = set(available)

    df = df.copy()
    # +1 平滑的分母安全列字典(只为可用计数器构建)
    col = {}
    for c in PMU_COUNTERS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)
        if c in available and c in df.columns:
            col[c] = df[c] + 1.0

    seq: list[str] = []
    built = set()
    for name, req, fn in _feature_specs():
        if all(r in col for r in req):
            df[name] = fn(col).astype(float)
            seq.append(name)
            built.add(name)

    # 时间一阶差分(同一 pid)
    g = df.groupby("pid", sort=False)
    for base in _DIFF_TARGETS:
        if base in built:
            dname = f"d_{base}"
            df[dname] = g[base].diff().fillna(0.0)
            seq.append(dname)

    df["pid_label"] = g["ransomware"].transform("max")
    return df, seq


def build_windows(df, seq_features, win=32, stride=16) -> dict:
    F = len(seq_features)
    Xs, Ls, Xt, ys, gs = [], [], [], [], []
    feat = df[seq_features].to_numpy(np.float32)
    labels = df["ransomware"].to_numpy()
    pids = df["pid"].to_numpy()

    for pid in pd.unique(pids):
        idx = np.where(pids == pid)[0]
        f = feat[idx]; lab = labels[idx]; L = len(idx)
        starts = [0] if L <= win else list(range(0, L - win + 1, stride))
        if L > win and starts[-1] != L - win:
            starts.append(L - win)
        for s in starts:
            e = min(s + win, L)
            w = f[s:e]; l = w.shape[0]
            wp = w if l == win else np.vstack([w, np.zeros((win - l, F), np.float32)])
            Xs.append(wp); Ls.append(l)
            Xt.append(_agg(w)); ys.append(int(lab[s:e].max())); gs.append(pid)

    return {
        "X_seq": np.stack(Xs).astype(np.float32),
        "lengths": np.asarray(Ls, np.int64),
        "X_tab": np.stack(Xt).astype(np.float32),
        "y": np.asarray(ys, np.int64),
        "groups": np.asarray(gs),
        "tab_names": _tab_names(seq_features),
        "seq_features": seq_features,
    }


def _agg(w):
    return np.concatenate([w.mean(0), w.std(0), w.min(0), w.max(0),
                           w[-1]]).astype(np.float32)


def _tab_names(seq_features):
    out = []
    for s in ["mean", "std", "min", "max", "last"]:
        out += [f"{s}_{f}" for f in seq_features]
    return out
