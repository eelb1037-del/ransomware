"""合成 ARM64 (ARMv8-A) PMU 硬件事件时序数据集(勒索软件检测)。

特征 **仅来自 CPU 性能监控单元(PMU)的架构事件计数器**——纯硬件信号,不依赖
文件系统/句柄/进程名等操作系统软件信息,因此难以被规避,且跨平台一致。

采集模型:对被监控的执行流(pid/线程)按固定间隔读取一组 PMU 计数器,每行是
**一个采样区间内的事件增量计数**(PMU 通常读后清零)。

勒索软件的硬件指纹(进入加密期后):
  - 加密扩展/SIMD 指令占比骤升(CRYPTO_SPEC / ASE_SPEC,AES 等);
  - 大范围流式遍历文件数据 -> 末级缓存未命中、内存访问/指令比升高、缓存局部性差;
  - store 占比升高(密文写回);分支可预测(紧密循环)-> 误预测率偏低。
良性进程:加密指令占比极低,工作负载多样,缓存命中相对更好。

输出: data/processed/dataset.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# 采集到的原始 PMU 计数器(每个采样区间的增量)。命名对应 ARMv8-A 架构事件:
#   cpu_cycles=0x11 inst_retired=0x08 inst_spec=0x1B l1d_cache=0x04
#   l1d_cache_refill=0x03 l2d_cache_refill=0x17 ll_cache_miss=0x37
#   mem_access=0x13 ld_spec=0x70 st_spec=0x71 br_retired=0x21
#   br_mis_pred=0x10 crypto_spec=0x77 ase_spec=0x74 dp_spec=0x73
#   stall_backend=0x24
PMU_COUNTERS = [
    "cpu_cycles", "inst_retired", "inst_spec", "l1d_cache", "l1d_cache_refill",
    "ll_cache_miss", "ld_spec", "st_spec", "br_retired", "br_mis_pred",
    "crypto_spec", "ase_spec",
]


def _gen_interval(rng, profile, ramp):
    """生成一个采样区间的 PMU 计数。profile 给出该执行流的基线行为画像,
    ramp∈[0,1] 表示勒索加密强度(良性恒为 0)。返回各计数器整数计数。"""
    # 该区间消耗的 CPU 周期(活动强度,含噪声)
    cycles = rng.lognormal(profile["cyc_mu"], 0.4)
    # IPC:加密为紧密循环,IPC 偏高且稳定
    ipc = np.clip(profile["ipc"] + ramp * 0.4 + rng.normal(0, 0.15), 0.2, 4.0)
    inst = cycles * ipc
    spec = inst * rng.uniform(1.02, 1.25)  # 投机执行略多于退休

    # 指令构成(占 inst_spec 的比例),随加密强度调整
    crypto = np.clip(profile["crypto"] + ramp * rng.uniform(0.08, 0.22)
                     + rng.normal(0, 0.04), 0, 0.6)
    ase = np.clip(profile["ase"] + ramp * rng.uniform(0.03, 0.12)
                  + rng.normal(0, 0.05), 0, 0.5)
    ld = np.clip(profile["ld"] + rng.normal(0, 0.05), 0.05, 0.6)
    st = np.clip(profile["st"] + ramp * rng.uniform(0.03, 0.10)
                 + rng.normal(0, 0.05), 0.02, 0.5)
    dp = np.clip(1.0 - crypto - ase - ld - st, 0.02, 1.0)

    crypto_spec = spec * crypto
    ase_spec = spec * ase
    ld_spec = spec * ld
    st_spec = spec * st
    dp_spec = spec * dp

    # 访存与缓存:加密流式遍历 -> 访存多、局部性差(未命中率升高)
    mem_per_inst = np.clip(profile["mem_pi"] + ramp * rng.uniform(0.05, 0.15)
                           + rng.normal(0, 0.06), 0.05, 0.9)
    mem_access = inst * mem_per_inst
    l1d_cache = mem_access * rng.uniform(0.9, 1.1)
    l1d_miss = np.clip(profile["l1miss"] + ramp * rng.uniform(0.05, 0.18)
                       + rng.normal(0, 0.01), 0.005, 0.6)
    l1d_cache_refill = l1d_cache * l1d_miss
    l2d_cache_refill = l1d_cache_refill * np.clip(
        profile["l2frac"] + ramp * 0.15 + rng.normal(0, 0.03), 0.05, 0.9)
    ll_cache_miss = l2d_cache_refill * np.clip(
        profile["llfrac"] + ramp * 0.2 + rng.normal(0, 0.03), 0.02, 0.9)

    # 分支:加密紧密循环可预测性高 -> 误预测率偏低
    br_retired = inst * np.clip(profile["br_pi"] + rng.normal(0, 0.02), 0.02, 0.4)
    br_mis = br_retired * np.clip(profile["br_mis"] - ramp * 0.02
                                  + rng.normal(0, 0.005), 0.002, 0.3)
    stall_backend = cycles * np.clip(profile["stall"] + ramp * 0.05
                                     + rng.normal(0, 0.04), 0.0, 0.9)

    # l2d_cache_refill / mem_access / dp_spec / stall_backend 仅作中间量,不输出
    vals = [cycles, inst, spec, l1d_cache, l1d_cache_refill,
            ll_cache_miss, ld_spec, st_spec, br_retired, br_mis,
            crypto_spec, ase_spec]
    return [max(0, int(v)) for v in vals]


def _benign_profile(rng):
    # 约 25% 良性进程是“类加密”负载(备份压缩/TLS/媒体编解码),会用到加密/SIMD
    # 指令并产生可观访存——使单一 crypto 信号不再是充分判据,任务更接近真实。
    cryptoish = rng.random() < 0.25
    return {
        "cyc_mu": rng.uniform(16, 19), "ipc": rng.uniform(0.8, 2.2),
        "crypto": rng.uniform(0.05, 0.18) if cryptoish else rng.uniform(0.0, 0.03),
        "ase": rng.uniform(0.08, 0.22) if cryptoish else rng.uniform(0.0, 0.10),
        "ld": rng.uniform(0.18, 0.32), "st": rng.uniform(0.06, 0.18),
        "mem_pi": rng.uniform(0.2, 0.5) if cryptoish else rng.uniform(0.15, 0.4),
        "l1miss": rng.uniform(0.02, 0.10) if cryptoish else rng.uniform(0.01, 0.06),
        "l2frac": rng.uniform(0.1, 0.45), "llfrac": rng.uniform(0.05, 0.35),
        "br_pi": rng.uniform(0.12, 0.22), "br_mis": rng.uniform(0.02, 0.08),
        "stall": rng.uniform(0.1, 0.4),
    }


def _ransom_profile(rng):
    # 加密前基线与良性相近(加密期才暴露),避免任务过于平凡
    return {
        "cyc_mu": rng.uniform(16, 19), "ipc": rng.uniform(1.0, 2.0),
        "crypto": rng.uniform(0.0, 0.03), "ase": rng.uniform(0.0, 0.08),
        "ld": rng.uniform(0.18, 0.30), "st": rng.uniform(0.06, 0.14),
        "mem_pi": rng.uniform(0.15, 0.35), "l1miss": rng.uniform(0.01, 0.06),
        "l2frac": rng.uniform(0.1, 0.4), "llfrac": rng.uniform(0.05, 0.3),
        "br_pi": rng.uniform(0.12, 0.20), "br_mis": rng.uniform(0.03, 0.08),
        "stall": rng.uniform(0.1, 0.35),
    }


def generate(n_proc=1200, ransom_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    pid = 1000
    t0 = 1.7235e9
    for _ in range(n_proc):
        is_ransom = rng.random() < ransom_frac
        T = int(np.clip(rng.lognormal(1.7, 0.7), 3, 400))
        dt = rng.uniform(0.05, 0.5)  # PMU 采样间隔(秒),高频
        prof = _ransom_profile(rng) if is_ransom else _benign_profile(rng)
        onset = rng.integers(max(1, T // 4), max(2, T - 1)) if (is_ransom and T > 2) else T
        ts = t0 + np.cumsum(np.full(T, dt)) + rng.uniform(0, 1e6)
        pid += 1
        for k in range(T):
            ramp = 0.0
            if is_ransom and k >= onset:
                ramp = float(min(1.0, 0.4 + 0.12 * (k - onset)))
            counts = _gen_interval(rng, prof, ramp)
            rows.append([round(float(ts[k]), 4), pid] + counts + [int(is_ransom)])
    cols = ["timestamp", "pid"] + PMU_COUNTERS + ["ransomware"]
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=1200)
    ap.add_argument("--ransom-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(os.path.dirname(here), "data", "processed", "dataset.csv")
    ap.add_argument("--out", default=out)
    args = ap.parse_args()

    df = generate(args.n_proc, args.ransom_frac, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    pid_lab = df.groupby("pid")["ransomware"].max()
    print(f"已生成 {len(df)} 行 / {df['pid'].nunique()} 执行流 -> {args.out}")
    print(f"  特征: {len(PMU_COUNTERS)} 个 ARM64 PMU 事件计数器")
    print(f"  执行流级: 勒索 {int((pid_lab==1).sum())} / 良性 {int((pid_lab==0).sum())}")
    print(f"  行级标签分布: {df['ransomware'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
