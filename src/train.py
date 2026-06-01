"""端到端训练与验证:XGBoost + LSTM 的 Stacking 混合勒索软件检测。

  1. 加载数据 -> 行为特征 -> 滑动窗口
  2. 按 pid 分组 + 标签分层 切分 train/val/test(防进程泄漏)
  3. 特征标准化(仅用训练集拟合)
  4. 训练 XGBoost(窗口聚合特征) 与 LSTM(时序序列)
  5. 验证集预测训练 Logistic 元学习器(stacking 融合)
  6. 验证集选最优阈值(最大化 F1),测试集评估 XGB / LSTM / Hybrid
  7. 保存 metrics.json、ROC/PR/混淆矩阵/特征重要性图、模型文件
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

import data as data_mod
import model as model_mod


def group_stratified_split(groups, y, test_size, val_size, seed):
    from sklearn.model_selection import train_test_split
    uniq = np.array(sorted(set(groups.tolist())))
    glab = np.array([y[groups == g].max() for g in uniq])
    g_tv, g_te = train_test_split(uniq, test_size=test_size,
                                  random_state=seed, stratify=glab)
    lab_tv = np.array([y[groups == g].max() for g in g_tv])
    g_tr, g_va = train_test_split(g_tv, test_size=val_size / (1 - test_size),
                                  random_state=seed, stratify=lab_tv)
    s_tr, s_va, s_te = set(g_tr), set(g_va), set(g_te)
    idx_tr = np.array([i for i, g in enumerate(groups) if g in s_tr])
    idx_va = np.array([i for i, g in enumerate(groups) if g in s_va])
    idx_te = np.array([i for i, g in enumerate(groups) if g in s_te])
    return idx_tr, idx_va, idx_te


def fit_seq_scaler(X_seq, lengths):
    T = X_seq.shape[1]
    valid = X_seq[np.arange(T)[None, :] < lengths[:, None]]
    mean, std = valid.mean(0), valid.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_seq_scaler(X_seq, lengths, mean, std):
    Xs = (X_seq - mean) / std
    T = X_seq.shape[1]
    mask = (np.arange(T)[None, :] < lengths[:, None])[..., None]
    return (Xs * mask).astype(np.float32)


def evaluate(y, prob, thr):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    pred = (prob >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "threshold": float(thr),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def best_f1_threshold(y, prob):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, prob)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1[:-1])])


def make_plots(y, probs, hyb_cm, xgb, tab_names, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    plt.figure(figsize=(6, 5))
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y,p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC — Test")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc.png"), dpi=120); plt.close()

    plt.figure(figsize=(6, 5))
    for n, p in probs.items():
        pr, rc, _ = precision_recall_curve(y, p)
        plt.plot(rc, pr, label=f"{n} (AP={average_precision_score(y,p):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR — Test")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr.png"), dpi=120); plt.close()

    cm = np.array(hyb_cm)
    plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.xticks([0, 1], ["benign", "ransom"]); plt.yticks([0, 1], ["benign", "ransom"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion — Hybrid"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_hybrid.png"), dpi=120); plt.close()

    imp = xgb.feature_importances_
    order = np.argsort(imp)[::-1][:20][::-1]
    plt.figure(figsize=(7, 6)); plt.barh(range(len(order)), imp[order])
    plt.yticks(range(len(order)), [tab_names[i] for i in order], fontsize=7)
    plt.xlabel("gain"); plt.title("XGBoost Top-20 Features"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "xgb_importance.png"), dpi=120); plt.close()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        root, "data", "processed", "dataset.csv"))
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.join(root, "reports"))
    ap.add_argument("--modeldir", default=os.path.join(root, "models"))
    ap.add_argument("--counters", default="16",
                    help="PMU 计数器子集: 6 / 12 / 16(见 data.COUNTER_SUBSETS)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    print("加载与特征工程 ...")
    available = data_mod.COUNTER_SUBSETS.get(args.counters)
    if available is None:
        available = [c.strip() for c in args.counters.split(",") if c.strip()]
    df = data_mod.load_dataframe(args.data)
    df, seq = data_mod.engineer_features(df, available=available)
    print(f"  可用计数器={len(available)} {available}")
    print(f"  行 {len(df)} | 序列特征数 F={len(seq)}")
    pk = data_mod.build_windows(df, seq, args.win, args.stride)
    X_seq, lengths = pk["X_seq"], pk["lengths"]
    X_tab, y, groups, tab_names = pk["X_tab"], pk["y"], pk["groups"], pk["tab_names"]
    print(f"  窗口 N={len(y)} | 正例={int(y.sum())} ({y.mean()*100:.1f}%) "
          f"| 表格维度 D={X_tab.shape[1]}")

    idx_tr, idx_va, idx_te = group_stratified_split(
        groups, y, args.test_size, args.val_size, args.seed)
    print(f"  切分 train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")
    for nm, ix in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
        print(f"    {nm} 正例率={y[ix].mean()*100:.1f}%")

    from sklearn.preprocessing import StandardScaler
    tab_scaler = StandardScaler().fit(X_tab[idx_tr])
    Xt_tr, Xt_va, Xt_te = (tab_scaler.transform(X_tab[i])
                           for i in (idx_tr, idx_va, idx_te))
    s_mean, s_std = fit_seq_scaler(X_seq[idx_tr], lengths[idx_tr])
    Xs_tr = apply_seq_scaler(X_seq[idx_tr], lengths[idx_tr], s_mean, s_std)
    Xs_va = apply_seq_scaler(X_seq[idx_va], lengths[idx_va], s_mean, s_std)
    Xs_te = apply_seq_scaler(X_seq[idx_te], lengths[idx_te], s_mean, s_std)
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"  pos_weight={pos_weight:.2f}")

    print("训练 XGBoost ...")
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
        scale_pos_weight=pos_weight, objective="binary:logistic",
        eval_metric="aucpr", early_stopping_rounds=40, n_jobs=0,
        random_state=args.seed, tree_method="hist")
    xgb.fit(Xt_tr, y_tr, eval_set=[(Xt_va, y_va)], verbose=False)
    p_xgb_va = xgb.predict_proba(Xt_va)[:, 1]
    p_xgb_te = xgb.predict_proba(Xt_te)[:, 1]
    print(f"  best_iteration={xgb.best_iteration}")

    print("训练 LSTM ...")
    lstm = model_mod.train_lstm(
        Xs_tr, lengths[idx_tr], y_tr, Xs_va, lengths[idx_va], y_va,
        n_features=len(seq), pos_weight=pos_weight, epochs=args.epochs,
        device=device, seed=args.seed)
    p_lstm_va = model_mod.predict_lstm(lstm, Xs_va, lengths[idx_va], device)
    p_lstm_te = model_mod.predict_lstm(lstm, Xs_te, lengths[idx_te], device)

    print("训练元学习器(Logistic stacking)...")
    from sklearn.linear_model import LogisticRegression
    Z_va = np.column_stack([p_xgb_va, p_lstm_va])
    Z_te = np.column_stack([p_xgb_te, p_lstm_te])
    meta = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Z_va, y_va)
    p_hyb_va = meta.predict_proba(Z_va)[:, 1]
    p_hyb_te = meta.predict_proba(Z_te)[:, 1]
    print(f"  meta 权重 [xgb,lstm]={meta.coef_[0].round(3).tolist()}")

    thr = {"XGBoost": best_f1_threshold(y_va, p_xgb_va),
           "LSTM": best_f1_threshold(y_va, p_lstm_va),
           "Hybrid": best_f1_threshold(y_va, p_hyb_va)}
    probs_te = {"XGBoost": p_xgb_te, "LSTM": p_lstm_te, "Hybrid": p_hyb_te}
    results = {n: evaluate(y_te, probs_te[n], thr[n]) for n in probs_te}

    print("\n================ 测试集结果 ================")
    print(f"{'model':<9}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>8}{'Prec':>8}{'Recall':>8}")
    for n, r in results.items():
        print(f"{n:<9}{r['roc_auc']:>9.4f}{r['pr_auc']:>9.4f}{r['f1']:>8.4f}"
              f"{r['precision']:>8.4f}{r['recall']:>8.4f}")
    print("===========================================\n")

    make_plots(y_te, probs_te, results["Hybrid"]["confusion_matrix"],
               xgb, tab_names, args.outdir)
    summary = {
        "dataset": os.path.basename(args.data), "n_windows": int(len(y)),
        "window": {"win": args.win, "stride": args.stride},
        "split": {"train": int(len(idx_tr)), "val": int(len(idx_va)),
                  "test": int(len(idx_te))},
        "pos_rate": {k: float(y[v].mean()) for k, v in
                     [("train", idx_tr), ("val", idx_va), ("test", idx_te)]},
        "pos_weight": pos_weight, "meta_coef": meta.coef_[0].tolist(),
        "results_test": results}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    xgb.save_model(os.path.join(args.modeldir, "xgb.json"))
    torch.save(lstm.state_dict(), os.path.join(args.modeldir, "lstm.pt"))
    with open(os.path.join(args.modeldir, "preproc.pkl"), "wb") as f:
        pickle.dump({"tab_scaler": tab_scaler, "seq_mean": s_mean,
                     "seq_std": s_std, "meta": meta, "thresholds": thr,
                     "seq_features": seq, "tab_names": tab_names,
                     "win": args.win, "stride": args.stride}, f)
    print(f"已保存: {args.outdir}/metrics.json + 图表; {args.modeldir}/模型")


if __name__ == "__main__":
    main()
