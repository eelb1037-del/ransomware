"""对比不同 PMU 计数器子集(6 / 12)的检测原型表现。

公平性:同一份底层合成数据(物理行为一致)、同一切分、同一超参,唯一变量是
“探针可并发采集的计数器数量” available。每个子集都跑完整 XGBoost+LSTM+stacking。
"""
from __future__ import annotations

import json
import os

import numpy as np

import data as data_mod
import model as model_mod
import train as train_mod


def run_subset(df_raw, available, win, stride, epochs, seed, device):
    df, seq = data_mod.engineer_features(df_raw, available=available)
    pk = data_mod.build_windows(df, seq, win, stride)
    X_seq, lengths = pk["X_seq"], pk["lengths"]
    X_tab, y, groups = pk["X_tab"], pk["y"], pk["groups"]

    idx_tr, idx_va, idx_te = train_mod.group_stratified_split(
        groups, y, 0.2, 0.16, seed)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tab[idx_tr])
    Xt_tr, Xt_va, Xt_te = (sc.transform(X_tab[i]) for i in (idx_tr, idx_va, idx_te))
    m, s = train_mod.fit_seq_scaler(X_seq[idx_tr], lengths[idx_tr])
    Xs_tr = train_mod.apply_seq_scaler(X_seq[idx_tr], lengths[idx_tr], m, s)
    Xs_va = train_mod.apply_seq_scaler(X_seq[idx_va], lengths[idx_va], m, s)
    Xs_te = train_mod.apply_seq_scaler(X_seq[idx_te], lengths[idx_te], m, s)
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
    pw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
        scale_pos_weight=pw, objective="binary:logistic", eval_metric="aucpr",
        early_stopping_rounds=40, n_jobs=0, random_state=seed, tree_method="hist")
    xgb.fit(Xt_tr, y_tr, eval_set=[(Xt_va, y_va)], verbose=False)
    p_xgb_va = xgb.predict_proba(Xt_va)[:, 1]
    p_xgb_te = xgb.predict_proba(Xt_te)[:, 1]

    lstm = model_mod.train_lstm(
        Xs_tr, lengths[idx_tr], y_tr, Xs_va, lengths[idx_va], y_va,
        n_features=len(seq), pos_weight=pw, epochs=epochs, device=device,
        seed=seed, verbose=False)
    p_lstm_va = model_mod.predict_lstm(lstm, Xs_va, lengths[idx_va], device)
    p_lstm_te = model_mod.predict_lstm(lstm, Xs_te, lengths[idx_te], device)

    from sklearn.linear_model import LogisticRegression
    Z_va = np.column_stack([p_xgb_va, p_lstm_va])
    Z_te = np.column_stack([p_xgb_te, p_lstm_te])
    meta = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Z_va, y_va)
    p_hyb_va = meta.predict_proba(Z_va)[:, 1]
    p_hyb_te = meta.predict_proba(Z_te)[:, 1]

    out = {}
    for name, pva, pte in [("XGBoost", p_xgb_va, p_xgb_te),
                           ("LSTM", p_lstm_va, p_lstm_te),
                           ("Hybrid", p_hyb_va, p_hyb_te)]:
        thr = train_mod.best_f1_threshold(y_va, pva)
        out[name] = train_mod.evaluate(y_te, pte, thr)
    return {"n_features": len(seq), "n_test": int(len(idx_te)),
            "pos_test": int(y_te.sum()), "results": out}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    data_path = os.path.join(root, "data", "processed", "dataset.csv")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_raw = data_mod.load_dataframe(data_path)
    print(f"数据 {len(df_raw)} 行 / {df_raw['pid'].nunique()} 执行流  device={device}\n")

    summary = {}
    for key in ["6", "12"]:
        avail = data_mod.COUNTER_SUBSETS[key]
        print(f"=== 计数器子集 {key} ({len(avail)} 个) ===")
        r = run_subset(df_raw, avail, win=32, stride=16, epochs=40,
                       seed=42, device=device)
        summary[key] = {"counters": avail, **r}
        print(f"  序列特征数 F={r['n_features']} | test={r['n_test']} "
              f"(正例 {r['pos_test']})")
        for nm in ["XGBoost", "LSTM", "Hybrid"]:
            m = r["results"][nm]
            print(f"  {nm:<8} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} "
                  f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
        print()

    print("================== 汇总 (Hybrid 测试集) ==================")
    print(f"{'计数器':>6} {'特征数':>6} {'ROC-AUC':>9} {'PR-AUC':>9} "
          f"{'F1':>8} {'Prec':>8} {'Recall':>8}")
    for key in ["6", "12"]:
        h = summary[key]["results"]["Hybrid"]
        print(f"{key:>6} {summary[key]['n_features']:>6} {h['roc_auc']:>9.4f} "
              f"{h['pr_auc']:>9.4f} {h['f1']:>8.4f} {h['precision']:>8.4f} "
              f"{h['recall']:>8.4f}")

    outp = os.path.join(root, "reports", "counter_comparison.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {outp}")


if __name__ == "__main__":
    main()
