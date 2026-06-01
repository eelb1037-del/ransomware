"""LSTM 时序分类器(PyTorch)与训练/推理工具。

与数据 schema 无关:输入是 (B, T, F) 的时序张量 + 每条序列的真实长度 lengths,
输出每条序列为“勒索”的 logit。配合掩码池化处理变长序列(尾部 padding)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)
from torch.utils.data import DataLoader, TensorDataset


class LSTMClassifier(nn.Module):
    """输入投影 -> 双向 LSTM -> (掩码均值池化 + 末步隐状态) -> MLP -> logit。"""

    def __init__(self, n_features: int, hidden: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        packed = pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)  # (B,T,2H)

        T = out.size(1)
        mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None])
        mask_f = mask.unsqueeze(-1).float()
        mean_pool = (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        last = out[torch.arange(out.size(0), device=x.device),
                   (lengths - 1).clamp(min=0)]
        return self.head(torch.cat([mean_pool, last], dim=1)).squeeze(-1)


def train_lstm(X_tr, len_tr, y_tr, X_va, len_va, y_va, n_features,
               pos_weight, epochs=40, batch_size=128, lr=1e-3, patience=6,
               device="cpu", seed=42, verbose=True):
    """训练 LSTM,按验证集 AUC 早停,返回最佳模型。"""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); np.random.seed(seed)

    model = LSTMClassifier(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(len_tr),
                       torch.from_numpy(y_tr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    Xva = torch.from_numpy(X_va).to(device)
    lva = torch.from_numpy(len_va).to(device)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, lb, yb in dl:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva, lva)).cpu().numpy()
        auc = roc_auc_score(y_va, p) if len(np.unique(y_va)) > 1 else 0.5
        if verbose:
            print(f"  [LSTM] epoch {ep:02d} loss={total/len(ds):.4f} "
                  f"val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  [LSTM] early stop @ {ep} (best={best_auc:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_lstm(model, X, lengths, device="cpu", batch_size=256):
    model.eval(); out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        lb = torch.from_numpy(lengths[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(xb, lb)).cpu().numpy())
    return np.concatenate(out)
