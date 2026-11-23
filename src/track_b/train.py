"""Track B: 시계열 예측 — (1) 베이스라인 먼저 측정 → (2) LSTM/GRU/Attn-LSTM → (3) 진단 → (4) 비교 정량화.

  python -m src.track_b.train --ticker AAPL --start 2015-01-01
  python -m src.track_b.train --csv data/AAPL.csv            # 오프라인
  python -m src.track_b.train --config outputs/track_b/hparam/best_config.json   # 탐색 결과 적용
"""
import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.utils import diagnose_bias_variance, ensure_dir, get_device, plot_loss_curves, save_json, set_seed
from src.track_b import baselines as B
from src.track_b.data import load_series, make_windows, time_split, to_price
from src.track_b.models import MODEL_ZOO, RNNForecaster

DEFAULT_CFG = dict(lookback=30, hidden=64, layers=2, dropout=0.2, lr=1e-3, weight_decay=1e-5,
                   batch_size=64, epochs=100, patience=12)


@torch.no_grad()
def predict(m, X, device, bs=512):
    m.eval()
    out, att = [], []
    for i in range(0, len(X), bs):
        p, a = m(torch.from_numpy(X[i:i + bs]).to(device))
        out.append(p.cpu())
        if a is not None:
            att.append(a.cpu())
    return torch.cat(out).numpy(), (torch.cat(att).numpy() if att else None)


def fit_model(sp, arch: str, cfg: dict, seed: int, device, verbose=False):
    set_seed(seed)
    Xtr, ytr, _ = make_windows(sp, "train", cfg["lookback"])
    Xva, yva, _ = make_windows(sp, "validation", cfg["lookback"])
    # 분할은 시간순(체크포인트 1). 미니배치 shuffle은 'Train 구간 내부' 샘플 순서만 섞는 것 → 누수 아님
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=cfg["batch_size"], shuffle=True)
    m = RNNForecaster(hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"], **MODEL_ZOO[arch]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])  # weight_decay = L2
    mse = nn.MSELoss()
    hist = {"train_loss": [], "val_loss": []}
    best, state, best_ep, wait = math.inf, None, 0, 0
    for ep in range(1, cfg["epochs"] + 1):
        m.train()
        tl = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = mse(m(x)[0], y)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(y)
        vl = float(((predict(m, Xva, device)[0] - yva) ** 2).mean())
        hist["train_loss"].append(tl / len(ytr)); hist["val_loss"].append(vl)
        if verbose and ep % 10 == 0:
            print(f"    ep{ep:03d} train={tl / len(ytr):.4f} validation={vl:.4f}")
        if vl < best - 1e-5:
            best, state, best_ep, wait = vl, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    m.load_state_dict(state)
    train_mse = float(((predict(m, Xtr, device)[0] - ytr) ** 2).mean())  # eval 모드(dropout off) Train Loss
    return m, hist, best_ep, train_mse, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODEL_ZOO))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/track_b")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    cfg = DEFAULT_CFG | (json.load(open(args.config)) if args.config else {})
    out, rep = ensure_dir(args.out), ensure_dir(args.reports)
    device = get_device()

    s = load_series(args.ticker, args.start, args.end, args.csv)
    sp = time_split(s)
    test_t = sp.idx("test", cfg["lookback"])
    train_t = sp.idx("train", cfg["lookback"])
    y_test = sp.price[test_t]

    # ---------------------------------------------------------------- (1) 베이스라인 먼저
    preds = B.run_all(sp.price, test_t, train_t)
    rows = [{"Model": k, **B.metrics(y_test, v), "type": "baseline"} for k, v in preds.items()]
    pd.DataFrame(rows).assign(measured_at=datetime.now().isoformat(timespec="seconds")) \
        .to_csv(out / "baseline_results.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------------------------------------------------------- (2) 딥러닝
    diag_rows, attn = [], None
    for arch in args.models:
        runs = []
        for seed in range(args.seeds):
            m, hist, best_ep, tr_mse, va_mse = fit_model(sp, arch, cfg, seed, device)
            Xte, _, t = make_windows(sp, "test", cfg["lookback"])
            z, a = predict(m, Xte, device)
            assert (t == test_t).all()  # 공정 비교: 동일 Test 구간
            runs.append((B.metrics(y_test, to_price(sp, t, z)), to_price(sp, t, z), hist, best_ep, tr_mse, va_mse))
            if a is not None and seed == 0:
                attn = a
            print(f"[{arch} seed{seed}] best_ep={best_ep} train_mse={tr_mse:.4f} validation_mse={va_mse:.4f} {runs[-1][0]}")
        met = {k: float(np.mean([r[0][k] for r in runs])) for k in runs[0][0]}
        std = {k + "_std": float(np.std([r[0][k] for r in runs])) for k in runs[0][0]}
        rows.append({"Model": arch, **met, **std, "type": "deep"})
        preds[arch] = runs[0][1]
        r0 = runs[0]
        plot_loss_curves(r0[2], f"Track B — {arch} (scaled MSE)", out / f"loss_curve_{arch}.png", r0[3])
        save_json(r0[2], out / f"history_{arch}.json")
        # 진단 기준: 평균 예측(=스케일 공간 0) 시 MSE ≈ 1 (Train 수익률 분산)
        d = diagnose_bias_variance(float(np.mean([r[4] for r in runs])), float(np.mean([r[5] for r in runs])), 1.0)
        if d["state"].startswith("High Bias"):  # 시계열용 처방 (이미지용 'Unfreeze' 문구 대체)
            d["fix"] = ("rel_train≈1이면 모델 용량 부족보다 Bayes error(예측 불가능한 잡음)일 가능성이 크다. "
                        "용량↑는 Validation Loss가 실제로 개선될 때만 채택하고, 우선 Naive 대비 성능·다변량 피쳐(거래량·지수)·"
                        "타깃 재정의(방향 분류, 다중 시점)를 검토한다. Validation < Train 이면 구간별 변동성 차이를 먼저 확인.")
        diag_rows.append({"Model": arch, **d, "best_epoch": r0[3], "epochs_ran": len(r0[2]["train_loss"])})

    res = pd.DataFrame(rows)
    res.to_csv(out / "all_results.csv", index=False)

    # ---------------------------------------------------------------- (3) 시각화
    dates = sp.dates[test_t]
    deep = args.models
    fig, axes = plt.subplots(len(deep), 2, figsize=(14, 3.6 * len(deep)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.2, 1]})
    for i, arch in enumerate(deep):
        for j, sl in enumerate([slice(None), slice(-60, None)]):
            ax = axes[i, j]
            ax.plot(dates[sl], y_test[sl], color="blue", lw=1.4, label="Actual")
            ax.plot(dates[sl], preds[arch][sl], color="red", lw=1.1, label=f"{arch} pred")
            if j == 1:
                ax.plot(dates[sl], preds["Naive"][sl], color="gray", ls="--", lw=1, label="Naive")
            ax.set_title(f"{arch} — {'Test (full)' if j == 0 else 'last 60 days'}")
            ax.set_xlabel("Date"); ax.set_ylabel("Close"); ax.grid(alpha=.3); ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out / "pred_vs_actual.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    col = ["#d62728" if t == "deep" else "#7f7f7f" for t in res.type]
    for ax, k in zip(axes, ["MAE", "RMSE"]):
        ax.bar(res.Model, res[k], color=col)
        ax.set_title(f"{k} on identical Test period (red = deep learning)")
        ax.set_ylabel(k); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=.3)
        lo = res[k].min()
        ax.set_ylim(lo * 0.9, min(res[k].max(), res.loc[res.Model == "Naive", k].item() * 3) * 1.05)
    fig.tight_layout(); fig.savefig(out / "baseline_comparison.png", dpi=140); plt.close(fig)

    if attn is not None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        L = cfg["lookback"]
        ax.bar(np.arange(-L, 0), attn.mean(0))
        ax.set_xlabel("time step relative to target (t-k)"); ax.set_ylabel("mean attention")
        ax.set_title("Attn-LSTM: average attention weight over Test windows")
        fig.tight_layout(); fig.savefig(out / "attention_weights.png", dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- (4) 리포트
    base = res[res.type == "baseline"]
    best_b = base.loc[base.RMSE.idxmin(), "Model"]
    imp = []
    for arch in deep:
        r = res[res.Model == arch].iloc[0]
        for ref in ["Naive", best_b] if best_b != "Naive" else ["Naive"]:
            b = res[res.Model == ref].iloc[0]
            dm, p = B.diebold_mariano(y_test, preds[arch], preds[ref])
            imp.append({"Deep model": arch, "vs": ref,
                        **{f"{k} 개선(%)": (b[k] - r[k]) / b[k] * 100 for k in ["MAE", "RMSE", "MAPE(%)"]},
                        "DM stat": dm, "p-value": p})
    imp = pd.DataFrame(imp)
    imp.to_csv(out / "improvement.csv", index=False)

    tbl = res[["Model", "MAE", "RMSE", "MAPE(%)", "type"]].round(4)
    L = [f"# 베이스라인 비교 리포트 — {args.ticker} (자동 생성)", "",
         f"- Test 구간: {sp.checks['test_period']} · 모든 모델 동일 Test 시점 {len(test_t)}개에서 평가",
         f"- 베이스라인은 딥러닝 학습 전에 먼저 측정됨 (`outputs/track_b/baseline_results.csv`의 measured_at)",
         f"- 딥러닝 지표는 seed {args.seeds}개 평균 (std는 `all_results.csv`). 설정: `{cfg}`", "",
         "## 1. 성능 비교표", "", tbl.to_markdown(index=False), "",
         "## 2. 개선율 (양수 = 딥러닝이 우수) & Diebold-Mariano 검정", "",
         "개선율 = (베이스라인 오차 − 딥러닝 오차) / 베이스라인 오차 × 100. DM p-value < 0.05 일 때만 '통계적으로 유의한 차이'.", "",
         imp.round(4).to_markdown(index=False), "",
         "![bar](../outputs/track_b/baseline_comparison.png)", "", "![pred](../outputs/track_b/pred_vs_actual.png)", ""]
    (rep / "baseline_comparison.md").write_text("\n".join(L), encoding="utf-8")

    dg = pd.DataFrame(diag_rows)[["Model", "train_loss", "val_loss", "gap", "rel_train", "state", "best_epoch", "epochs_ran"]] \
        .rename(columns={"train_loss": "Train Loss", "val_loss": "Validation Loss", "gap": "Gap"})
    L = ["# Track B — 편향/분산 진단 (자동 생성)", "",
         "- Loss = 스케일된 로그수익률의 MSE. 기준 Loss 1.0 = 'Train 평균 수익률만 예측'했을 때의 MSE (≈ 1 − R²).",
         "- ⚠️ 금융 수익률은 Bayes error(예측 불가능한 잡음)가 매우 크다. rel_train≈1 이 나와도 '모델이 작아서'가 아니라 "
         "'신호 자체가 적어서'일 수 있다 → Naive 대비 개선 여부와 함께 해석할 것.", "",
         dg.round(4).to_markdown(index=False), "", "## 처방", ""]
    L += [f"- **{r['Model']}** — {r['state']}: {r['fix']}" for r in diag_rows]
    L += [""] + [f"![{m}](../outputs/track_b/loss_curve_{m}.png)" for m in deep]
    (rep / "track_b_diagnosis.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    save_json(sp.checks, out / "leakage_checks.json")
    print("\n", tbl.to_string(index=False), "\n\n", imp.round(3).to_string(index=False))
    print("\nLeakage checks:", json.dumps(sp.checks, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
