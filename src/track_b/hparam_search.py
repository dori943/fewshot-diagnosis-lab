"""보너스: Random Search → Coarse-to-Fine 하이퍼파라미터 탐색 (선택 기준은 Validation Loss만, Test는 사용 금지).

  python -m src.track_b.hparam_search --csv data/AAPL.csv --arch GRU --n_coarse 24 --n_fine 12
  python -m src.track_b.train --csv data/AAPL.csv --config outputs/track_b/hparam/best_config.json
"""
import argparse
import gc
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from src.common.utils import ensure_dir, get_device, save_json
from src.track_b.data import load_series, time_split
from src.track_b.train import DEFAULT_CFG, fit_model

# (kind, low, high) — log 스케일 파라미터는 로그 균등 샘플링 (Random search의 핵심)
SPACE = {"lr": ("log", 1e-4, 1e-2), "hidden": ("int_log", 16, 256), "layers": ("int", 1, 3),
         "dropout": ("lin", 0.0, 0.5), "weight_decay": ("log", 1e-7, 1e-3), "lookback": ("int", 10, 90)}


def sample(rng, space):
    c = {}
    for k, (kind, lo, hi) in space.items():
        if kind == "log":
            c[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        elif kind == "int_log":
            c[k] = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
        elif kind == "int":
            c[k] = int(rng.integers(lo, hi + 1))
        else:
            c[k] = float(rng.uniform(lo, hi))
    return c


def narrow(df, space, top_k=3):
    """Coarse 결과 상위 k개가 차지한 범위로 탐색 공간을 좁힌다 (약간의 여유 포함)."""
    top = df.nsmallest(top_k, "val_loss")
    new = {}
    for k, (kind, lo, hi) in space.items():
        a, b = top[k].min(), top[k].max()
        if kind in ("log", "int_log"):
            a, b = max(lo, a / 1.5), min(hi, b * 1.5)
        else:
            pad = (hi - lo) * 0.1
            a, b = max(lo, a - pad), min(hi, b + pad)
        new[k] = (kind, type(lo)(a) if kind.startswith("int") else a, type(hi)(b) if kind.startswith("int") else b)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL"); ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--csv", default=None); ap.add_argument("--arch", default="GRU")
    ap.add_argument("--n_coarse", type=int, default=24); ap.add_argument("--n_fine", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/track_b/hparam")
    args = ap.parse_args()
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    sp = time_split(load_series(args.ticker, args.start, None, args.csv))
    device = get_device()

    rows, space = [], SPACE
    for stage, n in [("coarse", args.n_coarse), ("fine", args.n_fine)]:
        if stage == "fine":
            space = narrow(pd.DataFrame(rows), SPACE)
            save_json(space, out / "fine_space.json")
        for i in range(n):
            c = DEFAULT_CFG | sample(rng, space) | {"epochs": args.epochs, "patience": 8}
            m, hist, ep, tr, va = fit_model(sp, args.arch, c, seed=0, device=device)
            del m  # trial마다 장치 메모리 해제 (누적 방지)
            gc.collect()
            if device.type == "xpu":
                torch.xpu.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            rows.append({**{k: c[k] for k in SPACE}, "stage": stage, "train_loss": tr, "val_loss": va, "best_epoch": ep})
            print(f"[{stage} {i + 1}/{n}] val={va:.4f} train={tr:.4f} {({k: round(c[k], 5) for k in SPACE})}")
    df = pd.DataFrame(rows)
    df.rename(columns={"val_loss": "validation_loss"}).to_csv(out / "trials.csv", index=False)

    # 영향도: 랜덤포레스트 중요도 + 스피어만 상관 (log 파라미터는 log 변환)
    X = df[list(SPACE)].copy()
    for k, (kind, *_r) in SPACE.items():
        if "log" in kind:
            X[k] = np.log10(X[k])
    rf = RandomForestRegressor(300, random_state=0).fit(X, df.val_loss)
    imp = pd.DataFrame({"param": list(SPACE), "rf_importance": rf.feature_importances_,
                        "spearman_rho": [spearmanr(X[k], df.val_loss)[0] for k in SPACE]}).sort_values("rf_importance", ascending=False)
    imp.to_csv(out / "param_importance.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    ax_list = axes.ravel()
    for ax, k in zip(ax_list, SPACE):
        for st, col in [("coarse", "#7f7f7f"), ("fine", "#d62728")]:
            d = df[df.stage == st]
            ax.scatter(d[k], d.val_loss, c=col, s=22, alpha=.8, label=st)
        if "log" in SPACE[k][0]:
            ax.set_xscale("log")
        ax.set_xlabel(k); ax.set_ylabel("Validation Loss"); ax.grid(alpha=.3)
    ax_list[0].legend()
    ax_list[6].barh(imp.param[::-1], imp.rf_importance[::-1], color="#4c72b0")
    ax_list[6].set_title("Param importance (RandomForest)")
    best_so_far = df.val_loss.cummin()
    ax_list[7].plot(np.arange(1, len(df) + 1), df.val_loss, "o", ms=3, color="gray", alpha=.6)
    ax_list[7].plot(np.arange(1, len(df) + 1), best_so_far, color="#d62728")
    ax_list[7].axvline(args.n_coarse + .5, ls="--", color="k"); ax_list[7].set_title("Best Validation Loss so far")
    ax_list[7].set_xlabel("trial")
    fig.suptitle(f"Coarse-to-Fine random search — {args.arch}")
    fig.tight_layout(); fig.savefig(out / "hparam_search.png", dpi=140); plt.close(fig)

    best = df.loc[df.val_loss.idxmin()]
    cfg = {k: (int(best[k]) if SPACE[k][0].startswith("int") else float(best[k])) for k in SPACE}
    save_json(cfg, out / "best_config.json")
    print(imp.to_string(index=False)); print("best:", cfg, "validation_loss=", best.val_loss)


if __name__ == "__main__":
    main()
