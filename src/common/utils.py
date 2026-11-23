"""공통 유틸: 시드 고정, 디바이스, Train/Validation Loss 곡선, 편향/분산 진단."""
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def get_device():
    """우선순위: 환경변수 DEVICE(cpu/cuda/xpu/mps) > CUDA > Intel XPU(Arc) > MPS > CPU"""
    if os.environ.get("DEVICE"):
        return torch.device(os.environ["DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def plot_loss_curves(history: dict, title: str, path, best_epoch: int | None = None):
    """X축 Epoch, Y축 Loss. Train/Validation 두 곡선을 겹쳐 Gap을 확인한다."""
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax.plot(ep, history["val_loss"], label="Validation Loss", color="#d62728", lw=2)
    ax.fill_between(ep, history["train_loss"], history["val_loss"], color="gray", alpha=0.12, label="Gap")
    if best_epoch:
        ax.axvline(best_epoch, ls="--", color="k", lw=1, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def diagnose_bias_variance(train_loss: float, val_loss: float, baseline_loss: float,
                           high_loss_ratio: float = 0.6, gap_ratio: float = 0.25) -> dict:
    """Train/Validation Loss로 편향/분산 상태를 규칙 기반 진단한다.

    - baseline_loss: '아무것도 학습하지 않은' 기준 Loss
      (분류: ln(C) = 균등 예측 CE, 회귀: Naive 예측의 MSE 등)
    - High Bias  : Train Loss 자체가 기준 대비 높음 (학습조차 잘 안 됨)
    - High Variance: (Validation − Train) / Validation 이 큼 (Train만 잘 맞춤)
    임계값은 휴리스틱이며 리포트에서 곡선과 함께 해석해야 한다.
    """
    rel_train = train_loss / max(baseline_loss, 1e-12)
    gap = val_loss - train_loss
    rel_gap = gap / max(val_loss, 1e-12)
    high_bias = rel_train > high_loss_ratio
    high_var = rel_gap > gap_ratio
    if high_bias and high_var:
        state = "High Bias + High Variance"
        fix = "모델 용량/학습 시간을 먼저 늘려 Train Loss를 낮춘 뒤, 정규화·데이터 증강으로 Gap을 줄인다."
    elif high_bias:
        state = "High Bias (Underfitting)"
        fix = "모델 복잡도↑(더 많은 층 Unfreeze, hidden 크기↑), 학습률/Epoch 조정, 정규화 완화."
    elif high_var:
        state = "High Variance (Overfitting)"
        fix = "정규화(Dropout↑, Weight Decay/L2), 데이터 증강, Early Stopping, Freeze 범위↑, 데이터 추가."
    else:
        state = "Balanced (Good fit)"
        fix = "현 설정 유지. 남은 오차는 오차 분석으로 데이터 측면 개선점을 찾는다."
    return dict(train_loss=train_loss, val_loss=val_loss, gap=gap, rel_gap=rel_gap,
                rel_train=rel_train, state=state, fix=fix)
