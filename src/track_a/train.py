"""Track A: Few-shot 전이 학습 — 직접 학습(Scratch) vs Linear Probing vs Partial vs Full Fine-tuning.

사용 예)
  python -m src.track_a.train --strategy all                    # 4가지 전략 비교
  python -m src.track_a.train --strategy full --aug strong --tag aug_strong   # 오차 분석 후 개선 실험
  # CPU에서 빠르게: --img_size 128 --epochs 15
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

from src.common.utils import (diagnose_bias_variance, ensure_dir, get_device, plot_loss_curves,
                              save_json, set_seed)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
STRATEGIES = ["scratch", "linear_probe", "partial", "full"]


# ----------------------------------------------------------------------------- transforms
def build_transforms(img_size: int, aug: str):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    eval_tf = T.Compose([T.Resize((img_size, img_size)), *norm])
    if aug == "none":
        return eval_tf, eval_tf
    basic = [T.RandomResizedCrop(img_size, scale=(0.7, 1.0)), T.RandomHorizontalFlip()]
    if aug == "basic":
        return T.Compose([*basic, *norm]), eval_tf
    # strong: 오차 분석 태그에 대응하는 증강
    #   어두운 조명/대비 → ColorJitter, 흐림 → GaussianBlur, 가림/배경 → RandomErasing + 작은 crop
    strong = [
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.03),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(), T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ]
    return T.Compose(strong), eval_tf


# ----------------------------------------------------------------------------- model
def build_model(arch: str, n_classes: int, strategy: str, dropout: float):
    pretrained = strategy != "scratch"
    if arch.startswith("resnet"):
        w = {"resnet18": models.ResNet18_Weights.DEFAULT, "resnet50": models.ResNet50_Weights.DEFAULT}[arch]
        m = getattr(models, arch)(weights=w if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, n_classes))
        head, last_stage = m.fc, m.layer4
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[-1].in_features, n_classes))
        head, last_stage = m.classifier, m.features[-3:]
    else:
        raise ValueError(arch)

    # Layer Freezing Strategy
    if strategy in ("linear_probe", "partial"):
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        if strategy == "partial":
            for p in last_stage.parameters():
                p.requires_grad = True
    return m, head, last_stage


def param_groups(m, head, strategy, lr, backbone_lr_mult):
    """Full/Partial fine-tuning 시 사전학습 층은 작은 LR(차등 학습률)로 천천히 업데이트."""
    head_ids = {id(p) for p in head.parameters()}
    body = [p for p in m.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": list(head.parameters()), "lr": lr}]
    if body:
        mult = 1.0 if strategy == "scratch" else backbone_lr_mult
        groups.append({"params": body, "lr": lr * mult})
    return groups


def set_train_mode(m, strategy, last_stage):
    """Linear probing: 동결된 Backbone의 BatchNorm 통계까지 고정(eval)해야 진짜 '동결'이다."""
    m.train()
    if strategy in ("linear_probe", "partial"):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        if strategy == "partial":
            for mod in last_stage.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.train()


# ----------------------------------------------------------------------------- loops
@torch.no_grad()
def evaluate(m, loader, device, criterion):
    m.eval()
    tot, n, correct, probs, ys = 0.0, 0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = m(x)
        tot += criterion(out, y).item() * len(y)
        n += len(y)
        p = out.softmax(1)
        correct += (p.argmax(1) == y).sum().item()
        probs.append(p.cpu())
        ys.append(y.cpu())
    return tot / n, correct / n, torch.cat(probs).numpy(), torch.cat(ys).numpy()


def run(args, strategy: str, data_dir: Path, out_root: Path):
    set_seed(args.seed)
    device = get_device()
    tf_train, tf_eval = build_transforms(args.img_size, "none" if strategy == "linear_probe" and args.aug == "auto" else
                                         ("basic" if args.aug == "auto" else args.aug))
    ds = {s: datasets.ImageFolder(data_dir / s, tf_train if s == "train" else tf_eval)
          for s in ("train", "validation", "test")}
    ds_train_clean = datasets.ImageFolder(data_dir / "train", tf_eval)  # 진단용: 증강 없는 Train
    classes = ds["train"].classes
    dl = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers)
    loaders = {"train": dl(ds["train"], True), "validation": dl(ds["validation"], False),
               "test": dl(ds["test"], False), "train_clean": dl(ds_train_clean, False)}

    m, head, last_stage = build_model(args.arch, len(classes), strategy, args.dropout)
    m.to(device)
    n_train_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    lr = args.lr * (10 if strategy == "linear_probe" else 1)  # Head만 학습할 땐 큰 LR이 유리
    opt = torch.optim.AdamW(param_groups(m, head, strategy, lr, args.backbone_lr_mult), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, best_ep, wait = math.inf, None, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        set_train_mode(m, strategy, last_stage)
        tl, tc, tn = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = m(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)
        sched.step()
        vl, va, _, _ = evaluate(m, loaders["validation"], device, criterion)
        hist["train_loss"].append(tl / tn); hist["train_acc"].append(tc / tn)
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        print(f"  [{strategy}] ep{ep:03d} train_loss={tl/tn:.4f} validation_loss={vl:.4f} train_acc={tc/tn:.3f} validation_acc={va:.3f}")
        if vl < best - 1e-4:
            best, best_state, best_ep, wait = vl, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep, 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  early stop @ {ep} (best {best_ep})")
                break

    m.load_state_dict(best_state)
    tr_l, tr_a, _, _ = evaluate(m, loaders["train_clean"], device, criterion)
    va_l, va_a, _, _ = evaluate(m, loaders["validation"], device, criterion)
    te_l, te_a, probs, ys = evaluate(m, loaders["test"], device, criterion)

    # 진단 기준 Loss: 균등 예측 시 CE = ln(C)
    diag = diagnose_bias_variance(tr_l, va_l, baseline_loss=math.log(len(classes)))
    name = strategy + (f"_{args.tag}" if args.tag else "")
    out = ensure_dir(out_root / name)
    plot_loss_curves(hist, f"Track A — {args.arch} / {name}", out / "loss_curve.png", best_ep)
    torch.save(m.state_dict(), out / "best.pt")

    # 전체 Test 예측 저장 (오차 분석 입력)
    files = [Path(p).as_posix() for p, _ in ds["test"].samples]
    pred = probs.argmax(1)
    srt = np.sort(probs, 1)
    df = pd.DataFrame({"file": files, "true": [classes[i] for i in ys], "pred": [classes[i] for i in pred],
                       "confidence": probs.max(1), "prob_true": probs[np.arange(len(ys)), ys],
                       "margin": srt[:, -1] - srt[:, -2], "correct": pred == ys})
    for i, c in enumerate(classes):
        df[f"p_{c}"] = probs[:, i]
    df.to_csv(out / "predictions_test.csv", index=False)

    cm = confusion_matrix(ys, pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion — {name}")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    metrics = dict(run=name, strategy=strategy, arch=args.arch, aug=args.aug, img_size=args.img_size,
                   trainable_params=n_train_params, best_epoch=best_ep, epochs_ran=len(hist["train_loss"]),
                   train_loss=tr_l, train_acc=tr_a, val_loss=va_l, val_acc=va_a, test_loss=te_l, test_acc=te_a,
                   n_errors=int((~df.correct).sum()), diagnosis=diag, minutes=(time.time() - t0) / 60,
                   n_train=len(ds["train"]), classes=classes)
    save_json(hist, out / "history.json")
    save_json(metrics, out / "metrics.json")
    print(f"  → {name}: Train acc {tr_a:.3f} | Validation acc {va_a:.3f} | Test acc {te_a:.3f} | {diag['state']}")
    return metrics


# ----------------------------------------------------------------------------- report
def summarize(out_root: Path, report_path: Path):
    rows = [json.load(open(p, encoding="utf-8")) for p in sorted(out_root.glob("*/metrics.json"))]
    if not rows:
        return
    df = pd.DataFrame([{k: r[k] for k in ["run", "trainable_params", "train_loss", "val_loss", "test_loss",
                                           "train_acc", "val_acc", "test_acc", "n_errors"]} |
                       {"gap": r["diagnosis"]["gap"], "state": r["diagnosis"]["state"]} for r in rows])
    rank = lambda run: next((i for i, s in enumerate(STRATEGIES) if run.startswith(s)), 99)
    df = df.assign(_r=df.run.map(rank)).sort_values(["_r", "run"]).drop(columns="_r")
    df.rename(columns=lambda c: c.replace("val_", "validation_")).to_csv(out_root / "comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(df))
    for k, (col, lab) in enumerate([("train_acc", "Train"), ("val_acc", "Validation"), ("test_acc", "Test")]):
        axes[0].bar(x + (k - 1) * 0.27, df[col], 0.27, label=lab)
    axes[0].set_xticks(x, df.run, rotation=20); axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy by strategy"); axes[0].legend(); axes[0].grid(axis="y", alpha=.3)
    axes[1].bar(x - 0.2, df.train_loss, 0.4, label="Train Loss"); axes[1].bar(x + 0.2, df.val_loss, 0.4, label="Validation Loss")
    axes[1].set_xticks(x, df.run, rotation=20); axes[1].set_title("Train vs Validation Loss (Gap)"); axes[1].legend()
    axes[1].grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_root / "strategy_comparison.png", dpi=150); plt.close(fig)

    lines = ["# Track A — 전이 학습 전략 비교 & 편향/분산 진단 (자동 생성)", "",
             "> `python -m src.track_a.train` 실행 시 자동 갱신. 해석/결정은 `reports/DIAGNOSIS_REPORT.md`에 작성.", "",
             "| Run | 학습 파라미터 | Train Loss | Validation Loss | Gap | Train Acc | Validation Acc | Test Acc | 오분류 | 진단 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.run} | {r.trainable_params:,} | {r.train_loss:.4f} | {r.val_loss:.4f} | {r.gap:+.4f} | "
                     f"{r.train_acc:.3f} | {r.val_acc:.3f} | {r.test_acc:.3f} | {r.n_errors} | {r.state} |")
    lines += ["", "## Run별 처방", ""]
    for r in rows:
        lines.append(f"- **{r['run']}** — {r['diagnosis']['state']}: {r['diagnosis']['fix']}")
    lines += ["", "![strategy](../outputs/track_a/strategy_comparison.png)", ""]
    for r in rows:
        lines.append(f"![{r['run']}](../outputs/track_a/{r['run']}/loss_curve.png)")
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/fewshot")
    ap.add_argument("--out", default="outputs/track_a")
    ap.add_argument("--report", default="reports/track_a_diagnosis.md")
    ap.add_argument("--strategy", default="all", choices=STRATEGIES + ["all"])
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--aug", default="auto", choices=["auto", "none", "basic", "strong"],
                    help="auto: linear_probe=none, 나머지=basic")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="", help="run 이름 접미사 (예: aug_strong)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]
    out_root = ensure_dir(args.out)
    for s in strategies:
        print(f"=== {s} ===")
        run(args, s, Path(args.data_dir), out_root)
    summarize(out_root, Path(args.report))


if __name__ == "__main__":
    main()
