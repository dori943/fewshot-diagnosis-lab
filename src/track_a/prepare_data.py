"""Few-shot 데이터셋 구축 (클래스당 Train 50장 미만).

결과 구조 (ImageFolder 호환):
    data/fewshot/{train,validation,test}/{class}/img_xxxxx.png

사용 예)
  # 1) CIFAR-10 일부 클래스 (기본: 외형이 비슷한 동물 5종 → 클래스 유사성 오차 관찰에 유리)
  python -m src.track_a.prepare_data --source cifar10 --classes cat dog deer horse bird

  # 2) Kaggle 등에서 받은 폴더에서 필요한 클래스만 샘플링 복사 (shutil)
  #    src_dir/{class}/*.jpg  (Food-101: food-101/images)  또는
  #    src_dir/{class}.*.jpg  (Dogs vs Cats: train/cat.1.jpg 형태) 모두 지원
  python -m src.track_a.prepare_data --source folder --src_dir ~/food-101/images \
         --classes ramen pho bibimbap
"""
import argparse
import random
import shutil
from pathlib import Path

from src.common.utils import ensure_dir

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "validation", "test")


def _write_split(out: Path, split: str, cls: str, items, save_fn):
    d = ensure_dir(out / split / cls)
    for name, obj in items:
        save_fn(obj, d / name)


def from_cifar10(args, out: Path):
    from torchvision.datasets import CIFAR10

    tr = CIFAR10(args.cache, train=True, download=True)
    te = CIFAR10(args.cache, train=False, download=True)
    rng = random.Random(args.seed)
    for cls in args.classes:
        c = tr.classes.index(cls)
        tr_idx = [i for i, y in enumerate(tr.targets) if y == c]
        te_idx = [i for i, y in enumerate(te.targets) if y == c]
        rng.shuffle(tr_idx)
        rng.shuffle(te_idx)
        # Train/Validation은 CIFAR train split에서, Test는 CIFAR test split에서 → 중복 없음
        parts = {
            "train": [(f"img_tr{i:05d}.png", tr.data[i]) for i in tr_idx[: args.n_train]],
            "validation": [(f"img_va{i:05d}.png", tr.data[i]) for i in tr_idx[args.n_train: args.n_train + args.n_val]],
            "test": [(f"img_te{i:05d}.png", te.data[i]) for i in te_idx[: args.n_test]],
        }
        from PIL import Image
        for split, items in parts.items():
            _write_split(out, split, cls, items, lambda a, p: Image.fromarray(a).save(p))
        print(f"[{cls}] train={len(parts['train'])} validation={len(parts['validation'])} test={len(parts['test'])}")


def from_folder(args, out: Path):
    src = Path(args.src_dir).expanduser()
    rng = random.Random(args.seed)
    for cls in args.classes:
        if (src / cls).is_dir():
            files = [p for p in (src / cls).iterdir() if p.suffix.lower() in IMG_EXT]
        else:  # Dogs vs Cats 형식: cat.123.jpg
            files = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXT and p.name.lower().startswith(cls.lower() + ".")]
        files.sort()
        rng.shuffle(files)
        need = args.n_train + args.n_val + args.n_test
        assert len(files) >= need, f"{cls}: 이미지 {len(files)}장 < 필요 {need}장"
        cut = [0, args.n_train, args.n_train + args.n_val, need]
        for s, split in enumerate(SPLITS):
            items = [(p.name, p) for p in files[cut[s]: cut[s + 1]]]
            _write_split(out, split, cls, items, lambda a, p: shutil.copy2(a, p))
        print(f"[{cls}] copied {need} / {len(files)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cifar10", "folder"], default="cifar10")
    ap.add_argument("--src_dir", default=None)
    ap.add_argument("--classes", nargs="+", default=["cat", "dog", "deer", "horse", "bird"])
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--out", default="data/fewshot")
    ap.add_argument("--cache", default="data/_cache")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert args.n_train < 50, "Few-shot 조건: 클래스당 Train 50장 미만"

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (from_cifar10 if args.source == "cifar10" else from_folder)(args, out)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
