"""Track A: 오차 분석 (Human-in-the-loop).

1) extract   : 오분류 이미지 추출 + 번호 붙인 리뷰 시트 + 태깅용 CSV(자동 '힌트'만 채움)
2) (사람)     : 리뷰 시트/원본 이미지를 직접 보고 CSV의 `tag` 열을 채운다  ← 필수, 자동화 금지
3) summarize : 태그 통계 + 태그별 오분류 갤러리 + reports/error_analysis.md 생성
4) compare   : 개선(증강 등) 후 Run에서 태그별로 오분류가 얼마나 회복됐는지 정량화

  python -m src.track_a.error_analysis extract   --run outputs/track_a/full
  python -m src.track_a.error_analysis summarize --table reports/error_analysis_table.csv
  python -m src.track_a.error_analysis compare   --table reports/error_analysis_table.csv \
         --before outputs/track_a/full --after outputs/track_a/full_aug_strong
"""
import argparse
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from src.common.utils import ensure_dir

TAGS = {
    "데이터 품질": "데이터 증강 (밝기/대비/블러/해상도)",
    "배경/잡음": "배경 제거·Crop/RandomErasing 증강",
    "클래스 유사성": "특징 학습 강화 (Fine-tuning 범위↑, 고해상도, Metric learning)",
    "라벨 오류": "데이터 정제 (재라벨링/제외)",
    "모델 한계": "모델 구조 변경 (더 큰 Backbone, Attention/ViT)",
}
COLS = ["no", "file", "true", "pred", "confidence", "prob_true", "auto_hint", "tag", "detail", "note"]

for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    from matplotlib import font_manager
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------------- image stats (힌트용)
def image_stats(path):
    g = Image.open(path).convert("L")
    a = np.asarray(g, dtype=np.float32) / 255.0
    lap = np.asarray(g.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128)), np.float32)
    edges = np.asarray(g.filter(ImageFilter.FIND_EDGES), np.float32) / 255.0
    return dict(brightness=a.mean(), contrast=a.std(), sharpness=lap.var(), edge_density=(edges > 0.2).mean(),
                resolution=min(g.size))


def auto_hints(row, ref):
    """사람이 볼 때 참고할 '의심' 힌트. 최종 태그가 아니다."""
    h = []
    if row.brightness < ref["brightness"][0]: h.append("어두움?")
    if row.brightness > ref["brightness"][1]: h.append("과노출?")
    if row.contrast < ref["contrast"][0]: h.append("저대비?")
    if row.sharpness < ref["sharpness"][0]: h.append("흐림?")
    if row.edge_density > ref["edge_density"][1]: h.append("복잡한 배경?")
    if row.confidence >= 0.8: h.append("고신뢰 오답")
    if row.margin < 0.15: h.append("경계 사례(2지선다)")
    return ", ".join(h) or "-"


def extract(args):
    run = Path(args.run)
    df = pd.read_csv(run / "predictions_test.csv")
    st = pd.DataFrame([image_stats(f) for f in df.file])
    df = pd.concat([df, st], axis=1)
    ref = {c: (df[c].quantile(0.15), df[c].quantile(0.85)) for c in st.columns}

    err = df[~df.correct].sort_values("confidence", ascending=False).reset_index(drop=True)
    if len(err) < 30:
        print(f"[경고] 오분류 {len(err)}건 < 30건. Test 샘플(--n_test)을 늘리거나 더 어려운 클래스를 선택하세요.")
    err["no"] = np.arange(1, len(err) + 1)
    err["auto_hint"] = [auto_hints(r, ref) for r in err.itertuples()]
    err["tag"] = err["detail"] = err["note"] = ""

    img_dir = ensure_dir(run / "misclassified")
    for r in err.itertuples():
        shutil.copy2(r.file, img_dir / f"{r.no:03d}_{r.true}_as_{r.pred}{Path(r.file).suffix}")
    err["file"] = err.file.map(lambda p: Path(p).name)

    table = Path(args.table)
    if table.exists() and not args.force:
        raise SystemExit(f"{table} 이미 존재(사람이 태깅한 내용 보호). 덮어쓰려면 --force")
    ensure_dir(table.parent)
    err[COLS].round(4).to_csv(table, index=False, encoding="utf-8-sig")

    # 번호가 붙은 리뷰 시트 (사람이 보면서 태깅)
    per = 30
    for s in range(0, len(err), per):
        chunk = err.iloc[s:s + per]
        cols = 6
        rows = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, r in zip(np.atleast_1d(axes).ravel(), chunk.itertuples()):
            ax.imshow(Image.open(img_dir / f"{r.no:03d}_{r.true}_as_{r.pred}{Path(r.file).suffix}").convert("RGB"))
            ax.set_title(f"#{r.no} {r.true}→{r.pred}\nconf {r.confidence:.2f}", fontsize=8)
        fig.tight_layout()
        fig.savefig(run / f"review_sheet_{s // per + 1}.png", dpi=130)
        plt.close(fig)
    print(f"오분류 {len(err)}건 → {table}\n이미지: {img_dir}\n리뷰 시트: {run}/review_sheet_*.png")
    print("다음 단계: 이미지를 직접 확인하고 CSV의 tag 열을 아래 중 하나로 채우세요 →", list(TAGS))


# ----------------------------------------------------------------------------- summarize
def load_tagged(table, run_dir, allow_draft):
    df = pd.read_csv(table, encoding="utf-8-sig", dtype={"tag": str, "detail": str, "note": str}).fillna("")
    df.columns = df.columns.str.strip().str.lower()  # Excel이 헤더 'true'를 불리언 'TRUE'로 바꾸는 문제 대응
    df["tag"] = df.tag.astype(str).str.split(";").str[0].str.strip()
    missing = df.tag.eq("")
    if missing.any() and not allow_draft:
        raise SystemExit(f"tag 미입력 {missing.sum()}건. 사람이 직접 확인 후 채워야 합니다. "
                         f"(구조 확인용 초안은 --allow_draft)")
    bad = set(df.tag[~missing]) - set(TAGS)
    if bad:
        raise SystemExit(f"허용되지 않은 태그: {bad}. 허용: {list(TAGS)}")
    if missing.any():
        df.loc[missing, "tag"] = "미태깅(DRAFT)"
    df["path"] = [Path(run_dir) / "misclassified" / f"{r.no:03d}_{r.true}_as_{r.pred}{Path(r.file).suffix}"
                  for r in df.itertuples()]
    return df, missing.any()


def gallery(df, path, per_tag=8):
    tags = df.tag.value_counts().index.tolist()
    fig, axes = plt.subplots(len(tags), per_tag, figsize=(per_tag * 1.9, len(tags) * 2.3), squeeze=False)
    for i, t in enumerate(tags):
        sub = df[df.tag == t].head(per_tag)
        for j in range(per_tag):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(sub):
                r = sub.iloc[j]
                ax.imshow(Image.open(r.path).convert("RGB"))
                ax.set_title(f"#{r.no} {r.true}→{r.pred}\n{r.confidence:.2f}", fontsize=7)
        axes[i, 0].text(-0.15, 0.5, f"{t}\n(n={int((df.tag == t).sum())})", transform=axes[i, 0].transAxes,
                        ha="right", va="center", fontsize=10, fontweight="bold")
    fig.suptitle("Misclassified gallery grouped by failure tag", fontsize=12)
    fig.tight_layout(rect=(0.08, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summarize(args):
    table = Path(args.table)
    run_dir = Path(args.run)
    df, draft = load_tagged(table, run_dir, args.allow_draft)
    stats = df.tag.value_counts().rename_axis("tag").reset_index(name="count")
    stats["ratio(%)"] = (stats["count"] / len(df) * 100).round(1)
    stats["mean_conf"] = stats.tag.map(df.groupby("tag").confidence.mean()).round(3)
    stats["개선 방향"] = stats.tag.map(lambda t: TAGS.get(t, "-"))
    pairs = (df.true + " → " + df.pred).value_counts().head(10)

    gal = run_dir / "error_gallery_by_tag.png"
    gallery(df, gal)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(stats.tag[::-1], stats["count"][::-1], color="#4c72b0")
    ax.set_xlabel("count"); ax.set_title("Failure tags"); fig.tight_layout()
    fig.savefig(run_dir / "error_tag_stats.png", dpi=140); plt.close(fig)

    rel = lambda p: Path("..") / p
    L = ["# 오차 분석표 (Error Analysis Table)", ""]
    if draft:
        L += ["> ⚠️ **DRAFT** — 일부 행의 태그가 비어 있습니다. 제출 전 반드시 사람이 직접 이미지를 확인하고 태깅하세요.", ""]
    L += [f"- 분석 대상 Run: `{run_dir}`  |  오분류 {len(df)}건 (최소 기준 30건 {'충족' if len(df) >= 30 else '미달'})",
          f"- 원본 표: `{table}` (사람이 태깅한 CSV)", "", "## 1. 태그 통계", "",
          stats.to_markdown(index=False), "", f"![tags]({rel(run_dir / 'error_tag_stats.png').as_posix()})", "",
          "## 2. 자주 틀리는 클래스 쌍 (실제 → 예측)", "", pairs.rename_axis("pair").reset_index(name="count").to_markdown(index=False),
          "", "## 3. 태그별 오분류 갤러리", "", f"![gallery]({rel(gal).as_posix()})", "",
          "## 4. 전체 오차 분석표", "",
          df[["no", "file", "true", "pred", "confidence", "tag", "detail", "note"]]
          .assign(**{"실제/예측": df.true + " → " + df.pred})
          [["no", "file", "실제/예측", "confidence", "tag", "detail", "note"]]
          .rename(columns={"no": "번호", "file": "파일명", "confidence": "신뢰도", "tag": "실패 원인 태그",
                           "detail": "세부", "note": "메모"}).round(3).to_markdown(index=False), ""]
    Path(args.report).write_text("\n".join(L), encoding="utf-8")
    print(stats.to_string(index=False))
    print(f"→ {args.report}")


# ----------------------------------------------------------------------------- compare (개선 효과)
def compare(args):
    before = pd.read_csv(Path(args.before) / "predictions_test.csv")
    after = pd.read_csv(Path(args.after) / "predictions_test.csv")
    df, _ = load_tagged(args.table, args.before, allow_draft=True)
    # 키 = 클래스폴더/파일명 (서로 다른 클래스에 같은 파일명이 있을 수 있음)
    after["key"] = after.file.map(lambda p: f"{Path(p).parent.name}/{Path(p).name}")
    fixed = (df.true + "/" + df.file).map(after.set_index("key").correct)
    df["fixed"] = fixed.astype(bool)
    rec = df.groupby("tag").agg(n=("fixed", "size"), recovered=("fixed", "sum"))
    rec["recovery(%)"] = (rec.recovered / rec.n * 100).round(1)
    acc_b, acc_a = before.correct.mean(), after.correct.mean()
    L = ["", "## 5. 개선 실험 효과 (오차 분석 → 처방 → 재학습)", "",
         f"- Before `{args.before}` Test Acc **{acc_b:.3f}** → After `{args.after}` Test Acc **{acc_a:.3f}** "
         f"(**{(acc_a - acc_b) * 100:+.1f}%p**)",
         f"- 새로 틀린 이미지(회귀): {int((before.correct & ~after.correct).sum())}건", "",
         "### 태그별 회복률 (Before에서 틀린 이미지 중 After에서 맞춘 비율)", "", rec.reset_index().to_markdown(index=False), ""]
    with open(args.report, "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--run", required=True)
    e.add_argument("--table", default="reports/error_analysis_table.csv"); e.add_argument("--force", action="store_true")
    s = sp.add_parser("summarize"); s.add_argument("--table", default="reports/error_analysis_table.csv")
    s.add_argument("--run", default="outputs/track_a/full"); s.add_argument("--report", default="reports/error_analysis.md")
    s.add_argument("--allow_draft", action="store_true")
    c = sp.add_parser("compare"); c.add_argument("--table", default="reports/error_analysis_table.csv")
    c.add_argument("--before", required=True); c.add_argument("--after", required=True)
    c.add_argument("--report", default="reports/error_analysis.md")
    a = ap.parse_args()
    {"extract": extract, "summarize": summarize, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
