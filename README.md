# AI가 어디서 자꾸 틀리는지 찾아내서 더 똑똑하게 만들기

두 트랙에 같은 성능 개선 사이클을 적용한다.
**① 베이스라인 → ② 딥러닝 → ③ 편향/분산 진단 + 오차 분석 → ④ 개선 + 정량화**

| 트랙 | 문제 | 모델 | 베이스라인 / 비교 대상 |
|---|---|---|---|
| A. 이미지 | Few-shot 분류 (클래스당 Train 40장) | ResNet18 / EfficientNet-B0 | 직접 학습(Scratch) vs Linear Probing vs Partial vs Full Fine-tuning |
| B. 시계열 | 다음 날 종가 예측 (AAPL, 2015~) | LSTM / GRU / Attn-LSTM | Naive, SMA(5/10/20), EMA(0.1/0.3/0.5), LinReg(time/lag5) |

## 레포 구조
```
src/
  common/utils.py              시드, Loss 곡선, 편향/분산 규칙 기반 진단
  track_a/prepare_data.py      Few-shot 데이터 구축 (CIFAR-10 일부 / Kaggle 폴더 샘플링)
  track_a/train.py             4가지 전이 학습 전략 학습·비교, 예측 저장, 혼동행렬
  track_a/error_analysis.py    오분류 추출 → (사람 태깅) → 통계·갤러리 → 개선 효과 비교
  track_b/data.py              시간순 분할, Train-only 정규화, 윈도우, 누수 자동 검사
  track_b/baselines.py         베이스라인 + 지표(MAE/RMSE/MAPE) + Diebold-Mariano 검정
  track_b/models.py            LSTM / GRU / Attention-LSTM
  track_b/train.py             베이스라인 먼저 → 딥러닝 → 진단/비교 리포트 자동 생성
  track_b/hparam_search.py     (보너스) Coarse-to-Fine Random Search + 파라미터 영향도
reports/
  DIAGNOSIS_REPORT.md          성능 진단 리포트 (메인, 사람이 해석 작성)
  LEAKAGE_PREVENTION.md        누수 방지 근거
  track_a_diagnosis.md         (자동) 전략별 Train/Validation/Test + 진단
  track_b_diagnosis.md         (자동) 시계열 모델 진단
  baseline_comparison.md       (자동) 베이스라인 비교표 + 개선율
  error_analysis_table.csv     (사람 태깅) 오차 분석표 원본
  error_analysis.md            (자동) 태그 통계 + 갤러리 + 개선 효과
docs/CV_THEORY_AND_EXTENSIONS.md  관련 이론 + 확장 기술 정리
docs/MISSION_KEY_CONCEPTS.md      미션 핵심 개념 (전공자용 심화)
```

## 실행
- Windows PowerShell: `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`
- macOS/Linux: `bash run_all.sh`

```bash
pip install -r requirements.txt     # Python 3.10+

# Track B — 인터넷 불가 시 --csv data/AAPL.csv
python -m src.track_b.train --ticker AAPL --start 2015-01-01
python -m src.track_b.hparam_search --csv data/AAPL.csv --arch GRU          # 보너스

# Track A — CPU면 --img_size 128 --epochs 15 권장
python -m src.track_a.prepare_data --source cifar10 --classes cat dog deer horse bird
python -m src.track_a.train --strategy all
python -m src.track_a.error_analysis extract --run outputs/track_a/full
#   ↳ reports/error_analysis_table.csv 의 tag 열을 이미지를 직접 보고 채운다 (Human-in-the-loop)
python -m src.track_a.error_analysis summarize --run outputs/track_a/full
python -m src.track_a.train --strategy full --aug strong --tag aug_strong    # 태그 통계 기반 처방
python -m src.track_a.error_analysis compare --before outputs/track_a/full --after outputs/track_a/full_aug_strong
```

## 필수 시각화 위치
| 그래프 | 파일 |
|---|---|
| Train/Validation Loss 곡선 | `outputs/track_a/<run>/loss_curve.png`, `outputs/track_b/loss_curve_<model>.png` |
| 예측 vs 실제 | `outputs/track_b/pred_vs_actual.png` (실제=파랑, 예측=빨강) |
| 베이스라인 비교 막대 | `outputs/track_b/baseline_comparison.png` |
| 오분류 이미지 갤러리 | `outputs/track_a/full/error_gallery_by_tag.png` |
| (추가) 전략 비교 / 혼동행렬 / Attention / 탐색 | `strategy_comparison.png`, `confusion_matrix.png`, `attention_weights.png`, `hparam/hparam_search.png` |

## 설계 메모
- **용어**: 문서·코드 전반 Train / Validation / Test 로 통일.
- **Track B 타깃**: 가격 대신 로그수익률을 학습하고 `ŷ(t)=y(t-1)·exp(r̂)`로 가격 복원 → 베이스라인과 같은 가격 공간·같은 Test 시점에서 비교.
- **공정성**: 딥러닝은 seed 3개 평균, 개선율과 함께 Diebold-Mariano p-value 보고(우연한 차이 배제).
- **Linear Probing**: 동결된 Backbone의 BatchNorm도 eval 모드로 고정(통계 갱신 방지).
- **진단용 Train Loss**: 증강·Dropout 없는 eval 모드로 다시 계산해 Validation과 같은 조건에서 Gap 측정.
