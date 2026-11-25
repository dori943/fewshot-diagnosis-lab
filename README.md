# fewshot-diagnosis-lab
### AI가 어디서 자꾸 틀리는지 찾아내서 더 똑똑하게 만들기

Few-shot 이미지 분류(Track A)와 주가 시계열 예측(Track B)에 **같은 성능 개선 사이클**을 적용한 실험 레포.

> **① 베이스라인 → ② 딥러닝 → ③ 편향/분산 진단 + 오차 분석 → ④ 개선 적용 + 정량화**

---

## 1. 한눈에 보는 결과

### Track A — Few-shot 전이 학습 (CIFAR-10 5클래스: cat·dog·deer·horse·bird, 클래스당 Train 40 / Validation 20 / Test 100장)

| Run | Backbone / 입력 크기 | 학습 범위 | Validation Acc | Test Acc |
|---|---|---|---:|---:|
| scratch (직접 학습) | ResNet18 (랜덤 초기화) / 128px | 전체 | 0.420 | 0.436 |
| linear_probe | ResNet18 (ImageNet) / 128px | 분류기만 | 0.750 | 0.756 |
| **partial** (기준 모델) | ResNet18 (ImageNet) / 128px | layer4 + 분류기 | 0.820 | 0.750 |
| full | ResNet18 (ImageNet) / 128px | 전체 (backbone LR ×0.1) | 0.780 | 0.714 |
| 처방 1-a res224_only | ResNet18 / 224px | partial | 0.800 | 0.772 |
| 처방 1-b res224 | ResNet18 / 224px + label smoothing | partial | 0.820 | 0.778 |
| **처방 1-c effb0_224 (최종)** | **EfficientNet-B0 / 224px** | partial | **0.870** | **0.806** |
| 처방 2 aug_strong | ResNet18 / 128px + 강한 증강·정규화 | partial | 0.800 | 0.752 |

- 사전학습 효과: Test 0.436 → 0.756 (**+32%p**)
- Freezing 전략: 데이터가 적을수록 학습 범위가 좁은 쪽이 유리 (linear_probe ≈ partial > full)
- 오차 분석(오분류 125건 직접 태깅): 모델 한계 45.6% · 클래스 유사성 28.0% · 데이터 품질 17.6% · 배경/잡음 8.8%
- 원인 74%를 겨냥한 처방(입력 크기 + Backbone)이 **+5.6%p**, 26%를 겨냥한 증강은 +0.2%p

### Track B — 시계열 (AAPL 일봉 2015-01-02 ~ 2026-11-13, Train/Validation/Test = 7/1/2 시간순)

| Model | MAE | RMSE | MAPE(%) | vs Naive RMSE 개선 | DM p-value |
|---|---:|---:|---:|---:|---:|
| **Naive** | 2.971 | 4.296 | 1.186 | – | – |
| SMA(5 / 10 / 20) | 4.83 / 6.46 / 8.64 | 6.54 / 8.22 / 10.66 | 1.94 / 2.58 / 3.41 | – | – |
| EMA(α 0.1 / 0.3 / 0.5) | 7.24 / 4.53 / 3.61 | 9.12 / 6.01 / 5.04 | 2.86 / 1.82 / 1.45 | – | – |
| LinReg(time) / LinReg(lag5) | 54.85 / 2.975 | 63.22 / 4.308 | 20.23 / 1.190 | – | – |
| LSTM | 2.959 | 4.291 | 1.182 | +0.11% | 0.38 |
| GRU | 2.968 | 4.298 | 1.186 | −0.06% | 0.91 |
| Attn-LSTM (보너스) | 2.959 | 4.291 | 1.182 | +0.12% | 0.52 |

- 딥러닝은 Naive를 **유의하게 이기지 못함** (하이퍼파라미터 탐색 후 재학습해도 동일, `reports/tuned/`)
- 주가 ≈ 랜덤워크 → 최적 예측이 곧 Naive. 예측선이 실제값을 하루 늦게 따라감 = 모델이 Naive를 학습
- 누수 자동 검사 전 항목 통과 (`outputs/track_b/leakage_checks.json`)

---

## 2. 과제 목표별 설명 (평가 대응)

| 과제 목표 | 핵심 답 | 근거 파일 |
|---|---|---|
| **① Loss 차이로 편향/분산 진단** | Bias = 기준 Loss(분류 ln5, 시계열 평균예측 MSE 1.0) 대비 Train Loss, Variance = Train–Validation 격차. **Track A**: 전 Run Train Acc≈1.0, Gap 큼 → **High Variance**. **Track B**: Train Loss≈기준 → 자동 진단은 High Bias지만, 하이퍼파라미터를 바꿔도 Validation Loss 불변(0.507~0.516) → 원인은 용량 부족이 아닌 **Bayes error(예측 불가 잡음)**, 모델 확대 처방은 기각. Validation Loss < Train Loss는 Validation 기간 변동성이 낮아서(분산 비 0.51) | `DIAGNOSIS_REPORT.md` §1·§2.2·§3.2, `src/common/utils.py` `diagnose_bias_variance()`, `outputs/**/loss_curve*.png` |
| **② 데이터 크기별 Freezing 범위** | 클래스당 40장 → 학습 범위가 좁을수록 유리 (linear_probe 0.756 ≈ partial 0.750 > full 0.714). full은 사전학습 특징이 소량 데이터에 맞춰 왜곡. 결정 기준 = 데이터 크기 × 도메인 유사도 → **Partial 채택**(Validation 1위), 비용이 작은 linear_probe도 동등한 대안 | `DIAGNOSIS_REPORT.md` §2.3, `src/track_a/train.py` `build_model()`·`param_groups()`·`set_train_mode()` |
| **③ CNN vs RNN 구조·도메인** | CNN: 공간 지역성·가중치 공유 → 이미지, 사전학습 특징 전이가 핵심. RNN: 시간 순서를 은닉 상태로 요약 → 시계열. LSTM은 게이트·셀 상태로 장기 의존성 보존, GRU는 게이트 2개로 경량, Attention은 중요한 과거 시점 선택 | `docs/MISSION_KEY_CONCEPTS.md` §2.8, `src/track_b/models.py` |
| **④ 오차 분석 → 증강 전략** | 125건 직접 태깅 → 태그 비율 = 개선 상한(ceiling). 74%(모델 한계·클래스 유사성) → 입력 크기·Backbone 처방(+5.6%p, 모델 한계 회복률 52.6%). 26%(데이터 품질·배경) → 강한 증강(+0.2%p, 가설 기각: 32px 원본에 증강이 새 정보를 못 줌) | `reports/error_analysis.md`, `reports/error_analysis_table.csv`, `outputs/track_a/partial/error_gallery_by_tag.png` |
| **⑤ 딥러닝 vs 베이스라인 정량 비교** | 동일 Test 597일, MAE/RMSE/MAPE + 개선율. LSTM 계열 −0.06~+0.15%, DM p>0.38 → "우수하다고 말할 수 없음". Track A는 scratch 대비 사전학습 +32%p, 처방 +5.6%p (Test 500장 신뢰구간 ±4%p와 비교) | `reports/baseline_comparison.md`, `outputs/track_b/baseline_comparison.png`, `pred_vs_actual.png` |
| **⑥ 시계열 Data Leakage** | 시간순 분할 / Train 구간만으로 정규화 fit / 입력 윈도우 t−L…t−1만 사용 / 분할 전 생성하되 과거값만 쓰는 피쳐 / 선택은 Validation만, Test 1회 / 시계열 증강 미적용 — 모두 코드 assert·자동 검사 | `reports/LEAKAGE_PREVENTION.md`, `src/track_b/data.py`, `outputs/track_b/leakage_checks.json` |

---

## 3. 미션 요구사항 체크리스트

| 요구사항 | 위치 |
|---|---|
| 전이 학습 + RNN 소스 코드 | `src/track_a/`, `src/track_b/` |
| 성능 진단 리포트 (편향/분산 + 해결 전략) | `reports/DIAGNOSIS_REPORT.md` (자동 표: `track_a_diagnosis.md`, `track_b_diagnosis.md`) |
| 오차 분석표 (30건 이상, 사람 태깅 + 통계) | `reports/error_analysis_table.csv` (125건), `reports/error_analysis.md` |
| 베이스라인 비교 리포트 | `reports/baseline_comparison.md`, 튜닝 후 `reports/tuned/baseline_comparison.md` |
| 누수 방지 근거 | `reports/LEAKAGE_PREVENTION.md` |
| 클래스당 50장 미만 | `src/track_a/prepare_data.py` (`assert n_train < 50`, 40장 사용) |
| Linear Probing vs Fine-tuning | `outputs/track_a/strategy_comparison.png`, `outputs/track_a/comparison.csv` |
| 베이스라인 ≥2개 (SMA 5/10/20, EMA 0.1/0.3/0.5, LinReg time/lag) | `src/track_b/baselines.py` |
| 베이스라인을 딥러닝보다 먼저 측정 | `src/track_b/train.py` 실행 순서, `outputs/track_b/baseline_results.csv`의 `measured_at` |
| 7/1/2 시간순 분할, Train만으로 정규화 | `src/track_b/data.py` `time_split()` |
| 보너스: Random / Coarse-to-Fine 탐색 + 영향도 | `src/track_b/hparam_search.py`, `outputs/track_b/hparam/` |
| 보너스: Attention | `src/track_b/models.py` `AdditiveAttention`, `outputs/track_b/attention_weights.png` |
| 용어 Train / Validation / Test 통일 | 문서·리포트 전체 |

### 필수 시각화 4종
| 그래프 | 파일 |
|---|---|
| Train/Validation Loss 곡선 | `outputs/track_a/<run>/loss_curve.png`, `outputs/track_b/loss_curve_<model>.png` |
| 예측 vs 실제 (실제=파랑, 예측=빨강) | `outputs/track_b/pred_vs_actual.png` |
| 베이스라인 비교 막대그래프 (MAE/RMSE) | `outputs/track_b/baseline_comparison.png` |
| 오분류 이미지 갤러리 (태그별 그룹) | `outputs/track_a/partial/error_gallery_by_tag.png` |

추가: 전략 비교 `outputs/track_a/strategy_comparison.png`, 혼동행렬 `outputs/track_a/<run>/confusion_matrix.png`, 태그 통계 `outputs/track_a/partial/error_tag_stats.png`, 탐색 결과 `outputs/track_b/hparam/hparam_search.png`

---

## 4. 레포 구조
```
src/
  common/utils.py              시드, 디바이스(CUDA/Intel XPU/CPU), Loss 곡선, 편향/분산 진단
  track_a/prepare_data.py      Few-shot 데이터 구축 (CIFAR-10 일부 / Kaggle 폴더 샘플링)
  track_a/train.py             전이 학습 전략(scratch/linear_probe/partial/full) 학습·비교
  track_a/error_analysis.py    오분류 추출 → (사람 태깅) → 통계·갤러리 → 처방 효과 비교
  track_b/data.py              시간순 분할, Train-only 정규화, 윈도우, 누수 자동 검사
  track_b/baselines.py         베이스라인 9종 + MAE/RMSE/MAPE + Diebold-Mariano 검정
  track_b/models.py            LSTM / GRU / Attention-LSTM
  track_b/train.py             베이스라인 먼저 → 딥러닝 → 진단·비교 리포트 자동 생성
  track_b/hparam_search.py     (보너스) Coarse-to-Fine Random Search + 파라미터 영향도
reports/                       진단·오차 분석·베이스라인·누수 방지 리포트
outputs/                       그래프, 예측 결과, 지표 (track_a, track_b, track_b_tuned)
docs/                          이론 정리 (MISSION_KEY_CONCEPTS.md, CV_THEORY_AND_EXTENSIONS.md)
data/AAPL.csv                  다운로드 원본 고정 (오프라인 재현용)
```

---

## 5. 실행 (실제 실험에 사용한 명령)

```powershell
conda create -n fewshot python=3.11 -y; conda activate fewshot
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu   # Intel Arc GPU (NVIDIA면 CUDA용, 없으면 생략)
pip install -r requirements.txt
```

**Track B** (작은 모델이라 CPU 권장: `$env:DEVICE="cpu"`)
```powershell
python -m src.track_b.train --ticker AAPL --start 2015-01-01          # 베이스라인 → LSTM/GRU/Attn-LSTM
python -m src.track_b.hparam_search --csv data/AAPL.csv --arch GRU    # 보너스: 탐색
python -m src.track_b.train --csv data/AAPL.csv --config outputs/track_b/hparam/best_config.json --out outputs/track_b_tuned --reports reports/tuned
```

**Track A**
```powershell
python -m src.track_a.prepare_data --source cifar10 --classes cat dog deer horse bird
python -m src.track_a.train --strategy all --img_size 128 --epochs 15 --workers 0
python -m src.track_a.error_analysis extract --run outputs/track_a/partial
#   ↳ reports/error_analysis_table.csv 의 tag 열을 이미지를 직접 보고 채움 (Human-in-the-loop)
python -m src.track_a.error_analysis summarize --run outputs/track_a/partial

# 처방 (기준 partial에서 해당 항목만 변경, 224px는 메모리 제약으로 batch 16)
python -m src.track_a.train --strategy partial --img_size 224 --tag res224_only --epochs 15 --workers 0 --batch_size 16
python -m src.track_a.train --strategy partial --img_size 224 --label_smoothing 0.1 --tag res224 --epochs 15 --workers 0 --batch_size 16
python -m src.track_a.train --arch efficientnet_b0 --strategy partial --img_size 224 --tag effb0_224 --epochs 15 --workers 0 --batch_size 16
python -m src.track_a.train --strategy partial --aug strong --dropout 0.5 --weight_decay 1e-3 --tag aug_strong --img_size 128 --epochs 15 --workers 0 --batch_size 16
python -m src.track_a.error_analysis compare --before outputs/track_a/partial --after outputs/track_a/partial_effb0_224   # 처방마다 1회
```
일괄 실행: `run_all.ps1` (Windows) / `run_all.sh` — 태깅 단계에서 멈춤.

---

## 6. 설계 메모
- **Track A 입력 크기**: 32px 원본을 bilinear 보간으로 128/224px로 리사이즈(초해상도 아님, 정보량 동일). 224px는 ImageNet 사전학습 스케일에 맞추는 목적. 정규화는 Train/Validation/Test 모두 같은 ImageNet mean/std (fit 없음).
- **Freezing**: 동결 구간의 BatchNorm도 eval로 고정해 running 통계 갱신 방지. Fine-tuning 시 사전학습 층은 학습률 ×0.1(차등 학습률).
- **진단용 Train Loss**: 증강·Dropout 없는 eval 모드로 다시 계산해 Validation과 같은 조건에서 Gap 측정.
- **모델 선택**: Validation 기준만 사용, Test는 최종 비교용.
- **Track B 타깃**: 로그수익률을 학습하고 `ŷ(t) = y(t−1)·exp(r̂)`로 가격 복원 → 베이스라인과 같은 가격 공간·같은 Test 시점 비교. 딥러닝은 seed 3개 평균.
- **정규화 통계**(μ, σ)는 `leakage_checks.json`의 `2_scaler_mu_sd`에 기록.

## 7. 한계
- 오차 분석을 Test 셋으로 수행 → 처방 효과 수치는 낙관적일 수 있음
- Track A 단일 seed, Validation 100장(신뢰구간 ±8%p)으로 모델 선택 노이즈 존재
- 처방 1-c는 입력 크기와 Backbone을 함께 바꿈 (1-a와 비교해 기여를 분리: 입력 크기 +2.2%p, Backbone +3.4%p)
- Track B는 단변량 가격만 사용 → 다변량 피쳐, 방향 예측, walk-forward 검증이 다음 단계
