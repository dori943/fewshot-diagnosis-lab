# 성능 진단 리포트 (Diagnosis Report)

> 자동 생성 수치: [Track A 진단](track_a_diagnosis.md) · [Track B 진단](track_b_diagnosis.md) · [베이스라인 비교](baseline_comparison.md) · [튜닝 후 비교](tuned/baseline_comparison.md) · [오차 분석](error_analysis.md) · [누수 방지](LEAKAGE_PREVENTION.md)
> 이 문서는 곡선과 표를 보고 **사람이 판단·결정한 내용**을 기록한다.

## 1. 진단 방법

| 상태 | Train Loss | Validation Loss | 해결책 |
|---|---|---|---|
| High Bias (Underfitting) | 높음 | 높음 (비슷) | 모델 복잡도↑, Unfreeze 범위↑, 학습 시간↑, 정규화 완화 |
| High Variance (Overfitting) | 낮음 | 높음 (큰 격차) | 정규화(Dropout, L2), 데이터 증강, Early Stopping, 데이터↑ |

- **기준 Loss**: "아무것도 모르는 모델"의 Loss — 분류는 균등 예측 CE = ln(5) ≈ 1.609, 시계열은 Train 평균 수익률 예측 MSE = 1.0(스케일 공간).
- `rel_train = Train Loss / 기준 Loss` > 0.6 → High Bias 의심, `(Validation − Train) / Validation` > 0.25 → High Variance 의심 (`src/common/utils.py`). 임계값은 휴리스틱이므로 곡선과 함께 판단했다.
- 진단용 Train Loss는 증강/Dropout을 끈 eval 모드로 다시 계산했다 (Validation과 같은 조건).
- Track A는 Test 정확도 차이(%p)와 태그별 회복률로 비교한다. Validation 100장의 95% 신뢰구간이 약 ±8%p, Test 500장은 약 ±4%p이므로 수 %p 이하의 차이는 노이즈일 수 있음을 감안한다. Track B는 개선율과 함께 Diebold-Mariano 검정(p < 0.05일 때만 유의)을 보고한다.

---

## 2. Track A — Few-shot 전이 학습 (CIFAR-10 5클래스, 클래스당 Train 40 / Validation 20 / Test 100장)

### 2.1 결과 요약
| Run | Backbone / 입력 | 학습 파라미터 | Train Acc | Validation Acc | Test Acc | Gap(Loss) | 진단 |
|---|---|---:|---:|---:|---:|---:|---|
| scratch (직접 학습) | ResNet18 / 128px | 11.2M | 0.930 | 0.420 | 0.436 | +1.391 | High Variance |
| linear_probe | ResNet18 / 128px | 2.6K | 0.985 | 0.750 | 0.756 | +0.736 | High Variance |
| partial (layer4 + Head) | ResNet18 / 128px | 8.4M | 1.000 | 0.820 | 0.750 | +0.516 | High Variance |
| full (차등 LR) | ResNet18 / 128px | 11.2M | 0.990 | 0.780 | 0.714 | +0.561 | High Variance |
| partial_aug_strong (처방 2) | ResNet18 / 128px | 8.4M | 1.000 | 0.800 | 0.752 | +0.563 | High Variance |
| partial_res224_only (처방 1-a) | ResNet18 / 224px | 8.4M | 1.000 | 0.800 | 0.772 | +0.522 | High Variance |
| partial_res224 (처방 1-b) | ResNet18 / 224px + LS 0.1 | 8.4M | 1.000 | 0.820 | 0.778 | +0.398* | High Variance |
| **partial_effb0_224 (처방 1-c)** | EfficientNet-B0 / 224px | 3.2M | 1.000 | **0.870** | **0.806** | +0.432 | High Variance |

\* label smoothing(LS) 사용 시 Train Loss의 하한이 0보다 커서 Gap이 작아 보임 → 다른 Run과 Gap 직접 비교 불가.

### 2.2 해석
- **직접 학습 vs 사전학습**: scratch는 Train 0.93 / Validation 0.42로 Gap이 가장 큰 High Variance. 11.2M 파라미터를 200장으로 학습하면 일반화가 불가능하다. 사전학습 특징을 쓰면 Test Acc가 0.436 → 0.756(+32%p)로 오른다.
- **모든 Run이 High Variance**: Head만 학습하는 linear_probe(2,565개 파라미터)조차 Train 0.985까지 외운다. 512차원 특징 공간에서 200개 샘플은 선형 분리가 쉬우므로, 근본 병목은 "데이터 수"다.
- **학습 범위가 넓을수록 Test가 나빠지는 경향**: linear_probe(0.756) ≈ partial(0.750) > full(0.714). full은 사전학습 특징이 200장에 맞춰 왜곡되며(LP-FT 논문이 지적한 현상) Validation Loss도 partial보다 높다.
- **Validation 셋이 작은 문제**: Validation 100장의 95% 신뢰구간은 약 ±8%p. partial은 Validation 0.82였지만 Test 0.750으로 떨어졌다 → 선택 과정에서 Validation 노이즈에 맞은 모델이 뽑힐 수 있다.

### 2.3 Layer Freezing 결정과 최종 모델
| 데이터 크기 | 도메인 유사도 높음 | 도메인 유사도 낮음 |
|---|---|---|
| 작음 (<50/클래스) | Linear Probing | Partial (마지막 stage) |
| 큼 | Partial → Full | Full (차등 LR) |

- **Freezing 범위: Partial(마지막 stage + Head)** 채택. ResNet18 기준 Validation Acc 1위이고, Test에서 linear_probe와 거의 같다(0.750 vs 0.756) — 실무에서는 비용이 작은 linear_probe도 동등한 대안. full은 이 데이터 규모에서 특징 왜곡·과적합 비용이 크다(Test 0.714, partial 대비 −3.6%p).
- 근거: CIFAR 동물 사진은 ImageNet과 **내용은 유사**하지만 32px 업샘플이라 **저수준 통계가 다르다** → 마지막 stage만 적응시키는 partial이 합리적 절충.
- **최종 모델: EfficientNet-B0 partial, 224px** — Validation Acc 0.87로 전체 1위(규칙대로 Validation 기준 선택), Test 0.806. 같은 Partial 전략을 유지하면서 Backbone과 입력 스케일만 바꾼 결과다.

### 2.4 오차 분석 → 처방 → 효과
**오차 분석 (ResNet18 partial 오분류 125건, 사람이 직접 태깅)**

| 태그 | 건수 | 비율 | 평균 신뢰도 |
|---|---:|---:|---:|
| 모델 한계 | 57 | 45.6% | 0.67 |
| 클래스 유사성 | 35 | 28.0% | 0.61 |
| 데이터 품질 | 22 | 17.6% | 0.66 |
| 배경/잡음 | 11 | 8.8% | 0.66 |

- 오분류의 **74%가 모델 한계·클래스 유사성** → 특징 추출 능력 강화가 1순위. 데이터 품질·배경은 합쳐 26%로, 완벽히 해결해도 오차 감소 상한이 26%(ceiling analysis).
- "모델 한계"의 원인 가설 두 가지를 세우고 하나씩 검증했다.
  - (a) **입력 스케일 불일치**: 32px를 128px로 넣으면 ResNet18의 마지막 특징맵이 4×4로, ImageNet 사전학습 때(224px, 7×7)보다 객체가 작게 보여 사전학습 특징을 충분히 못 쓴다. → 224px로 확대 (새 픽셀 정보는 없고 스케일만 맞춤).
  - (b) **Backbone 특징의 질**: ResNet18(ImageNet top-1 약 70%)보다 특징이 좋은 EfficientNet-B0(약 78%)로 교체.
  - 보조: 고신뢰 오답이 많아(평균 신뢰도 0.6~0.7) 과신 완화용 label smoothing도 시험.

| 처방 | 변경 | Test Acc (vs 0.750) | 모델 한계 회복률 | 클래스 유사성 | 데이터 품질 | 배경/잡음 | 회귀 |
|---|---|---|---:|---:|---:|---:|---:|
| 1-a | 224px | 0.772 (+2.2%p) | 42.1% | 25.7% | 22.7% | 18.2% | 29 |
| 1-b | 224px + LS 0.1 | 0.778 (+2.8%p) | 38.6% | 25.7% | 27.3% | 27.3% | 26 |
| **1-c** | **224px + EfficientNet-B0** | **0.806 (+5.6%p)** | **52.6%** | **45.7%** | 36.4% | 36.4% | 30 |
| 2 | 강한 증강 + Dropout 0.5 + L2 1e-3 (128px) | 0.752 (+0.2%p) | 15.8% | 17.1% | 13.6% | 9.1% | 18 |

**개선 분해 (0.750 → 0.806, +5.6%p)**

| 단계 | Test Acc | 기여 |
|---|---:|---:|
| 기준 (ResNet18, 128px) | 0.750 | – |
| 입력 크기 128→224 (bilinear 보간) | 0.772 | +2.2%p |
| label smoothing 추가 | 0.778 | +0.6%p |
| Backbone ResNet18→EfficientNet-B0 (224px끼리) | 0.806 | +3.4%p |
| **합계 (입력 크기 + Backbone)** | **0.806** | **+5.6%p** |

- **원인에 맞춘 처방이 효과적**: 겨냥한 "모델 한계"(52.6%)·"클래스 유사성"(45.7%)의 회복률이 가장 높다. 진단 → 처방 → 검증이 일치.
- **"모델 한계"는 파라미터 수가 아니라 특징의 질 문제**: EfficientNet-B0의 학습 파라미터(3.2M)는 ResNet18 partial(8.4M)보다 적은데 더 좋다. 기여는 Backbone 교체(+3.4%p)가 입력 크기 확대(+2.2%p)보다 크다. 단계별 차이는 각각 Test 신뢰구간(±4%p) 안이지만, 합계 +5.6%p는 그 범위를 넘는다.
- **label smoothing은 효과 없음**(+0.6%p): 과신 완화는 정확도 개선으로 이어지지 않았다.
- **처방 2는 가설 기각**: 겨냥한 데이터 품질(13.6%)·배경(9.1%) 회복률이 오히려 가장 낮다. 32px 이미지는 이미 정보가 적어 밝기·블러·가림 증강이 새로운 정보를 주지 못한 것으로 해석. 정규화를 강화해도 Train Acc 1.0·Gap 유지.
- **남은 상태**: 최종 모델도 Train Acc 1.0의 High Variance. 데이터 200장이라는 근본 한계는 그대로다.
- **한계**: (1) 오차 분석을 Test 셋으로 수행해 개선 수치가 낙관적으로 편향될 수 있다. (2) 224px 재학습은 메모리 제약으로 배치 32→16. (3) 단일 seed. (4) 처방 1-c는 EfficientNet-B0를 128px로 학습한 실험이 없어 입력 크기×구조 상호작용은 분리하지 않았다.
- **다음 단계**: 여러 seed 반복, EfficientNet-B0 + LP-FT, Grad-CAM으로 남은 "모델 한계" 사례가 객체를 보는지 확인, Validation 셋 기반 오차 분석으로 재검증.

---

## 3. Track B — 시계열 (AAPL 일봉 2015-01-02 ~ 2026-11-13, Train/Validation/Test = 7/1/2)

- Test 구간: 2024-07-01 ~ 2026-11-13 (597일), 모든 모델이 동일 Test 시점에서 평가. 누수 자동 검사 전 항목 `true`.
- 딥러닝은 로그수익률로 학습 후 `ŷ(t) = y(t−1)·exp(r̂)`로 가격 복원, seed 3개 평균.

### 3.1 결과 요약
| Model | Train Loss | Validation Loss | 진단(자동) | RMSE | vs Naive RMSE 개선(%) | DM p-value |
|---|---:|---:|---|---:|---:|---:|
| Naive | – | – | – | 4.2958 | – | – |
| LinReg(lag5) (베이스라인 2위) | – | – | – | 4.3081 | −0.29 | – |
| LSTM | 0.9969 | 0.5096 | High Bias | 4.2909 | +0.114 | 0.382 |
| GRU | 0.9967 | 0.5119 | High Bias | 4.2983 | −0.058 | 0.911 |
| Attn-LSTM | 0.9979 | 0.5082 | High Bias | 4.2905 | +0.121 | 0.519 |
| LSTM (튜닝 설정) | 1.0039 | 0.5100 | High Bias | 4.2936 | +0.051 | 0.550 |
| GRU (튜닝 설정) | 1.0052 | 0.5093 | High Bias | 4.2916 | +0.098 | 0.390 |
| Attn-LSTM (튜닝 설정) | 1.0055 | 0.5084 | High Bias | 4.2893 | +0.151 | 0.461 |

- 베이스라인: Naive가 최강(RMSE 4.30). SMA·EMA는 창이 길수록/α가 작을수록 나빠짐(과거 평균은 추세를 늦게 따라감). LinReg(time)은 RMSE 61~63으로 실패 — 2015~2023 추세선을 2024~2026에 외삽할 수 없다.

### 3.2 해석
- **딥러닝은 Naive를 유의하게 이기지 못했다.** 개선율은 −0.06% ~ +0.15%이고 모든 DM p-value가 0.38 이상이다. "딥러닝이 X% 우수하다"고 주장할 수 없다.
- **이론과 일치**: 일별 주가는 랜덤워크에 가깝고, 그때 최적 예측은 `y(t−1)` = Naive다. 예측 vs 실제 그래프(확대)에서 빨간 예측선이 파란 실제선을 하루 늦게 따라가며 Naive 점선과 거의 겹친다 → 모델이 사실상 Naive를 학습했다.
- **자동 진단 "High Bias"의 재해석**: rel_train ≈ 1.0, 대부분 best epoch = 1. 모델이 "평균 수익률"만 예측한다는 뜻이지만, 원인은 용량 부족이 아니라 **Bayes error(예측 불가능한 잡음)** 다. 근거: 하이퍼파라미터(hidden 16~256, layers 1~3)를 바꿔도 Validation Loss가 변하지 않는다(3.3). 따라서 "모델 복잡도↑" 처방은 **채택하지 않는다**.
- **Validation Loss < Train Loss인 이유**: 과적합과 무관. Validation 기간(2023-04 ~ 2024-06)의 수익률 변동성이 Train 기간(2020년 코로나 급락 포함)보다 낮아, 같은 모델이라도 MSE가 작게 나온다. 실측: 일간 로그수익률 표준편차 Train 0.0187, Validation 0.0134 → 분산 비 (0.0134/0.0187)² ≈ **0.51**로, Validation Loss 0.51과 정확히 일치한다. 즉 모델은 Validation에서도 "평균만 예측"하고 있다.
- **Attention 효과**: Attn-LSTM이 수치상 근소하게 가장 좋지만 유의하지 않다. `attention_weights.png`에서 가중치 분포를 확인하면, 긴 lookback에서도 선별할 정보 자체가 적다는 해석과 연결된다.

### 3.3 하이퍼파라미터 탐색 (보너스)
- 방법: GRU 대상 Random Search 24회(Coarse) → 상위 3개 범위로 좁혀 12회(Fine). lr·weight decay는 로그 균등 샘플링. 선택 기준은 Validation Loss만 사용(Test 미사용).
- 결과: 36회의 Validation Loss가 **0.507 ~ 0.516 (표준편차 0.002)** — 하이퍼파라미터에 거의 둔감하다.
- 영향도(RandomForest): **lr 0.30 > lookback 0.28** > dropout 0.13 > layers 0.12 > hidden 0.09 > weight_decay 0.08. 차이 폭 자체가 작아 "경향" 수준으로만 해석.
- 최적 설정: lr 1.1e-4, hidden 23, layers 1, dropout 0.32, lookback 64 — **가장 작은 모델 쪽**이 선택됨. 신호가 적을수록 단순한 모델이 유리하다는 해석과 일치.
- 주의: Fine 단계 최고(0.508)가 Coarse 최고(0.507)를 넘지 못했고, 최적 lr이 탐색 하한(1e-4) 근처다. 성능 차이가 미미해 범위 확장의 실익은 낮다고 판단.
- 튜닝 설정으로 재학습해도 Test RMSE 개선은 +0.05~0.15%, 유의하지 않음 → **병목은 하이퍼파라미터가 아니라 입력 정보(단변량 가격)** 다.
- **다음 단계**: 다변량 입력(거래량, 지수, 금리·환율), 타깃 재정의(방향 분류, 변동성 예측), walk-forward 검증.

---

## 4. 공통 결론
1. **진단은 기준점과 함께**: "Loss가 높다/Gap이 크다"는 기준 Loss·베이스라인·노이즈 범위와 같이 봐야 의미가 있다. Track B의 "High Bias"는 기준 없이 보면 오진(모델 확대)으로 이어진다.
2. **처방은 원인 비율이 정한다**: Track A에서 태그 비율 74%(모델 한계·클래스 유사성)를 겨냥한 입력 크기·Backbone 처방은 +5.6%p로 가장 큰 개선을 냈고, 26%를 겨냥한 증강은 효과가 없었다. 오차 분석은 "무엇을 먼저 고칠지" 정하는 도구다.
3. **개선은 노이즈 범위와 비교하고 분해한다**: +2.2%p(Track A, Test 신뢰구간 ±4%p 이내), +0.15%(Track B, DM p=0.46)처럼 수치상 개선이라도 노이즈 범위 안이면 "경향"으로만 보고한다. 여러 변경을 한 번에 넣었다면 단계별로 분리해 기여도를 확인한다. 작은 데이터에서는 Validation 노이즈(±8%p)가 전략 간 차이보다 클 수 있다.
