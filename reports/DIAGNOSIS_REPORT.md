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
- 두 모델 비교의 유의성: Track A는 같은 Test 이미지에 대한 **McNemar 정확 검정**, Track B는 **Diebold-Mariano 검정**. p < 0.05일 때만 "유의한 차이"로 판단한다.

---

## 2. Track A — Few-shot 전이 학습 (CIFAR-10 5클래스, 클래스당 Train 40 / Validation 20 / Test 100장, ResNet18, 128px)

### 2.1 결과 요약
| Run | 학습 파라미터 | Train Acc | Validation Acc | Test Acc | Gap(Loss) | 진단 |
|---|---:|---:|---:|---:|---:|---|
| scratch (직접 학습) | 11.2M | 0.930 | 0.420 | 0.436 | +1.391 | High Variance |
| linear_probe | 2.6K | 0.985 | 0.750 | 0.756 | +0.736 | High Variance |
| partial (layer4 + Head) | 8.4M | 1.000 | **0.820** | 0.750 | +0.516 | High Variance |
| full (차등 LR) | 11.2M | 0.990 | 0.780 | 0.714 | +0.561 | High Variance |
| partial_res224 (처방 1) | 8.4M | 1.000 | 0.820 | **0.778** | +0.398* | High Variance |
| partial_aug_strong (처방 2) | 8.4M | 1.000 | 0.800 | 0.752 | +0.563 | High Variance |

\* label smoothing 사용으로 Train Loss의 하한이 0보다 커서 Gap이 작아 보임 → 다른 Run과 Gap 직접 비교 불가.

**전략 간 McNemar 검정 (Test 500장)**

| 비교 | B만 맞춤 / A만 맞춤 | p-value | 결론 |
|---|---:|---:|---|
| scratch → linear_probe | 189 / 29 | < 0.001 | 사전학습 효과 **유의** |
| linear_probe → partial | 39 / 42 | 0.82 | 차이 없음 |
| partial → full | 32 / 50 | 0.060 | full이 나쁜 경향 (경계) |
| partial → partial_res224 | 40 / 26 | 0.11 | 개선 경향, 유의하지 않음 |
| partial → partial_aug_strong | 19 / 18 | 1.00 | 차이 없음 |

### 2.2 해석
- **직접 학습 vs 사전학습**: scratch는 Train 0.93 / Validation 0.42로 Gap이 가장 큰 High Variance. 11.2M 파라미터를 200장으로 학습하면 일반화가 불가능하다. 사전학습 특징을 쓰면 Test Acc가 0.436 → 0.756(+32%p, p<0.001)로 오른다. 데이터가 적을수록 사전학습의 가치가 결정적이라는 예상과 일치.
- **모든 전략이 High Variance**: Head만 학습하는 linear_probe(2,565개 파라미터)조차 Train 0.985까지 외운다. 512차원 특징 공간에서 200개 샘플은 선형 분리가 쉬우므로, 병목은 "모델 용량"이 아니라 "데이터 수"다.
- **학습 범위가 넓을수록 Test가 나빠지는 경향**: linear_probe(0.756) ≈ partial(0.750) > full(0.714). full은 사전학습 특징이 200장에 맞춰 왜곡되며(LP-FT 논문이 지적한 현상) Validation Loss도 partial보다 높다.
- **Validation 셋이 작은 문제**: partial은 Validation 0.82로 1위였지만 Test는 0.750으로 떨어졌다. Validation 100장의 95% 신뢰구간은 약 ±8%p라서, 선택 과정에서 Validation 노이즈에 운 좋게 맞은 모델이 뽑혔을 가능성이 있다.

### 2.3 Layer Freezing 결정
| 데이터 크기 | 도메인 유사도 높음 | 도메인 유사도 낮음 |
|---|---|---|
| 작음 (<50/클래스) | Linear Probing | Partial (마지막 stage) |
| 큼 | Partial → Full | Full (차등 LR) |

- **최종 선택: partial (layer4 + Head)**. 모델 선택은 규칙대로 Validation Acc 기준(0.82, 1위)으로 했다. Test에서 linear_probe와 통계적으로 같으므로(p=0.82), 실무에서는 학습 비용이 훨씬 작은 **linear_probe도 동등한 대안**이다.
- full은 선택하지 않는다: 데이터 40장/클래스에서는 전체 미세 조정의 이득보다 특징 왜곡·과적합 비용이 크다 (partial 대비 p=0.06).
- 근거: CIFAR 동물 사진은 ImageNet과 **내용은 유사**하지만 32px 업샘플이라 **저수준 통계가 다르다**. 그래서 마지막 stage만 적응시키는 partial이 합리적인 절충이다.

### 2.4 처방과 효과
**진단 → 처방**: 전 전략 High Variance → 정규화·데이터 측면 처방 필요. 동시에 오차 분석 태그 통계로 **어떤 처방을 우선할지** 정했다.

**오차 분석 (partial 오분류 125건, 사람이 직접 태깅)**

| 태그 | 건수 | 비율 | 평균 신뢰도 |
|---|---:|---:|---:|
| 모델 한계 | 57 | 45.6% | 0.67 |
| 클래스 유사성 | 35 | 28.0% | 0.61 |
| 데이터 품질 | 22 | 17.6% | 0.66 |
| 배경/잡음 | 11 | 8.8% | 0.66 |

- 오분류의 **74%가 모델 한계·클래스 유사성** → 특징 변별력 강화가 1순위. 데이터 품질·배경은 합쳐 26%로, 완벽히 해결해도 오차 감소 상한이 26%(ceiling analysis).
- 고신뢰 오답이 많다(평균 신뢰도 0.6~0.7) → 과신(overconfidence) 문제.

| 처방 | 근거 태그 | 변경 | Test Acc | 태그별 회복률 | 회귀 |
|---|---|---|---|---|---:|
| **처방 1** | 모델 한계 + 클래스 유사성 (74%) | 입력 128→224px (ImageNet 객체 스케일에 맞춤) + label smoothing 0.1 (과신 완화) | 0.750 → **0.778 (+2.8%p)** | 모델 한계 **38.6%**, 데이터 품질 27.3%, 배경 27.3%, 클래스 유사성 25.7% | 26건 |
| **처방 2** | 데이터 품질 + 배경/잡음 (26%) | ColorJitter·GaussianBlur·RandomErasing·강한 Crop + Dropout 0.5 + Weight decay 1e-3 | 0.750 → 0.752 (+0.2%p) | 클래스 유사성 17.1%, 모델 한계 15.8%, 데이터 품질 13.6%, 배경 9.1% | 18건 |

- **처방 1이 유효**: 겨냥한 "모델 한계"의 회복률이 38.6%로 가장 높아 처방 근거와 일치. 다만 McNemar p=0.11로 아직 통계적으로 유의하지 않다 → "개선 경향".
- **처방 2는 가설 기각**: 겨냥한 데이터 품질(13.6%)·배경(9.1%)의 회복률이 오히려 가장 낮다. 32px 이미지는 이미 정보가 적어서, 밝기·블러·가림 증강이 새로운 정보를 주지 못한 것으로 해석한다. 정규화를 강화해도 Train Acc 1.0·Gap 유지 → 이 데이터 규모에선 정규화만으로 Variance가 줄지 않는다.
- **한계**: (1) 처방 1은 해상도와 label smoothing을 동시에 바꿔 기여도 분리 불가 (ablation 필요). (2) 오차 분석을 Test 셋으로 수행했으므로 개선 수치는 낙관적으로 편향될 수 있다. (3) 재학습은 메모리 제약으로 배치 32→16. (4) 단일 seed.
- **다음 단계**: 해상도/label smoothing 분리 실험, 더 큰 Backbone(ResNet50, EfficientNet), LP-FT, Grad-CAM으로 "모델 한계" 사례가 배경을 보고 있는지 검증.

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
1. **진단은 기준점과 함께**: "Loss가 높다/Gap이 크다"는 기준 Loss·베이스라인·유의성 검정과 같이 봐야 의미가 있다. Track B의 "High Bias"는 기준 없이 보면 오진(모델 확대)으로 이어진다.
2. **처방은 원인 비율이 정한다**: Track A에서 태그 비율 74%를 겨냥한 처방 1만 효과가 있었고, 26%를 겨냥한 증강은 효과가 없었다. 오차 분석은 "무엇을 먼저 고칠지" 정하는 도구다.
3. **개선은 통계로 확인한다**: +2.8%p(p=0.11), +0.15%(p=0.46)처럼 수치상 개선이라도 유의하지 않으면 "경향"으로만 보고한다. 작은 데이터에서는 Validation 노이즈(±8%p)가 전략 간 차이보다 클 수 있다.
