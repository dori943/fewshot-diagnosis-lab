# 미션 핵심 개념 정리 — AI 전공자용

> 범위: 이 미션(Few-shot 전이 학습 + 시계열 예측 + 진단/오차 분석)에 **직접 쓰이는 것만** 다룬다.
> Part 1은 이미 아는 내용을 한 줄씩 확인하는 용도, Part 2부터가 이 미션에서 더 알아야 할 부분이다.

---

## Part 1. 이미 아는 것 — 한 줄 확인

| 개념 | 이 미션에서의 쓰임 |
|---|---|
| Train / Validation / Test | Validation으로 모든 선택(Early Stopping, 전략, 하이퍼파라미터), Test는 마지막에 한 번만 |
| Early Stopping | Validation Loss 최저 epoch의 가중치 복원 (`patience` 사용) |
| AdamW / Cosine LR | 분류는 AdamW + Cosine, 시계열은 Adam + gradient clipping |
| Dropout / L2(Weight Decay) | High Variance일 때 처방 |
| 데이터 증강 | Train에만 적용. Validation/Test는 resize와 정규화만 |
| CrossEntropy / MSE | 분류 / 회귀 Loss |
| MAE / RMSE / MAPE | RMSE는 큰 오차에 민감, MAPE는 스케일과 무관 |
| Grid / Random Search | 보너스 과제에서 Random 방식과 Coarse-to-Fine 사용 |
| Seed 고정 | 재현성. 단, seed 하나의 결과는 "한 번 표본을 뽑은 것"일 뿐이다 (§2.6) |

---

## Part 2. 더 알아야 할 것 (자세히)

### 2.1 편향/분산 진단을 "제대로" 하기

**(1) 기준점이 있어야 높다/낮다를 말할 수 있다**
"Train Loss가 높다"는 무엇과 비교해서인지가 있어야 한다. 실무에서는 Andrew Ng식 분해를 쓴다.

```
Avoidable bias = Train 오차 − Bayes 오차(≈사람 수준, 최선의 베이스라인)
Variance       = Validation 오차 − Train 오차
```

- 분류의 상한 기준: 균등하게 찍었을 때의 CE = ln(C). 5클래스면 1.609이다.
- 시계열의 기준: Naive 예측의 오차가 사실상 Bayes 오차의 대리값이다. 주가는 랜덤워크에 가까워서 Train Loss가 기준과 비슷해도 모델 용량이 부족해서가 아닐 수 있다. 이때 모델을 키우면 분산만 늘어난다.
- 32×32 CIFAR 이미지는 사람도 헷갈리는 샘플이 있다. Bayes 오차가 0이 아니라는 점을 리포트에 적어 두면 진단의 깊이가 달라진다.

**(2) Train Loss를 공정하게 재야 한다**
학습 중 기록되는 Train Loss에는 증강과 Dropout이 켜져 있다. 그래서 Validation Loss가 Train Loss보다 낮게 나오는 역전이 흔하다. 진단용 Train Loss는 **eval 모드에서 증강 없이 다시 계산**해야 Validation과 같은 조건이 된다. (📍`track_a/train.py`의 `train_clean` 로더)

**(3) Epoch 곡선과 데이터 크기 곡선은 다르다**
- Epoch 축 곡선은 학습이 진행되는 과정을 보여준다. 과적합이 시작되는 지점을 찾는 데 쓴다.
- 데이터 크기 축의 Learning curve(K = 5, 10, 20, 40장)는 **데이터를 늘리면 해결되는 문제인지**를 알려준다.
  - Validation 곡선이 아직 내려가는 중이면 데이터↑가 효과적이다(Variance 문제).
  - 두 곡선이 이미 수렴해 평평하면 데이터를 더 모아도 소용없고 모델/특징을 바꿔야 한다(Bias 문제).
- Few-shot 미션에서 "데이터↑"를 처방으로 쓰려면 데이터 크기 곡선이 가장 강한 근거다.

**(4) Data mismatch 축**
Train과 Test의 분포가 다르면 세 번째 오차 원인이 생긴다. 예를 들어 CIFAR Train과 Test는 같은 분포지만, 실서비스 사진은 다르다. Train 분포에서 따로 떼어 둔 별도 검증 셋으로 분산과 분포 불일치를 분리한다. 이 미션 범위 밖이지만 리포트의 한계 절에 언급할 가치가 있다.

**(5) Validation 셋이 작으면 진단 자체가 흔들린다** ⚠️ 중요
클래스당 20장 × 5클래스면 Validation은 100장이다. 정확도 80%의 95% 신뢰구간은

```
±1.96 · sqrt(0.8·0.2/100) ≈ ±7.8%p
```

이다. 즉 전략 간 차이가 3~5%p 정도라면 **노이즈일 수 있다**. 대응은 다음과 같다.
- Validation/Test를 가능한 한 늘린다 (Train만 50장 미만이면 된다).
- 여러 seed와 여러 데이터 샘플링으로 평균±표준편차를 보고한다.
- 두 분류기를 비교할 때는 같은 Test 셋 기준 **McNemar 검정**을 쓴다 (§2.6).

---

### 2.2 전이 학습 — Freezing 범위 결정의 디테일

**(1) 결정 기준은 두 가지 축이다**

| | 원본 도메인과 비슷함 | 원본 도메인과 다름 |
|---|---|---|
| **데이터 적음** | Linear Probing | Partial (마지막 stage) |
| **데이터 많음** | Partial → Full | Full (차등 LR) |

CIFAR 동물 사진은 ImageNet과 내용은 비슷하다. 하지만 32px를 업샘플한 이미지라 **저수준 통계(해상도, 블러)가 다르다**. 그래서 Linear Probing이 생각보다 약하고 Partial이 이길 가능성이 있다. 이렇게 결과를 예측해 두고 실험으로 확인하는 서술이 좋은 리포트다.

**(2) BatchNorm 함정**
`requires_grad=False`만 걸면 `model.train()` 상태에서 BN의 running mean/var는 계속 갱신된다. 160장짜리 미니배치 통계가 ImageNet 통계를 덮어쓰게 되어, "동결했는데 특징이 바뀌는" 현상이 생긴다. 동결 구간의 BN은 eval 모드로 고정해야 한다. (📍`set_train_mode`)
- Full FT에서 배치가 매우 작으면(<16) BN 통계가 불안정하다. 이때는 BN을 고정하거나 GroupNorm 계열 backbone을 고려한다.

**(3) 차등 학습률 (Discriminative LR)**
새로 붙인 Head는 랜덤 초기화라 큰 LR이 필요하다. 사전학습 층은 이미 좋은 곳에 있으므로 작은 LR(× 0.1 등)이 맞다. 같은 LR을 쓰면 초반의 큰 gradient가 사전학습 특징을 망가뜨린다.

**(4) LP-FT (Kumar et al., ICLR 2022)**
랜덤 Head로 바로 Full FT를 하면 초기 gradient가 특징을 왜곡해 분포 밖(OOD) 성능이 떨어진다. 해결책은 **Linear probe로 Head를 먼저 수렴시킨 뒤 Full FT**하는 것이다. 이 미션의 4가지 전략에 5번째로 추가하기 가장 좋은 실험이다.

**(5) 입력 전처리 일치**
사전학습 모델은 ImageNet의 mean/std 정규화와 학습 해상도(224)를 전제로 한다. 정규화를 빼먹으면 Linear probe 성능이 크게 떨어진다. 해상도를 낮추면(128) 속도는 빨라지지만 성능이 떨어질 수 있다. 이것도 비교해 볼 만한 실험 축이다.

**(6) 사전학습 vs 직접 학습 비교의 공정성**
Scratch는 보통 더 큰 LR, 더 긴 학습, 더 강한 증강이 필요하다. 같은 설정으로 비교하면 Scratch에게 불리하다. 리포트에 "동일 예산 비교"인지 "각자 튜닝한 비교"인지 명시한다.

---

### 2.3 오차 분석 — 방법론으로 다루기

**(1) Ceiling analysis: 태그 비율 = 개선 상한**
오분류의 10%가 "흐림"이면, 흐림을 완벽히 해결해도 오차는 최대 10%만 줄어든다. **비율이 큰 태그부터 처방**한다. 이것이 오차 분석이 우선순위 도구인 이유다.

**(2) 태깅 프로토콜**
- 태그별 **판정 기준을 먼저 문서화**한다. 예: "어두움 = 객체를 알아보기 어려울 정도로 어두움".
- 원인이 여러 개면 주 태그 1개 + 보조 태그로 기록한다. 통계는 주 태그 기준으로 내고, 보조 태그는 겹침 분석에 쓴다.
- 태깅할 때 모델의 예측 확률을 먼저 보면 편향(anchoring)이 생긴다. 이미지 → 태그 → 신뢰도 확인 순서를 권장한다.
- 가능하면 일부 샘플을 2명이 독립 태깅해 일치도(Cohen's κ)를 본다.

**(3) Test 셋으로 오차 분석을 하면 생기는 문제** ⚠️
미션은 Test의 오분류를 분석하라고 하지만, 그 결과로 증강을 고르고 **같은 Test로 개선을 측정하면 Test에 과적합**된다(간접 누수). 정석은 다음과 같다.
- 오차 분석은 Validation(또는 별도 holdout)에서 하고, 개선 효과는 Test에서 한 번만 확인한다.
- 미션 형식을 그대로 따른다면, 리포트에 "Test 기반 분석이라 개선 수치는 낙관적으로 편향될 수 있다"고 명시한다.

**(4) 태그별로 근거를 보강하는 도구**

| 태그 | 정량 근거 | 도구 |
|---|---|---|
| 데이터 품질 | 밝기 평균, Laplacian 분산(블러), 해상도 | 📍`error_analysis.py` 자동 힌트 |
| 배경/잡음 | 모델이 객체가 아닌 배경을 보는지 | **Grad-CAM** 히트맵 |
| 클래스 유사성 | 특정 클래스 쌍에 오류가 집중되는지 | 혼동행렬, top-2 margin |
| 라벨 오류 | 모델 확신이 높은데 라벨과 불일치 | **Confident Learning** (`cleanlab`) |
| 모델 한계 | 고신뢰 오답 비율, 과신 | **Calibration** (ECE, reliability diagram) |

**(5) 처방 효과는 태그 단위로 측정한다**
전체 정확도 +2%p만 보고하면 무엇이 효과였는지 알 수 없다.
- **태그별 회복률**: 이전에 틀린 이미지 중 이번에 맞춘 비율.
- **회귀(regression)**: 이전에 맞췄는데 이번에 틀린 이미지 수. 증강이 다른 곳을 망가뜨리지 않았는지 확인한다.
- 📍`error_analysis.py compare`가 둘 다 계산한다.

**(6) Calibration — "신뢰도" 열을 믿어도 되는가**
딥러닝 분류기는 대체로 과신한다(Guo et al., 2017). 신뢰도 0.9로 예측한 샘플들의 실제 정확도가 0.7이면 신뢰도 열의 의미가 약해진다.
- ECE = Σ (bin 비중) × |bin 정확도 − bin 평균 신뢰도|
- 보정: **Temperature scaling**. Validation으로 스칼라 T 하나를 학습해 `softmax(z/T)`로 바꾼다. 정확도는 그대로이고 신뢰도만 보정된다.
- 오차 분석표의 "고신뢰 오답"을 해석하기 전에 한 번 확인할 가치가 있다.

---

### 2.4 데이터 증강 설계 원칙

- **라벨 보존**: 증강 후에도 사람이 같은 라벨을 붙일 수 있어야 한다. 색이 클래스를 결정하는 데이터(꽃, 음식)에 hue를 강하게 바꾸면 라벨이 훼손된다.
- **태그 → 증강 매핑** (📍`build_transforms("strong")`):
  - 어두움/저대비 → `ColorJitter(brightness, contrast)`
  - 흐림 → `GaussianBlur`
  - 가림, 일부만 보임 → `RandomErasing`, 작은 scale의 `RandomResizedCrop`
  - 클래스 유사성 → 증강보다는 Fine-tuning 범위 확대, 고해상도 입력, Mixup/CutMix
- **강도는 하이퍼파라미터다**. 너무 강하면 Train 분포가 Test와 멀어져 오히려 Bias가 생긴다. Train Acc가 크게 떨어지면 증강이 과한 신호다.
- **증강은 Variance 처방이다**. High Bias 상태에서 증강을 더하면 악화된다. 진단 결과와 처방이 일치하는지 리포트에서 확인한다.
- **시계열 증강 금지 이유**: Jittering은 수익률 분포의 꼬리를, Time-warping은 거래일 간격과 자기상관 구조를 바꾼다. 금융 시계열의 통계적 성질(변동성 군집 등) 자체가 신호라서 증강이 곧 왜곡이다.

---

### 2.5 시계열 예측에서 더 알아야 할 것

**(1) Naive가 강한 이유**
주가는 근사적으로 랜덤워크 `y(t) = y(t−1) + ε`이다. 그러면 조건부 기댓값의 최선은 `y(t−1)`, 즉 Naive다. LSTM이 Naive와 비슷하게 나오는 것은 **실패가 아니라 이론과 일치하는 결과**다. 이 미션에서 가장 중요한 해석 포인트다.

**(2) 가격 예측 그래프의 착시 (lag illusion)**
가격 수준을 직접 예측하면 모델은 쉽게 "어제 값 ≈ 내일 값"을 배운다. 그래프는 거의 완벽하게 겹쳐 보이지만, 확대하면 **빨간 선이 파란 선을 하루 늦게 따라간다**. 확인 방법은 다음과 같다.
- 예측과 실제를 1일 시프트해 비교한다.
- **수익률 공간 지표**와 **방향 정확도**(오르내림 맞춘 비율, 50%가 기준선)를 함께 본다.

**(3) 가격 대신 수익률을 타깃으로 하는 이유**
- 가격은 비정상(non-stationary)이다. Train 이후 가격이 Train 범위를 벗어나면 MinMax 스케일링한 모델은 외삽을 해야 한다.
- 로그수익률은 근사적으로 정상(stationary)이다. 📍 이 레포는 수익률로 학습하고 `ŷ(t) = y(t−1)·exp(r̂)`로 복원해 가격 공간에서 베이스라인과 비교한다.

**(4) 놓치기 쉬운 누수 유형**

| 유형 | 예 |
|---|---|
| 전처리 누수 | 전체 기간으로 scaler fit |
| Look-ahead | `rolling(center=True)`, `bfill`, 미래 값을 쓰는 피쳐 |
| 타깃 누수 | 피쳐에 t 시점 값(종가) 자체가 포함됨 |
| 선택 누수 | Test 성능을 보고 모델이나 하이퍼파라미터를 고름 |
| 데이터 소스 | 수정주가는 나중에 소급 변경됨 → 다운로드 원본을 고정 저장 |
| 생존 편향 | 지금 살아있는 종목만 고르면 과거 성과가 부풀려짐 (다종목 확장 시) |

체크포인트 4("피쳐는 분할 전에 생성")가 안전한 조건은 **피쳐가 인과적일 때**뿐이다. 📍`causality_test`가 미래 값을 바꿔도 과거 피쳐가 변하지 않음을 자동으로 확인한다.

**(5) 검증 방식: 단일 분할 vs Walk-forward**
7/1/2 단일 분할은 Test 1개 구간의 시장 국면(상승장/하락장)에 결과가 좌우된다. **Walk-forward(rolling-origin) 검증**은 학습 구간을 앞으로 밀면서 여러 Test 구간에서 평가한다. 실무 표준이므로 확장 실험으로 가치가 크다.

**(6) 지표 선택 주의**
- MAPE는 값이 0 근처면 폭발하고, 과소예측과 과대예측에 비대칭으로 반응한다. 주가처럼 양수가 큰 값이면 괜찮다.
- 스케일이 다른 여러 종목을 비교하려면 **MASE**(Naive 대비 MAE 비율)가 표준이다. MASE < 1이면 Naive보다 낫다는 뜻이라, 이 미션의 개선율과 직접 연결된다.

**(7) "몇 % 우수"를 말하려면 유의성 검정이 필요하다**
개선율 +0.5%는 우연일 수 있다. **Diebold-Mariano 검정**은 두 예측의 손실 차이 시계열 `d(t) = e1(t)² − e2(t)²`의 평균이 0인지 검정한다(자기상관 보정 포함). p ≥ 0.05면 "우월하다고 말할 수 없다"고 결론 낸다. (📍`baselines.py::diebold_mariano`)

---

### 2.6 비교 실험의 통계적 엄밀성

| 상황 | 방법 |
|---|---|
| seed 변동 | 최소 3~5 seed로 평균±표준편차를 보고한다. 차이가 표준편차보다 작으면 결론을 유보한다 |
| Few-shot 샘플링 변동 | 어떤 40장을 뽑았느냐에 따라 결과가 크게 다르다. 데이터 샘플링 seed도 바꿔서 반복한다 |
| 분류기 2개 비교 (같은 Test) | **McNemar 검정**: 한쪽만 맞춘 샘플 수 b, c로 χ² = (\|b−c\|−1)²/(b+c) |
| 예측기 2개 비교 (시계열) | Diebold-Mariano 검정 |
| 여러 설정 동시 비교 | 다중 비교 문제가 있다. 가장 좋은 것 하나를 Validation으로 고른 뒤 Test로 확인한다 |

McNemar 예시 코드 (Track A 전략 비교에 바로 쓸 수 있다):
```python
from statsmodels.stats.contingency_tables import mcnemar
a = pd.read_csv("outputs/track_a/partial/predictions_test.csv").correct
b = pd.read_csv("outputs/track_a/full/predictions_test.csv").correct
table = [[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]]
print(mcnemar(table, exact=True).pvalue)
```

---

### 2.7 하이퍼파라미터 탐색에서 추가로 알아야 할 것

- **Random이 Grid보다 나은 이유** (Bergstra & Bengio 2012): 성능에 중요한 파라미터는 소수다. Grid는 9번을 시도해도 중요한 축에서 3개 값만 보지만, Random은 9개 값을 본다.
- **로그 스케일**: lr, weight decay는 `10^U(a,b)`로 샘플링한다. 선형으로 샘플링하면 대부분이 큰 값 쪽에 몰린다.
- **Coarse-to-Fine의 함정**: 최적점이 탐색 범위 경계에 있으면 범위를 **넓혀야** 한다(좁히면 안 된다). 영향도 그래프에서 최적점이 x축 끝에 붙어 있는지 확인한다.
- **seed 노이즈 vs 하이퍼파라미터 효과**: trial마다 seed 노이즈가 섞인다. 상위 후보 몇 개는 여러 seed로 다시 돌려 확인한다.
- **영향도 분석은 trial 수가 적으면 불안정하다**. RandomForest 중요도를 30개 trial로 계산하면 순위가 쉽게 바뀐다. 결론은 "경향" 수준으로 적는다.
- **Validation 과적합**: trial을 많이 돌릴수록 Validation에 운 좋게 맞는 설정을 고를 확률이 커진다. 최종 Test 1회 평가가 그래서 중요하다.
- 다음 단계: Optuna(TPE), Hyperband/ASHA(나쁜 trial을 조기 중단).

---

### 2.8 CNN vs RNN — 구조적 차이 (과제 목표 3)

| | CNN | RNN (LSTM/GRU) |
|---|---|---|
| 귀납적 편향 | 지역성, 가중치 공유, 평행이동 등변성 | 순서, 시점 간 가중치 공유, 마르코프적 상태 |
| 정보 흐름 | 공간 계층 (엣지 → 부분 → 객체) | 시간 순차 (은닉 상태로 과거 요약) |
| 장거리 의존 | 깊이로 수용 영역을 확장 | 게이트로 기울기 소실을 완화하지만 길면 약함 |
| 병렬화 | 높음 | 낮음 (t는 t−1에 의존) |
| 이 미션 | 사전학습 특징의 전이가 핵심 | 사전학습 없이 작은 모델, 베이스라인 대비 검증이 핵심 |

**LSTM 게이트 요약**
- f, i, o = σ(W[h_{t−1}, x_t] + b)는 각각 망각·입력·출력 게이트다.
- 셀 상태는 c_t = f⊙c_{t−1} + i⊙tanh(·)로 갱신된다. 덧셈 경로라서 기울기가 잘 보존된다.
- GRU는 게이트를 update/reset 2개로 줄여 파라미터가 약 25% 적다. 데이터가 적을 때 유리한 경우가 많다.

**Attention (보너스) 해석 주의**
Additive attention은 과거 각 시점에 가중치 α_t를 준다. 하지만 **attention 가중치가 곧 설명은 아니다**(Jain & Wallace, 2019). 주가 데이터에서는 최근 시점에 가중치가 몰리는 경향이 있는데, 이는 Naive와 같은 전략을 학습했다는 신호일 수 있다.

---

## Part 3. 과제 목표 셀프 체크 (한 줄 답)

| 과제 목표 | 한 줄로 설명할 수 있어야 하는 것 |
|---|---|
| 편향/분산 진단 | 기준 Loss 대비 Train Loss로 Bias를, Train–Validation 격차로 Variance를 판단하고, 작은 Validation의 신뢰구간을 고려한다 |
| Freezing 범위 결정 | 데이터 크기 × 도메인 유사도로 정하고, BN 고정과 차등 LR까지 포함해야 "동결"이 완성된다 |
| CNN vs RNN | 공간적 지역성 가정 vs 시간적 순서 가정. 데이터 구조에 맞는 귀납적 편향을 고른다 |
| 오차 분석 → 증강 | 태그 비율이 개선 상한이고, 태그별 회복률과 회귀로 처방 효과를 검증한다 |
| 베이스라인 대비 정량화 | 동일 Test 구간에서 개선율을 내고, DM/McNemar 검정으로 유의성을 확인한다 |
| 시계열 Leakage | 시간순 분할, Train-only fit, 인과적 피쳐/윈도우, Test 기반 선택 금지 |

## 참고 문헌 (미션 관련만)
- Ng, *Machine Learning Yearning* — 편향/분산, 오차 분석, ceiling analysis
- Kumar et al., *Fine-Tuning can Distort Pretrained Features* (LP-FT), ICLR 2022
- Yosinski et al., *How transferable are features in deep neural networks?*, NeurIPS 2014
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017
- Northcutt et al., *Confident Learning*, JAIR 2021
- Selvaraju et al., *Grad-CAM*, ICCV 2017
- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 2012
- Diebold & Mariano, *Comparing Predictive Accuracy*, JBES 1995
- Hyndman & Koehler, *Another look at measures of forecast accuracy* (MASE), IJF 2006
- Jain & Wallace, *Attention is not Explanation*, NAACL 2019
