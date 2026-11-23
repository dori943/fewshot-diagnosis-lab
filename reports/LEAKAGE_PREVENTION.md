# 누수 방지 근거 (Data Leakage Prevention)

> 실제 기간/통계값은 실행 후 `outputs/track_b/leakage_checks.json`에 자동 기록된다. 아래 각 항목은 코드 위치와 **자동 검사**로 근거를 남긴다.

## 0. 데이터 흐름

```
원본 종가 y(0..N-1)
  └─ (분할 전) 인과적 피쳐 r(t)=log y(t) − log y(t−1)        ← 체크포인트 4
       └─ 시간순 분할  Train 70% | Validation 10% | Test 20%   ← 체크포인트 1
            └─ μ, σ = Train 구간 r 로만 계산 (fit)             ← 체크포인트 2
                 └─ 모든 구간 z=(r−μ)/σ (transform만)
                      └─ 윈도우: 입력 z(t−L..t−1) → 타깃 z(t)   ← 체크포인트 3
```

## 1. 시간순 분할 (shuffle 금지)
- 코드: `src/track_b/data.py::time_split` — 인덱스 `i_val = int(0.7N)`, `i_test = int(0.8N)`로 자른다. 무작위 분할 함수(`train_test_split` 등)를 쓰지 않는다.
- 검사: `1_chronological_split` = 날짜 단조 증가 & 각 경계에서 앞 구간 마지막 날 < 뒤 구간 첫날.
- 참고: 학습 루프의 `DataLoader(shuffle=True)`는 **Train 구간 내부의 윈도우 순서만** 섞는다. 각 윈도우는 이미 "과거 → 다음 날" 쌍으로 완결되어 있고 Validation/Test 샘플이 섞이지 않으므로 누수가 아니다 (SGD 수렴 안정화 목적).

## 2. 전처리 Fit 구간 제한
- 코드: `fit_r = ret[1:i_val]` → `mu, sd` 계산. Validation/Test는 같은 `mu, sd`로 변환만 한다(`make_windows`).
- 검사: `2_scaler_fit_last_date`가 Train 마지막 날과 같음을 기록.
- 왜 중요한가: 전체 기간으로 MinMax/Standard 스케일링하면 Test 구간의 최대·최소(미래 가격 수준)가 Train 입력에 스며든다. 특히 우상향 주가에서 MinMax 전체 Fit은 "미래에 가격이 얼마까지 오른다"는 정보를 주는 것과 같다.
- 베이스라인의 선형 회귀(`LinReg(time)`, `LinReg(lag5)`)도 **Train 시점만으로 Fit** 한다 (`baselines.py`).

## 3. 윈도우 생성 규칙 (Look-ahead Bias 방지)
- 코드: `win = t[:,None] + arange(-L, 0)` → 입력은 t−L … t−1, 타깃은 t.
- 검사: `assert (win.max(1) < t).all()` (모든 구간), Train 타깃 `t.max() < i_val`.
- Validation/Test 첫 윈도우는 직전 구간의 꼬리(과거)를 입력으로 쓴다. 이는 "실제 운영 시점에 이미 관측된 값"이므로 허용되며, 이렇게 해야 모든 모델이 **동일한 Test 시점**에서 평가된다.
- 베이스라인도 동일: Naive `y(t−1)`, SMA `mean(y[t−n..t−1])`, EMA 재귀식은 `y(t−1)`까지만 사용.

## 4. 피쳐 생성 시점
- 수익률/지연값/이동평균 같은 피쳐는 **분할 전에 전체 시계열에서 생성**한다. 분할 후 생성하면 각 구간 앞부분이 NaN이 되어 경계 시점 샘플이 사라지고, 구간마다 계산 기준이 달라진다.
- 단, 이것이 안전하려면 피쳐가 **인과적(causal)** 이어야 한다 — 시점 t의 피쳐는 t 이전(포함) 값만 사용.
  - ✅ `rolling(n).mean()` 후 `shift(1)` / `diff` / 과거 lag
  - ❌ `center=True` rolling, 전체 기간 z-score, `bfill`, 미래 수익률로 만든 라벨을 피쳐로 사용
- 검사: `causality_test` — 무작위 시점 t 이후의 가격을 임의로 바꿔도 `[0..t]` 피쳐가 변하지 않음을 20회 확인 (`4_feature_causality_test: true`).

## 5. 그 외 주의한 점
| 위험 | 대응 |
|---|---|
| Validation으로 Early Stopping/하이퍼파라미터 선택 후 Test 반복 확인 | 선택은 Validation Loss만 사용, Test는 최종 1회 평가 (`hparam_search.py`는 Test를 읽지 않음) |
| yfinance `auto_adjust` 조정가의 사후 수정 | 다운로드 원본을 `data/<TICKER>.csv`로 고정 저장해 재현 |
| 시계열 증강 | Jittering/Time-warping 등은 금융 시계열의 자기상관·변동성 구조를 훼손하므로 **미적용** |
| Track A 중복 | Train/Validation은 CIFAR train split, Test는 CIFAR test split에서 추출 → 동일 이미지 중복 없음 |
