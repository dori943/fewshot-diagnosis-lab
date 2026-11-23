# 베이스라인 비교 리포트 — AAPL (자동 생성)

- Test 구간: 2024-07-01 ~ 2026-11-13 (597) · 모든 모델 동일 Test 시점 597개에서 평가
- 베이스라인은 딥러닝 학습 전에 먼저 측정됨 (`outputs/track_b/baseline_results.csv`의 measured_at)
- 딥러닝 지표는 seed 3개 평균 (std는 `all_results.csv`). 설정: `{'lookback': 64, 'hidden': 23, 'layers': 1, 'dropout': 0.32359475578712504, 'lr': 0.00011393032658242415, 'weight_decay': 2.8942793492824353e-05, 'batch_size': 64, 'epochs': 100, 'patience': 12}`

## 1. 성능 비교표

| Model        |     MAE |    RMSE |   MAPE(%) | type     |
|:-------------|--------:|--------:|----------:|:---------|
| Naive        |  2.9713 |  4.2958 |    1.1863 | baseline |
| SMA(5)       |  4.8285 |  6.5415 |    1.9449 | baseline |
| SMA(10)      |  6.4568 |  8.2165 |    2.5826 | baseline |
| SMA(20)      |  8.6438 | 10.6589 |    3.4095 | baseline |
| EMA(0.1)     |  7.2371 |  9.1192 |    2.8601 | baseline |
| EMA(0.3)     |  4.5292 |  6.0133 |    1.8177 | baseline |
| EMA(0.5)     |  3.6098 |  5.0363 |    1.4506 | baseline |
| LinReg(time) | 52.7788 | 61.2817 |   19.4197 | baseline |
| LinReg(lag5) |  2.9755 |  4.3082 |    1.1895 | baseline |
| LSTM         |  2.961  |  4.2936 |    1.1828 | deep     |
| GRU          |  2.96   |  4.2916 |    1.1822 | deep     |
| Attn-LSTM    |  2.9576 |  4.2893 |    1.1813 | deep     |

## 2. 개선율 (양수 = 딥러닝이 우수) & Diebold-Mariano 검정

개선율 = (베이스라인 오차 − 딥러닝 오차) / 베이스라인 오차 × 100. DM p-value < 0.05 일 때만 '통계적으로 유의한 차이'.

| Deep model   | vs    |   MAE 개선(%) |   RMSE 개선(%) |   MAPE(%) 개선(%) |   DM stat |   p-value |
|:-------------|:------|------------:|-------------:|----------------:|----------:|----------:|
| LSTM         | Naive |      0.3465 |       0.0507 |          0.2914 |   -0.5976 |    0.5501 |
| GRU          | Naive |      0.3795 |       0.098  |          0.3484 |   -0.8594 |    0.3901 |
| Attn-LSTM    | Naive |      0.4589 |       0.1511 |          0.4236 |   -0.738  |    0.4605 |

![bar](../outputs/track_b/baseline_comparison.png)

![pred](../outputs/track_b/pred_vs_actual.png)
