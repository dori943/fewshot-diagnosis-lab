"""베이스라인 모델 (딥러닝보다 먼저 구현/측정). 모두 시점 t 예측에 y(t-1) 이전 값만 사용한다."""
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def metrics(y, yhat) -> dict:
    e = y - yhat
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAPE(%)": float((np.abs(e) / np.abs(y)).mean() * 100)}


def naive(price, t):
    return price[t - 1]  # ŷ(t) = y(t-1)


def sma(price, t, n):
    c = np.concatenate([[0.0], np.cumsum(price)])
    return (c[t] - c[t - n]) / n  # ŷ(t) = mean(y[t-n .. t-1])


def ema(price, t, alpha):
    e = np.empty(len(price))
    e[0] = e[1] = price[0]
    for i in range(2, len(price)):  # ŷ(t) = α·y(t-1) + (1-α)·ŷ(t-1)
        e[i] = alpha * price[i - 1] + (1 - alpha) * e[i - 1]
    return e[t]


def linreg_time(price, t, train_t):
    m = LinearRegression().fit(train_t[:, None], price[train_t])  # ŷ(t) = a·t + b  (Train에서만 Fit)
    return m.predict(t[:, None])


def linreg_lag(price, t, train_t, k=5):
    lag = lambda tt: np.stack([price[tt - i] for i in range(1, k + 1)], 1)  # y(t-1..t-k)
    m = LinearRegression().fit(lag(train_t), price[train_t])
    return m.predict(lag(t))


def run_all(price, test_t, train_t) -> dict:
    """dict[name] = 예측 배열 (test_t 시점)"""
    P = {"Naive": naive(price, test_t)}
    for n in (5, 10, 20):
        P[f"SMA({n})"] = sma(price, test_t, n)
    for a in (0.1, 0.3, 0.5):
        P[f"EMA({a})"] = ema(price, test_t, a)
    P["LinReg(time)"] = linreg_time(price, test_t, train_t)
    P["LinReg(lag5)"] = linreg_lag(price, test_t, train_t[train_t >= 5])
    return P


def diebold_mariano(y, p1, p2, h=1):
    """DM 검정(제곱오차). H0: 두 예측의 정확도 동일. stat<0 이면 p1이 더 정확."""
    d = (y - p1) ** 2 - (y - p2) ** 2
    n = len(d)
    gamma = [np.cov(d[k:], d[:n - k])[0, 1] if k else d.var() for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = d.mean() / np.sqrt(var)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))
