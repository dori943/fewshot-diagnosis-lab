"""Track B 데이터: 로드 → (인과적) 피쳐 생성 → 시간순 분할 → Train-only 정규화 → 윈도우 생성 + 누수 검사.

설계
- 예측 대상: 다음 날 종가 y(t).
- 딥러닝 입력/타깃은 로그수익률 r(t) = log y(t) - log y(t-1) (비정상 가격 → 근사 정상 시계열).
  Train 이후 가격이 Train 범위를 벗어나도(주가 상승) 스케일 외삽 문제가 없다.
  예측 가격 ŷ(t) = y(t-1) · exp(r̂(t)) 로 복원해 베이스라인과 '같은 가격 공간'에서 비교한다.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def load_series(ticker: str, start: str, end: str | None, csv: str | None, cache_dir="data") -> pd.Series:
    if csv:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):  # yfinance ≥0.2.x: (Price, Ticker)
            df.columns = df.columns.get_level_values(0)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(cache_dir) / f"{ticker}.csv")  # 재현성 + 오프라인 대체용
    s = df["Close"].dropna().astype(float).sort_index()
    s = s[~s.index.duplicated()]
    years = (s.index[-1] - s.index[0]).days / 365.25
    assert years >= 3, f"최소 3년 일봉 필요 (현재 {years:.1f}년)"
    return s


def make_features(price: np.ndarray) -> np.ndarray:
    """로그수익률. r[t]는 y[t], y[t-1]만 사용 → 과거 정보만 사용하는 인과적 피쳐.
    (체크포인트 4) 분할 전에 전체 시계열에 대해 생성하되, 인과성을 아래 perturbation 테스트로 증명한다."""
    r = np.full(len(price), np.nan)
    r[1:] = np.diff(np.log(price))
    return r


def causality_test(feature_fn, price: np.ndarray, n_probe: int = 20, seed: int = 0) -> bool:
    """미래 값(t 이후)을 무작위로 바꿔도 t 시점 피쳐가 변하지 않으면 Look-ahead 없음."""
    rng = np.random.default_rng(seed)
    base = feature_fn(price)
    for t in rng.integers(10, len(price) - 10, n_probe):
        p2 = price.copy()
        p2[t + 1:] *= rng.uniform(0.5, 1.5, len(p2) - t - 1)
        if not np.allclose(feature_fn(p2)[: t + 1], base[: t + 1], equal_nan=True):
            return False
    return True


@dataclass
class Split:
    dates: pd.DatetimeIndex
    price: np.ndarray
    ret: np.ndarray
    i_val: int  # Validation 시작 인덱스
    i_test: int  # Test 시작 인덱스
    mu: float = 0.0
    sd: float = 1.0
    checks: dict = None

    def idx(self, part: str, lookback: int) -> np.ndarray:
        """해당 구간에 속한 '예측 대상 시점 t' 목록. 입력은 t-lookback..t-1 (이전 구간 꼬리 사용 가능 = 과거라서 허용)."""
        lo, hi = {"train": (lookback + 1, self.i_val), "validation": (self.i_val, self.i_test),
                  "test": (self.i_test, len(self.price))}[part]
        return np.arange(lo, hi)


def time_split(s: pd.Series, ratios=(0.7, 0.1, 0.2)) -> Split:
    price = s.values
    n = len(price)
    i_val, i_test = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    sp = Split(s.index, price, make_features(price), i_val, i_test)
    # (체크포인트 2) 정규화 통계는 Train 구간 수익률 r[1:i_val]만으로 Fit
    fit_r = sp.ret[1:i_val]
    sp.mu, sp.sd = float(fit_r.mean()), float(fit_r.std())
    sp.checks = {
        "1_chronological_split": bool(s.index.is_monotonic_increasing and
                                      s.index[i_val - 1] < s.index[i_val] and s.index[i_test - 1] < s.index[i_test]),
        "train_period": f"{s.index[0].date()} ~ {s.index[i_val - 1].date()} ({i_val})",
        "validation_period": f"{s.index[i_val].date()} ~ {s.index[i_test - 1].date()} ({i_test - i_val})",
        "test_period": f"{s.index[i_test].date()} ~ {s.index[-1].date()} ({n - i_test})",
        "2_scaler_fit_last_date": str(s.index[i_val - 1].date()),
        "2_scaler_fit_within_train": True,
        "2_scaler_mu_sd": [sp.mu, sp.sd],
        "4_feature_causality_test": causality_test(make_features, price),
    }
    return sp


def make_windows(sp: Split, part: str, lookback: int):
    """X: (N, lookback, 1) 스케일된 과거 수익률, y: 스케일된 r(t), t: 대상 인덱스."""
    t = sp.idx(part, lookback)
    z = (sp.ret - sp.mu) / sp.sd
    win = t[:, None] + np.arange(-lookback, 0)[None, :]  # t-lookback .. t-1
    # (체크포인트 3) 입력 윈도우의 모든 시점 < 대상 시점 (Look-ahead 방지)
    assert (win.max(1) < t).all() and (win.min() >= 1), "window leakage"
    if part == "train":
        assert t.max() < sp.i_val, "train target이 Validation 구간 침범"
    sp.checks[f"3_window_{part}_max_input_lt_target"] = True
    return z[win][..., None].astype(np.float32), z[t].astype(np.float32), t


def to_price(sp: Split, t: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    return sp.price[t - 1] * np.exp(z_pred * sp.sd + sp.mu)
