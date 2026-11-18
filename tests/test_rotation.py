"""Tests for the sector-rotation project.

All tests use synthetic prices, so they run offline in seconds.
They test the finance rules, not the data download.
"""

import numpy as np
import pandas as pd
import pytest

from src.regimes import classify_sector, compute_features
from src.rotation import apply_turnover_cap, target_weights
from src.backtest import performance_metrics


def synthetic_prices(n=400, seed=0):
    """400 days of synthetic prices for the 10 sectors + SPY + AGG."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB",
               "XLRE", "XLC", "SPY", "AGG"]
    # Random-walk prices, starting at 100, ~1% daily vol.
    steps = rng.normal(0.0005, 0.01, size=(n, len(tickers)))
    prices = 100 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_sell_off_is_vol_explosion_plus_negative_momentum():
    assert classify_sector(2.5, -0.10) == "sell-off"
    # High vol alone is not a sell-off.
    assert classify_sector(2.5, 0.10) != "sell-off"
    # Falling alone, in calm markets, is not a sell-off.
    assert classify_sector(1.0, -0.10) != "sell-off"


def test_growth_needs_positive_momentum_without_vol_explosion():
    assert classify_sector(0.9, 0.12) == "growth"
    assert classify_sector(2.5, 0.12) != "growth"  # too volatile
    assert classify_sector(0.9, 0.02) != "growth"  # momentum too weak


def test_missing_inputs_are_neutral_not_a_crash():
    assert classify_sector(float("nan"), -0.10) == "neutral"
    assert classify_sector(2.5, float("nan")) == "neutral"


def test_features_have_no_look_ahead():
    """vol_ratio on day t must be computable from data up to day t only."""
    prices = synthetic_prices()
    features = compute_features(prices)
    # The first 252 days have no 1-year vol, so vol_ratio is NaN there.
    assert features["vol_ratio"].iloc[:250].isna().all().all()
    assert features["vol_ratio"].iloc[300].notna().all()
    # Regime is one of the three known labels, nothing else.
    assert set(features["regime"].stack().unique()) <= {
        "sell-off", "growth", "neutral"}


def test_target_weights_are_top_3_equal_weight_and_sum_to_one():
    prices = synthetic_prices()
    features = compute_features(prices)
    date = prices.index[300]
    weights, chosen = target_weights(features, date, top_n=3)
    assert len(chosen) == 3
    assert abs(weights.sum() - 1.0) < 1e-9
    assert set(weights[weights > 0].index) == set(chosen)
    # No sell-off sector is ever chosen.
    for ticker in chosen:
        assert features["regime"].at[date, ticker] != "sell-off"


def test_turnover_cap_is_respected():
    # Full replacement: 100% XLK -> 50/50 XLF/XLE is 100% turnover
    # (0.5 * (|0-1| + |0.5-0| + |0.5-0|) = 1.0). Cap at 60%.
    current = pd.Series([1.0, 0.0, 0.0], index=["XLK", "XLF", "XLE"])
    target = pd.Series([0.0, 0.5, 0.5], index=["XLK", "XLF", "XLE"])
    capped, turnover = apply_turnover_cap(current, target, max_turnover=0.60)
    assert turnover <= 0.60 + 1e-9
    assert capped["XLK"] == pytest.approx(0.40)
    assert capped["XLF"] == pytest.approx(0.30)
    assert capped["XLE"] == pytest.approx(0.30)
    # A small change passes through untouched.
    small = pd.Series([0.35, 0.35, 0.30], index=["XLK", "XLF", "XLE"])
    capped2, t2 = apply_turnover_cap(capped, small, max_turnover=0.60)
    assert capped2.equals(small)


def test_performance_metrics_on_a_known_curve():
    # $1 growing 0.1% a day for 252 days: CAGR ~ 28.6%, no drawdown.
    equity = pd.Series(1.001 ** np.arange(253),
                       index=pd.bdate_range("2020-01-01", periods=253))
    m = performance_metrics(equity)
    assert m["cagr"] == pytest.approx(0.286, abs=0.01)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    assert m["final_value"] == pytest.approx(1.001 ** 252, rel=1e-6)
    # A curve that halves and never recovers has max drawdown -50%.
    equity2 = pd.Series([1.0, 0.5, 0.5, 0.5],
                        index=pd.bdate_range("2020-01-01", periods=4))
    assert performance_metrics(equity2)["max_drawdown"] == pytest.approx(-0.5)
