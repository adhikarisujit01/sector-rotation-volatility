"""Backtest: rotation vs S&P 500 vs 60/40, out-of-sample.

Out-of-sample design
---------------------
The first 252 trading days (~1 year) are warm-up only: they are used
to estimate 1-year volatility and 6-month momentum, but no return
from that period is counted in the reported results. The backtest
starts the day after the warm-up ends. Nothing in the reported
numbers was used to choose the rules — the rules (vol_ratio > 2,
top 3, 60% turnover cap) were fixed in advance, in the README.

Benchmarks
----------
* SPY: buy and hold the S&P 500.
* 60/40: 60% SPY / 40% AGG, rebalanced monthly to target weights.
  The classic balanced portfolio — the hurdle a rotation strategy
  has to beat to be worth the trading.

Metrics (annualised, 252 trading days/year, risk-free rate = 0):
    CAGR        = (final / initial) ** (252 / n_days) - 1
    Volatility  = std(daily returns) * sqrt(252)
    Sharpe      = mean(daily returns) / std(daily returns) * sqrt(252)
    Max DD      = worst peak-to-trough fall of the equity curve
    Turnover    = mean monthly turnover, annualised (x12)
    Total cost  = sum of all rebalance costs, as % of initial capital
"""

import numpy as np
import pandas as pd

from . import RISK_FREE_RATE, SECTORS, WARMUP_DAYS


def performance_metrics(equity):
    """Metrics for one equity curve (Series, indexed by date)."""
    returns = equity.pct_change().dropna()
    n_days = len(equity)
    years = n_days / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = returns.std() * np.sqrt(252)
    sharpe = (
        (returns.mean() - RISK_FREE_RATE / 252) / returns.std() * np.sqrt(252)
        if returns.std() > 0 else np.nan
    )
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return {
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "final_value": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
    }


def run_backtest(prices, weights, trades):
    """Run the rotation, SPY and 60/40 from the end of warm-up.

    Costs are applied on the rebalance date: the equity curve is
    multiplied by (1 - cost) on that day, before the next day's
    returns. Weights apply from the day after the rebalance.
    """
    sectors = list(SECTORS)
    start_idx = min(WARMUP_DAYS, len(prices) - 1)
    start_date = prices.index[start_idx]
    px = prices.loc[start_date:].copy()
    w = weights.loc[start_date:].copy()
    sector_returns = px[sectors].pct_change().fillna(0.0)

    # Rotation equity, with costs deducted at each rebalance.
    cost_by_date = {t["date"]: t["cost"] for t in trades}
    equity = [1.0]
    dates = list(px.index)
    for i, date in enumerate(dates):
        if i == 0:
            continue
        day_return = float((w.iloc[i - 1] * sector_returns.iloc[i]).sum())
        value = equity[-1] * (1 + day_return)
        if date in cost_by_date:
            value *= (1 - cost_by_date[date])
        equity.append(value)
    rotation = pd.Series(equity, index=dates)

    # Benchmarks, same period, no costs (buy-and-hold SPY has none;
    # 60/40 rebalancing costs are small and are stated as omitted).
    spy_returns = px["SPY"].pct_change().fillna(0.0)
    spy = (1 + spy_returns).cumprod()
    spy.iloc[0] = 1.0

    agg_returns = px["AGG"].pct_change().fillna(0.0)
    balanced_returns = 0.60 * spy_returns + 0.40 * agg_returns
    balanced = (1 + balanced_returns).cumprod()
    balanced.iloc[0] = 1.0

    # Align: first value 1.0 for all three.
    rotation.iloc[0] = 1.0

    metrics = {
        "rotation": performance_metrics(rotation),
        "spy": performance_metrics(spy),
        "60/40": performance_metrics(balanced),
    }
    # Turnover / cost summary for the rotation only.
    if trades:
        turnovers = [t["turnover"] for t in trades]
        metrics["rotation"]["avg_monthly_turnover"] = float(np.mean(turnovers))
        metrics["rotation"]["annualised_turnover"] = float(np.mean(turnovers) * 12)
        metrics["rotation"]["total_cost_pct"] = float(sum(t["cost"] for t in trades) * 100)
        metrics["rotation"]["n_rebalances"] = len(trades)

    curves = pd.DataFrame({
        "rotation": rotation, "spy": spy, "60/40": balanced,
    })
    return curves, metrics, start_date
