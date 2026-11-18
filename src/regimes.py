"""Regime detection: is a sector in a sell-off, in growth, or neutral?

The finance, in one paragraph
------------------------------
Volatility by itself does not tell you to sell. A sector can be
volatile and rising. A price fall by itself does not tell you to sell
either — a 5% dip in a calm market is normal noise. The signal is the
*combination*:

    sell-off  = volatility has exploded AND momentum has turned negative
    growth    = momentum is positive AND volatility has not exploded
    neutral   = everything else

Definitions (all computed from prices up to and including today, so
there is no look-ahead bias):

    sigma_20  = annualised std. dev. of daily returns, last 20 days
                = std(r, 20) * sqrt(252)
    sigma_1y  = the same, over the last 252 days (1 trading year)
    vol_ratio = sigma_20 / sigma_1y
                > 1 means "more volatile than its own normal level"
    mom_6m    = price today / price 126 trading days ago - 1
    mom_3m    = price today / price 63 trading days ago - 1
    rel_str   = mom_6m(sector) - mom_6m(SPY)
                > 0 means "beating the market over 6 months"

Regime rules (the thresholds are explicit and deliberately simple —
they are a hypothesis to test, not a tuned black box):

    SELL-OFF  if vol_ratio > 2.0 AND mom_6m < 0
              Volatility has at least doubled vs its 1-year average
              *and* the sector is down over 6 months.
    GROWTH    if vol_ratio < 1.2 AND mom_6m > 0.05
              Rising more than 5% over 6 months without a volatility
              explosion.
    NEUTRAL   otherwise.
"""

import numpy as np
import pandas as pd

from . import SECTORS

SELL_OFF_VOL_RATIO = 2.0
SELL_OFF_MOMENTUM = 0.0
GROWTH_VOL_RATIO = 1.2
GROWTH_MOMENTUM = 0.05


def classify_sector(vol_ratio, mom_6m):
    """Classify one sector on one day. Pure function — easy to test."""
    if vol_ratio is None or mom_6m is None or pd.isna(vol_ratio) or pd.isna(mom_6m):
        return "neutral"
    if vol_ratio > SELL_OFF_VOL_RATIO and mom_6m < SELL_OFF_MOMENTUM:
        return "sell-off"
    if vol_ratio < GROWTH_VOL_RATIO and mom_6m > GROWTH_MOMENTUM:
        return "growth"
    return "neutral"


def compute_features(prices):
    """Per-sector daily features. Returns a dict of DataFrames.

    Each frame is indexed by date, columns are sector tickers:
      vol_ratio, mom_6m, mom_3m, rel_strength, regime
    """
    sectors = [t for t in SECTORS if t in prices.columns]
    if "SPY" not in prices.columns:
        raise ValueError("prices must include SPY for relative strength")

    returns = prices.pct_change()
    vol_20 = returns.rolling(20).std() * np.sqrt(252)
    vol_1y = returns.rolling(252).std() * np.sqrt(252)
    vol_ratio = vol_20 / vol_1y

    mom_6m = prices.pct_change(126)
    mom_3m = prices.pct_change(63)
    # Relative strength: sector 6-month return minus SPY 6-month return.
    rel_strength = mom_6m[sectors].sub(mom_6m["SPY"], axis=0)

    regime = pd.DataFrame(index=prices.index, columns=sectors, dtype=object)
    for ticker in sectors:
        regime[ticker] = [
            classify_sector(v, m)
            for v, m in zip(vol_ratio[ticker], mom_6m[ticker])
        ]

    return {
        "vol_ratio": vol_ratio[sectors],
        "mom_6m": mom_6m[sectors],
        "mom_3m": mom_3m[sectors],
        "rel_strength": rel_strength,
        "regime": regime,
    }


def monthly_regime_table(features, prices):
    """One row per month-end: each sector's regime and vol_ratio then.

    This is the audit table — it shows, in plain text, when a sector
    was flagged as a sell-off and what the rotation did about it.
    """
    regime = features["regime"]
    vol_ratio = features["vol_ratio"]
    # Last trading day of each month (see rotation.py for why this
    # is not resample("ME").last().index).
    month_ends = [g.index[-1] for _, g in regime.groupby(pd.Grouper(freq="ME"))]
    rows = []
    for date in month_ends:
        row = {"date": date.date().isoformat()}
        for ticker in regime.columns:
            row[f"{ticker}_regime"] = regime.at[date, ticker]
            row[f"{ticker}_vol_ratio"] = (
                round(float(vol_ratio.at[date, ticker]), 2)
                if pd.notna(vol_ratio.at[date, ticker]) else None
            )
        row["n_sell_off"] = sum(
            regime.at[date, t] == "sell-off" for t in regime.columns
        )
        row["n_growth"] = sum(
            regime.at[date, t] == "growth" for t in regime.columns
        )
        rows.append(row)
    return pd.DataFrame(rows)


def explain_month(date, features, prices, top_sectors, sold_sectors):
    """Plain-text narrator for one rebalance — no API key, no LLM call.

    It just cites the numbers that drove the decision, in sentences.
    That is the honest version of the 'LLM narrator' idea: the numbers
    are the evidence, the text is a template around them.
    """
    lines = [f"Rebalance {pd.Timestamp(date).date().isoformat()}:"]
    for ticker in sold_sectors:
        vr = features["vol_ratio"].at[date, ticker]
        mom = features["mom_6m"].at[date, ticker]
        regime = features["regime"].at[date, ticker]
        if regime == "sell-off":
            reason = "sell-off regime."
        else:
            reason = "dropped out of the top 3 on momentum / relative strength."
        lines.append(
            f"  - Sold {ticker} ({SECTORS[ticker]}): vol_ratio "
            f"{vr:.2f}x its 1-year average, 6-month return {mom:+.1%} — "
            f"{reason}"
        )
    for ticker in top_sectors:
        vr = features["vol_ratio"].at[date, ticker]
        mom = features["mom_6m"].at[date, ticker]
        rs = features["rel_strength"].at[date, ticker]
        regime = features["regime"].at[date, ticker]
        label = "growth regime." if regime == "growth" else "top-ranked on momentum and relative strength."
        rs_text = f"beating SPY by {rs:+.1%}" if rs >= 0 else f"trailing SPY by {abs(rs):.1%}"
        lines.append(
            f"  - Holding {ticker} ({SECTORS[ticker]}): 6-month return "
            f"{mom:+.1%}, {rs_text}, vol_ratio {vr:.2f}x — "
            f"{label}"
        )
    if not sold_sectors and not top_sectors:
        lines.append("  - No change: no sector met the sell-off or growth rules.")
    return "\n".join(lines)
