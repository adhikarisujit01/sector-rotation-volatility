"""Rotation engine: monthly, top-N sectors, with costs and a turnover cap.

Ranking (computed only from data available on the rebalance date):
    score = rank(mom_6m) + rank(rel_strength) - rank(vol_ratio)
    i.e. reward 6-month momentum, reward beating SPY, penalise high
    volatility relative to a sector's own history. Ranks are used
    instead of raw values so one extreme sector cannot dominate.

    Sell-off sectors are excluded outright — they cannot be bought,
    however strong their 3-month bounce looks.

Rules:
    * Rebalance monthly, on the last trading day of the month.
    * Hold TOP_N sectors, equal weight (1 / TOP_N each).
    * Turnover cap: at most MAX_TURNOVER (60%) of the portfolio may
      turn over in one rebalance. If the target weights would turn
      over more than that, the change is scaled back toward the
      current weights until the cap is met. This is the honest cost
      of discipline: sometimes you keep a weakening sector one more
      month because selling everything at once is too expensive.
    * Costs: (TRANSACTION_COST + SLIPPAGE) = 15 bps, charged on the
      notional actually traded at each rebalance.
    * No look-ahead: weights set on date t use only features dated
      <= t, and the new weights apply from the next trading day.
"""

import numpy as np
import pandas as pd

from . import MAX_TURNOVER, SECTORS, SLIPPAGE, TOP_N, TRANSACTION_COST, WARMUP_DAYS


def rank_sectors(features, date):
    """Rank sectors on one rebalance date. Returns a scored DataFrame."""
    mom = features["mom_6m"].loc[date]
    rs = features["rel_strength"].loc[date]
    vr = features["vol_ratio"].loc[date]
    regime = features["regime"].loc[date]

    df = pd.DataFrame({
        "mom_6m": mom, "rel_strength": rs, "vol_ratio": vr, "regime": regime,
    })
    df = df.dropna()
    if df.empty:
        return df
    # Higher rank = better. Volatility rank is subtracted: low vol wins.
    df["score"] = (
        df["mom_6m"].rank()
        + df["rel_strength"].rank()
        - df["vol_ratio"].rank()
    )
    return df.sort_values("score", ascending=False)


def target_weights(features, date, top_n=TOP_N):
    """Equal-weight target for the top-N eligible sectors on `date`."""
    ranked = rank_sectors(features, date)
    eligible = ranked[ranked["regime"] != "sell-off"]
    chosen = list(eligible.head(top_n).index)
    weights = pd.Series(0.0, index=list(SECTORS))
    if chosen:
        weights[chosen] = 1.0 / len(chosen)
    return weights, chosen


def apply_turnover_cap(current, target, max_turnover=MAX_TURNOVER):
    """Scale a proposed change back so turnover does not exceed the cap.

    Turnover = 0.5 * sum(|target - current|) — the standard definition
    (buying 10% of A and selling 10% of B is 10% turnover, not 20%).
    """
    turnover = 0.5 * (target - current).abs().sum()
    if turnover <= max_turnover or turnover == 0:
        return target, float(turnover)
    # Move only part-way from current to target: the fraction that
    # keeps turnover exactly at the cap.
    alpha = max_turnover / turnover
    capped = current + alpha * (target - current)
    return capped, float(0.5 * (capped - current).abs().sum())


def run_rotation(prices, features, top_n=TOP_N):
    """Walk forward month by month. Returns weights, trades and a log.

    weights: DataFrame, one row per trading day, columns = sectors.
             The weights in force on that day (set at the previous
             month-end rebalance).
    trades:  list of dicts, one per rebalance, with date, chosen
             sectors, turnover and cost charged.
    """
    sectors = list(SECTORS)
    # Last *trading* day of each month — NOT the calendar month-end,
    # which is often a weekend. resample("ME").last().index would
    # give calendar dates like Sat 2020-02-29, which are not in prices.
    month_ends = [g.index[-1] for _, g in prices.groupby(pd.Grouper(freq="ME"))]
    # Only rebalance once we have a full year of history to judge vol.
    warmup_date = prices.index[min(WARMUP_DAYS, len(prices) - 1)]
    month_ends = [d for d in month_ends if d >= warmup_date]
    # Invest on the first day after warm-up instead of sitting in cash
    # until the next month-end: treat the warm-up date as a rebalance.
    if month_ends and month_ends[0] > warmup_date:
        month_ends = [warmup_date] + month_ends
    # Drop a rebalance on the very last trading day: its weights
    # would apply from a day that does not exist in the data, so it
    # could only ever charge a cost and never earn a return.
    month_ends = [d for d in month_ends if d < prices.index[-1]]

    current = pd.Series(0.0, index=sectors)
    weights = pd.DataFrame(0.0, index=prices.index, columns=sectors)
    trades = []
    in_force = current.copy()

    for date in month_ends:
        target, chosen = target_weights(features, date, top_n=top_n)
        capped, turnover = apply_turnover_cap(current, target)
        cost = turnover * 2 * (TRANSACTION_COST + SLIPPAGE)
        if date == warmup_date:
            # Initial allocation, not a rebalance: no prior position
            # to turn over, so no cost is charged on it.
            cost = 0.0
        # cost is charged on traded notional: turnover*2 is the gross
        # traded amount (buys + sells); see backtest.py for how it is
        # applied to the equity curve.
        sold = [t for t in sectors if current[t] > 0 and capped[t] == 0]
        trades.append({
            "date": date, "chosen": chosen, "sold": sold,
            "turnover": turnover, "cost": cost,
            "weights": capped.to_dict(),
        })
        current = capped
        in_force = capped.copy()
        # Weights apply from the next trading day, not the rebalance day.
        nxt = prices.index[prices.index > date]
        if len(nxt):
            weights.loc[nxt[0]:, :] = in_force.values

    return weights, trades
