"""Run the sector rotation backtest end to end.

    python run.py [--refresh]

--refresh forces a fresh download from Yahoo Finance; otherwise the
cached data/prices.csv is used if it exists.

Outputs, all in results/:
    equity_curve.csv   daily equity for rotation, SPY and 60/40
    equity_curve.png   chart of the three curves
    metrics.json       CAGR, volatility, Sharpe, max drawdown, turnover
    regime_table.csv   monthly regime per sector
    rebalance_log.txt  plain-text narrator for every rebalance
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src import SECTORS, START_DATE
from src.backtest import run_backtest
from src.data import fetch_prices
from src.regimes import compute_features, explain_month, monthly_regime_table
from src.rotation import run_rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="force a fresh download from Yahoo Finance")
    args = parser.parse_args()

    root = Path(__file__).parent
    cache = root / "data" / "prices.csv"
    if args.refresh and cache.exists():
        cache.unlink()
    prices = fetch_prices(start=START_DATE, cache_path=str(cache))
    print(f"Prices: {len(prices)} trading days, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{len(prices.columns)} tickers")

    features = compute_features(prices)
    weights, trades = run_rotation(prices, features)
    curves, metrics, start_date = run_backtest(prices, weights, trades)

    print(f"\nOut-of-sample backtest: {start_date.date()} to "
          f"{curves.index[-1].date()} ({len(curves)} trading days)")
    print(f"{'Strategy':<12} {'CAGR':>8} {'Vol':>8} {'Sharpe':>7} {'Max DD':>8} {'Final $1':>9}")
    for name in ("rotation", "spy", "60/40"):
        m = metrics[name]
        print(f"{name:<12} {m['cagr']:>7.1%} {m['volatility']:>7.1%} "
              f"{m['sharpe']:>7.2f} {m['max_drawdown']:>7.1%} {m['final_value']:>9.2f}")
    r = metrics["rotation"]
    print(f"\nRotation turnover: {r['avg_monthly_turnover']:.1%} per month "
          f"({r['annualised_turnover']:.0%} annualised), "
          f"total costs {r['total_cost_pct']:.2f}% of capital over "
          f"{r['n_rebalances']} rebalances")

    # Regime audit: the most recent sell-off months, and the big ones.
    regime_table = monthly_regime_table(features, prices)
    sell_off_months = regime_table[regime_table["n_sell_off"] >= 3]
    print(f"\nMonths with 3+ sectors in sell-off: {len(sell_off_months)}")
    for _, row in sell_off_months.tail(12).iterrows():
        flagged = [t for t in SECTORS if row[f"{t}_regime"] == "sell-off"]
        print(f"  {row['date']}: {', '.join(flagged)}")

    # Save everything.
    out = root / "results"
    out.mkdir(exist_ok=True)
    curves.to_csv(out / "equity_curve.csv")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    regime_table.to_csv(out / "regime_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    curves.plot(ax=ax)
    ax.set_title("Sector rotation vs S&P 500 vs 60/40 (out-of-sample, $1 invested)")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=150)
    plt.close(fig)

    log_lines = []
    for trade in trades:
        log_lines.append(explain_month(
            trade["date"], features, prices,
            trade["chosen"], trade["sold"]))
    (out / "rebalance_log.txt").write_text("\n\n".join(log_lines))
    print(f"\nSaved: results/equity_curve.csv, results/metrics.json, "
          f"results/regime_table.csv, results/equity_curve.png, "
          f"results/rebalance_log.txt")
    print("Last rebalance:")
    print(log_lines[-1])


if __name__ == "__main__":
    main()
