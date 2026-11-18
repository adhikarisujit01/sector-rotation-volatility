"""Sector rotation on volatility.

Finance first, code second. The idea is simple and it is the idea that
carries the project, not the software engineering:

  * A sector in a sell-off is not just "down". It is down *while* its
    volatility has exploded relative to its own normal level.
  * A sector in a growth regime is rising *without* that volatility
    explosion, and it is beating the market.
  * Every month we move capital out of sell-off sectors and into the
    strongest growth sectors, pay realistic costs for doing so, and
    measure whether that beat simply holding the S&P 500 or a 60/40.

Data: free daily prices from Yahoo Finance (yfinance) for the 10
Select Sector SPDR ETFs, SPY and AGG. No API key, no paid data.
"""

SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

BENCHMARKS = {
    "SPY": "S&P 500",
    "AGG": "US Aggregate Bonds",
}

ALL_TICKERS = list(SECTORS) + list(BENCHMARKS)

# Backtest defaults — all of them are stated here, not hidden in code.
START_DATE = "2019-01-01"   # 5+ years of daily data
TOP_N = 3                   # hold the top 3 sectors, equal weight
REBALANCE = "ME"            # monthly, last trading day of the month
TRANSACTION_COST = 0.0010   # 10 bps per trade
SLIPPAGE = 0.0005           # 5 bps slippage, so 15 bps all-in per trade
MAX_TURNOVER = 0.60         # at most 60% of the portfolio may turn over
                            # in one monthly rebalance
WARMUP_DAYS = 252           # first year is warm-up / regime estimation
                            # only; the reported backtest is out-of-sample
                            # after that date
RISK_FREE_RATE = 0.0        # Sharpe is reported with rf = 0, stated here
