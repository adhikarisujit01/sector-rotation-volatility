"""Download and cache daily prices.

Only free Yahoo Finance data, via yfinance. Prices are adjusted for
splits and dividends (auto_adjust=True), so returns are total returns.
The downloaded file is cached locally at data/prices.csv so a rerun
does not have to hit Yahoo again.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

from . import ALL_TICKERS, START_DATE


def fetch_prices(tickers=None, start=START_DATE, end=None, cache_path=None):
    """Download adjusted daily closes. Returns a DataFrame, tickers as columns.

    If a cache file exists it is used instead of downloading. Set
    cache_path=None to force a fresh download.
    """
    tickers = list(tickers or ALL_TICKERS)
    if cache_path is not None and Path(cache_path).exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Only reuse the cache if it covers the tickers we asked for.
        if set(tickers).issubset(prices.columns):
            return prices[tickers]

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        progress=False, threads=True,
    )
    if raw.empty:
        raise RuntimeError(
            "yfinance returned no data. Check your internet connection "
            "and try again, or run with a cached data/prices.csv."
        )
    # yfinance returns a MultiIndex (Price, Ticker) when several tickers
    # are requested; with auto_adjust there is only "Close".
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw.to_frame(name=tickers[0]) if len(tickers) == 1 else raw
    prices = prices.dropna(how="all").sort_index()
    prices.index = pd.to_datetime(prices.index)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
    return prices


def load_or_fetch(cache_path="data/prices.csv", **kwargs):
    """The normal entry point: use the cache if present, else download."""
    return fetch_prices(cache_path=cache_path, **kwargs)
