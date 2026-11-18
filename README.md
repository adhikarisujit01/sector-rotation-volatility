# Sector Rotation on Volatility

**A finance project, not a coding project.** The code is deliberately
simple — pandas, a few formulas, one monthly loop. The substance is
the finance: how you tell a sector sell-off from normal noise, how
you rotate capital between sectors, and whether that rotation
actually beats doing nothing, after costs.

> Built by Sujit Adhikari — MS Business Analytics & Statistics,
> BA Finance, AI/NLP Engineer. The idea comes from his own work on
> market-data narration: volatility is the signal that a sector is
> breaking, not just falling.

## The idea in one paragraph

A sector in a sell-off is not just "down". It is down **while its
volatility has exploded relative to its own normal level**. A sector
in a growth regime is rising **without** that volatility explosion,
and it is beating the market. Every month this project moves capital
out of sell-off sectors and into the strongest growth sectors, pays
realistic costs for doing so, and measures whether that beat simply
holding the S&P 500 or a 60/40 portfolio.

## The rules (explicit, testable, fixed in advance)

### Regime detection

For each sector ETF, every day:

| Quantity | Definition |
|---|---|
| `sigma_20` | Annualised std. dev. of daily returns, last 20 days: `std(r, 20) * sqrt(252)` |
| `sigma_1y` | Same, over the last 252 days (1 trading year) |
| `vol_ratio` | `sigma_20 / sigma_1y` — > 1 means "more volatile than its own normal level" |
| `mom_6m` | `price / price 126 days ago - 1` |
| `mom_3m` | `price / price 63 days ago - 1` |
| `rel_strength` | `mom_6m(sector) - mom_6m(SPY)` — > 0 means "beating the market" |

Regimes:

- **SELL-OFF** if `vol_ratio > 2.0 AND mom_6m < 0`
  Volatility has at least doubled vs its 1-year average *and* the
  sector is down over 6 months.
- **GROWTH** if `vol_ratio < 1.2 AND mom_6m > 0.05`
  Rising more than 5% over 6 months without a volatility explosion.
- **NEUTRAL** otherwise.

### Rotation

- Rebalance **monthly**, on the last trading day of the month.
  Weights apply from the **next** trading day — no look-ahead.
- Score each sector by rank:
  `score = rank(mom_6m) + rank(rel_strength) - rank(vol_ratio)`
  Reward momentum, reward beating SPY, penalise high volatility.
  Ranks (not raw values) so one extreme sector cannot dominate.
- Hold the **top 3** eligible sectors, equal weight. Sell-off
  sectors are excluded outright, however strong their bounce looks.
- **Turnover cap: 60%** of the portfolio per rebalance. If the
  target would turn over more, the change is scaled back toward the
  current weights. Discipline has a cost: sometimes you keep a
  weakening sector one more month.
- **Costs: 15 bps all-in per trade** (10 bps transaction cost +
  5 bps slippage), charged on the notional actually traded.

### Out-of-sample design

The first 252 trading days (~1 year) are warm-up only — used to
estimate 1-year volatility and 6-month momentum, but no return from
that period is counted. The reported backtest starts after that.
The rules were fixed in advance; nothing in the reported numbers
was used to choose them.

## Data

Free daily prices from Yahoo Finance (`yfinance`), adjusted for
splits and dividends, for the 10 Select Sector SPDR ETFs plus SPY
and AGG:

XLK Technology, XLF Financials, XLE Energy, XLV Health Care,
XLI Industrials, XLP Consumer Staples, XLU Utilities, XLB Materials,
XLRE Real Estate, XLC Communication Services.

Period: 2019-01-01 to present (5+ years). No API key, no paid data.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py            # uses cached data/prices.csv if present
python run.py --refresh  # force a fresh download from Yahoo Finance
pytest                   # 7 tests, synthetic data, runs offline
```

## Results

Run on 2026-11-18, out-of-sample 2020-01-02 to 2026-11-18
(1,730 trading days), $1 invested, after 15 bps per trade and a
60% monthly turnover cap. **Read this honestly: the rotation did
not beat the S&P 500.** It beat 60/40 on return, but with far more
volatility and a worse drawdown than 60/40, and it lagged SPY on
every metric. That is the result — the project is worth showing
because the method is clean and the reporting is honest, not
because the backtest is a win.

| Strategy | CAGR | Volatility | Sharpe (rf=0) | Max drawdown | Final $1 |
|---|---|---|---|---|---|
| Sector rotation | 11.1% | 19.4% | 0.64 | -28.7% | $2.06 |
| S&P 500 (SPY) | 14.1% | 20.0% | 0.76 | -33.7% | $2.47 |
| 60/40 (SPY/AGG) | 9.0% | 12.7% | 0.74 | -21.7% | $1.81 |

Rotation turnover: 38.1% per month (457% annualised), total costs
9.34% of capital over 83 rebalances. Costs are a real drag — about
1.4 percentage points of return per year — and the turnover cap
binds often enough to matter.

What the regime rule did catch: March 2020, when 9 of 10 sectors
were flagged sell-off at the COVID crash (XLK vol_ratio 3.02x),
and April 2025, when 5 sectors were flagged (XLK, XLE, XLV, XLI,
XLB) in the tariff sell-off. What it missed: the whole of 2022.
Tech fell ~28% that year, but it fell slowly — XLK's vol_ratio
peaked at 1.66x, never reaching the 2.0x sell-off threshold, so
XLK was "neutral" all year. A volatility-explosion rule catches
crashes; it does not catch slow grinds. That is the single most
important limitation of this design, and the first thing to fix
(see Next steps).

What to look at beyond the headline: `results/regime_table.csv`
shows, month by month, which sectors were flagged as sell-off —
including the 2020 COVID crash and the 2022 tech sell-off — and
`results/rebalance_log.txt` narrates every rebalance in plain
English, citing the vol_ratio and momentum numbers that drove it.
No LLM, no API key: the numbers are the evidence, the text is a
template around them.

## What this proves — and what it does not

**It proves:** you can define a regime in equations, implement a
rotation with costs and a turnover cap, avoid look-ahead bias, and
report a backtest honestly against real benchmarks.

**It does not prove:** that this strategy will make money going
forward. The thresholds (2.0x vol, 5% momentum, top 3, 60% cap)
are simple round numbers chosen in advance, not optimised — that
is a feature, not a bug, because optimised thresholds overfit.
The backtest is one history: it includes COVID and 2022, but the
next crash will look different. 60/40 rebalancing costs are
omitted (small, stated here). Past performance is not indicative
of future results.

## Limitations

- Monthly rebalancing is slow: a sector can fall a long way inside
  a month before the next rebalance.
- Equal weighting ignores how strong the signal is — the #1 sector
  gets the same weight as #3.
- Sector ETFs are coarse: "Technology" hides very different
  companies, and rotation cannot pick within a sector.
- Yahoo Finance data is free and good, but it is not a professional
  price feed; corporate actions and survivorship are handled by
  Yahoo, not verified here.
- No risk model beyond volatility: no VaR/CVaR limit, no
  correlation between sectors, no drawdown control.

## Structure

```
run.py                  end-to-end run: download, regimes, rotate, backtest
src/data.py             yfinance download + local cache
src/regimes.py          vol_ratio, momentum, relative strength, regime rules
src/rotation.py         monthly ranking, top-3, turnover cap, costs
src/backtest.py         rotation vs SPY vs 60/40, CAGR / Sharpe / max DD
tests/test_rotation.py  7 tests on synthetic data (offline)
results/                equity curve, metrics, regime table, rebalance log
```

## Next steps (if you want to go deeper)

1. Sensitivity: re-run with vol_ratio 1.5x / 2.5x and top 4 — does
   the result survive, or was it one threshold?
2. Add a drawdown control: cut exposure when the portfolio itself
   is down 10% from its peak.
3. Add VaR / CVaR reporting per rebalance.
4. Replace equal weights with momentum-proportional weights.
