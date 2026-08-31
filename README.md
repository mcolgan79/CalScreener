# CalScreener

Screens the most liquid S&P 500 names for options calendar-spread candidates
whose IV term structure is in backwardation (front-month IV running hot
relative to the back month), using [yfinance](https://github.com/ranaroussi/yfinance)
for quotes.

Quotes come from whatever Yahoo Finance last published per contract — not a
live feed, but recent enough to read the day's overall IV term structure.
Contracts with no trade in the last 5 days are flagged `stale_quote=True` in
the output rather than silently trusted.

## What it screens for

1. **Universe**: scrapes the current S&P 500 list from Wikipedia, then ranks
   by trailing 10-day average dollar volume and keeps the top 50 (configurable).
2. **Expirations**: for each name, picks the listed expiration closest to
   60 DTE ("front") and the one closest to 90 DTE ("back"), within a
   tolerance window.
3. **Two candidate structures per name**:
   - **Long Calendar (ATM)** — call/put IV averaged at the strike closest to spot.
   - **Long Double Calendar (35Δ)** — call leg at +0.35 delta, put leg at
     -0.35 delta (delta computed via Black-Scholes from Yahoo's implied vol,
     since Yahoo doesn't publish greeks directly).
4. **Forward factor** — ratio of annualized variance between the two legs:

   ```
   forward_factor = (IV_front / IV_back)^2 - 1
   ```

   A positive value means front-month variance is richer than back-month
   variance (backwardation) — the condition a long calendar wants. This is a
   screening heuristic, not a formal no-arbitrage forward-vol calculation.

## Ranking

Results are sorted in this priority order (each one a tiebreaker for the one
before it):

1. Forward factor ≥ 20% (`--ff-threshold`) sorts first
2. Higher forward factor
3. Closer to 60/90 DTE on both legs
4. Higher combined open interest across both legs

## Usage

```bash
pip install -r requirements.txt
python3 calscreener.py
```

Useful flags:

```bash
# Quick test against a handful of names instead of the full S&P 500 scan
python3 calscreener.py --tickers AAPL MSFT NVDA

# Save full ranked output to CSV, show only the top 20
python3 calscreener.py --output results.csv --max 20

# Widen/narrow the DTE targets, tolerance, or the backwardation cutoff
python3 calscreener.py --front-dte 45 --back-dte 90 --dte-tolerance 15 --ff-threshold 0.15
```

Run `python3 calscreener.py --help` for the full flag list.

## Notes / limitations

- Yahoo's `impliedVolatility` field is Yahoo's own model output, not exchange
  data — treat it as directionally useful, not exact.
- The liquidity ranking and per-ticker option pulls make a lot of network
  calls; a full 50-name run can take a few minutes and is politeness-delayed
  (`--sleep`) to avoid hammering Yahoo.
- This is a screening tool, not a trading signal — validate any candidate's
  live quotes and greeks in your broker platform before trading it.
