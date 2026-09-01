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
4. **Forward IV / forward factor** — variance is additive over time, so the
   implied vol of the forward period *between* the two expirations solves:

   ```
   Forward_IV = sqrt( (IV_back^2 * T_back - IV_front^2 * T_front) / (T_back - T_front) )
   forward_factor = (IV_front - Forward_IV) / Forward_IV
   ```

   (`T` = DTE / 365.) A positive forward factor means the front month is
   priced hotter than the market's own implied vol for the stretch between
   the two expirations — backwardation, and the condition a long calendar
   wants. This is a screening heuristic, not a trade signal on its own.

   **When Forward_IV is unreliable**, `forward_factor` is set to NaN rather
   than trusted, and the row is flagged instead of ranked normally — it's
   still shown, but sorts after every row with a real forward factor:

   - `negative_forward_variance=True` — the expression under the square root
     went negative; no real forward vol satisfies the equation. The most
     extreme form of backwardation there is.
   - `forward_iv_below_floor=True` — the equation has a real, positive
     solution, but it's below the same 2% sanity floor used for raw quotes.
     Dividing by a near-zero Forward_IV means ordinary noise in Yahoo's IV
     quotes (a point or two either way) swings `forward_factor` by hundreds
     or thousands of percent — not a real signal, just the formula
     amplifying quote noise near its own singularity. This is what produces
     the implausible thousands-of-percent readings if left unguarded.

   Either way, review the row by hand rather than trust an unstable number.
   Note this instability is inherent to a *tight* front/back DTE spacing:
   with 60/90-day legs specifically, a real forward vol only exists at all
   when back IV is at least ≈82% of front IV (`sqrt(60/90)`); anything more
   inverted than that lands in one of the two flagged cases above, which is
   expected, not a bug.

   Any contract with no live bid/ask, or an implied vol outside a 2%-300%
   sanity band, is treated as an unreliable Yahoo quote and excluded from
   strike selection (falling back to the next nearest tradeable strike, or
   dropping the row entirely if nothing qualifies). Without this guard, an
   illiquid strike's junk IV can distort the forward-vol calculation.

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

## Running it from your phone (web UI)

`streamlit_app.py` wraps the screener in a mobile-friendly web page. Easiest
free way to get a real URL you can bookmark on your phone:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
2. Click **New app**, pick this repo (`mcolgan79/CalScreener`), the
   `claude/sp500-options-backwardation-jsiecc` branch (or `main` once merged),
   and set the main file path to `streamlit_app.py`.
3. Click **Deploy**. Streamlit installs `requirements.txt` and gives you a
   URL like `https://your-app-name.streamlit.app`.
4. Open that URL on your phone and add it to your home screen — it behaves
   like a lightweight app: adjust settings in the sidebar, tap **Run screen**,
   see a sortable results table, and download the CSV.

Notes:
- The free tier sleeps after inactivity, so the first load after a while can
  take ~30 seconds to spin back up.
- A full 50-name scan still has to make dozens of network calls to Yahoo
  Finance per run, so expect it to take a couple of minutes on mobile data.
- To test locally first: `pip install -r requirements.txt && streamlit run streamlit_app.py`.

## Notes / limitations

- Yahoo's `impliedVolatility` field is Yahoo's own model output, not exchange
  data — treat it as directionally useful, not exact.
- The liquidity ranking and per-ticker option pulls make a lot of network
  calls; a full 50-name run can take a few minutes and is politeness-delayed
  (`--sleep`) to avoid hammering Yahoo.
- This is a screening tool, not a trading signal — validate any candidate's
  live quotes and greeks in your broker platform before trading it.
