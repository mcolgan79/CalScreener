#!/usr/bin/env python3
"""
CalScreener - screen S&P 500 options chains for calendar-spread candidates
sitting in IV term-structure backwardation.

Data source: Yahoo Finance via yfinance. Quotes are whatever Yahoo last
published for each contract (not necessarily real-time), which is fine for
judging the overall shape of a day's IV term structure but is not a live
trading feed.

Definitions
-----------
IV index (per leg):
    ATM calendar   -> average of the call and put implied vol at the strike
                       closest to spot.
    35-delta double
    calendar       -> implied vol at the strike whose Black-Scholes delta is
                       closest to +0.35 (calls) / -0.35 (puts).

Forward factor:
    Ratio of annualized variance between the front and back leg:

        forward_factor = (IV_front^2 / IV_back^2) - 1

    IV is already an annualized number, so no additional time-scaling is
    needed to compare the two variances. A positive forward factor means
    front-month variance is richer than back-month variance, i.e. the term
    structure is in backwardation -- exactly the condition that makes a long
    calendar (short front / long back) attractive. forward_factor > 0.20
    means front variance is running at least 20% hot relative to the back
    month.

This is a screening heuristic, not a formal no-arbitrage forward-vol
calculation -- treat the ranking as a starting point for further diligence,
not a trade signal on its own.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# --------------------------------------------------------------------------
# Config defaults
# --------------------------------------------------------------------------
TOP_N_DEFAULT = 50
FRONT_TARGET_DTE = 60
BACK_TARGET_DTE = 90
DTE_TOLERANCE = 20          # how far from the target DTE an expiry may sit
FF_THRESHOLD = 0.20         # forward factor priority cutoff (20%)
RISK_FREE_RATE = 0.05
TARGET_DELTA = 0.35
STALE_DAYS = 5              # contract quotes older than this are flagged
LIQUIDITY_LOOKBACK_DAYS = "1mo"

FALLBACK_TICKERS = [
    # Used only if the live Wikipedia constituent list can't be fetched.
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "MA", "HD", "PG", "COST", "JNJ", "NFLX", "BAC",
    "ABBV", "CRM", "CVX", "KO", "MRK", "AMD", "PEP", "TMO", "WMT", "LIN",
    "ADBE", "MCD", "CSCO", "ACN", "ABT", "ORCL", "WFC", "DHR", "TXN", "GE",
    "PM", "IBM", "CAT", "INTU", "NOW", "VZ", "DIS", "AMGN", "QCOM", "UNP",
]


# --------------------------------------------------------------------------
# Universe selection
# --------------------------------------------------------------------------
def get_sp500_tickers() -> list[str]:
    """Scrape the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (CalScreener)"}
    try:
        import requests

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        tickers = tables[0]["Symbol"].tolist()
        # yfinance wants BRK-B / BF-B rather than BRK.B / BF.B
        return [t.replace(".", "-") for t in tickers]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not fetch S&P 500 list from Wikipedia ({exc}); "
              f"falling back to a static large-cap list.", file=sys.stderr)
        return list(FALLBACK_TICKERS)


def rank_by_liquidity(tickers: list[str], top_n: int) -> list[str]:
    """Rank tickers by trailing average dollar volume and return the top N."""
    print(f"[info] pulling ~1mo of price history for {len(tickers)} names "
          f"to rank liquidity...")
    data = yf.download(
        tickers,
        period=LIQUIDITY_LOOKBACK_DAYS,
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=True,
    )

    dollar_vol = {}
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna()
            if df.empty:
                continue
            avg_close = df["Close"].tail(10).mean()
            avg_vol = df["Volume"].tail(10).mean()
            if np.isnan(avg_close) or np.isnan(avg_vol):
                continue
            dollar_vol[t] = avg_close * avg_vol
        except Exception:  # noqa: BLE001
            continue

    ranked = sorted(dollar_vol.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in ranked[:top_n]]


# --------------------------------------------------------------------------
# Options math
# --------------------------------------------------------------------------
def bs_delta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str) -> float:
    """Black-Scholes delta. T in years, sigma annualized."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def forward_factor(iv_front: float, iv_back: float) -> float:
    if not iv_front or not iv_back or iv_back == 0:
        return np.nan
    return (iv_front ** 2) / (iv_back ** 2) - 1.0


def is_stale(last_trade_date, as_of: dt.datetime, max_age_days: int) -> bool:
    if last_trade_date is None or pd.isna(last_trade_date):
        return True
    ltd = pd.Timestamp(last_trade_date)
    if ltd.tzinfo is not None:
        ltd = ltd.tz_convert(None)
    return (as_of - ltd).days > max_age_days


# --------------------------------------------------------------------------
# Per-ticker data structures
# --------------------------------------------------------------------------
@dataclass
class LegQuote:
    strike: float
    iv: float
    open_interest: float
    stale: bool
    delta: Optional[float] = None


@dataclass
class ExpiryChain:
    expiry: str
    dte: int
    calls: pd.DataFrame
    puts: pd.DataFrame


def pick_expiry(expirations: list[str], today: dt.date, target_dte: int,
                 tolerance: int, exclude: Optional[str] = None) -> Optional[tuple[str, int]]:
    best = None
    best_diff = None
    for exp in expirations:
        if exp == exclude:
            continue
        exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        diff = abs(dte - target_dte)
        if diff <= tolerance and (best_diff is None or diff < best_diff):
            best, best_diff = (exp, dte), diff
    return best


def atm_leg(calls: pd.DataFrame, puts: pd.DataFrame, spot: float,
            as_of: dt.datetime) -> Optional[LegQuote]:
    if calls.empty or puts.empty:
        return None
    c_idx = (calls["strike"] - spot).abs().idxmin()
    p_idx = (puts["strike"] - spot).abs().idxmin()
    c, p = calls.loc[c_idx], puts.loc[p_idx]
    iv_vals = [v for v in (c["impliedVolatility"], p["impliedVolatility"])
               if v and v > 0]
    if not iv_vals:
        return None
    stale = is_stale(c.get("lastTradeDate"), as_of, STALE_DAYS) or \
        is_stale(p.get("lastTradeDate"), as_of, STALE_DAYS)
    oi = np.nansum([c.get("openInterest", 0) or 0, p.get("openInterest", 0) or 0])
    return LegQuote(strike=float(c["strike"]), iv=float(np.mean(iv_vals)),
                     open_interest=float(oi), stale=stale)


def delta_leg(chain: pd.DataFrame, spot: float, T: float, r: float,
              option_type: str, target_delta: float,
              as_of: dt.datetime) -> Optional[LegQuote]:
    if chain.empty:
        return None
    df = chain.copy()
    df = df[df["impliedVolatility"] > 0]
    if df.empty:
        return None
    signed_target = target_delta if option_type == "call" else -target_delta
    df["delta"] = df.apply(
        lambda row: bs_delta(spot, row["strike"], T, r,
                              row["impliedVolatility"], option_type), axis=1)
    df = df.dropna(subset=["delta"])
    if df.empty:
        return None
    idx = (df["delta"] - signed_target).abs().idxmin()
    row = df.loc[idx]
    stale = is_stale(row.get("lastTradeDate"), as_of, STALE_DAYS)
    return LegQuote(strike=float(row["strike"]), iv=float(row["impliedVolatility"]),
                     open_interest=float(row.get("openInterest", 0) or 0),
                     stale=stale, delta=float(row["delta"]))


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------
def screen_ticker(ticker: str, front_target: int, back_target: int,
                   tolerance: int, r: float, target_delta: float) -> list[dict]:
    rows: list[dict] = []
    today = dt.date.today()
    now = dt.datetime.now()

    try:
        tk = yf.Ticker(ticker)
        expirations = list(tk.options)
        if not expirations:
            return rows

        spot = tk.fast_info.get("lastPrice") if hasattr(tk, "fast_info") else None
        if not spot:
            hist = tk.history(period="5d")
            if hist.empty:
                return rows
            spot = float(hist["Close"].iloc[-1])

        front = pick_expiry(expirations, today, front_target, tolerance)
        if front is None:
            return rows
        back = pick_expiry(expirations, today, back_target, tolerance,
                            exclude=front[0])
        if back is None:
            return rows

        front_exp, front_dte = front
        back_exp, back_dte = back
        if back_dte <= front_dte:
            return rows

        front_chain = tk.option_chain(front_exp)
        back_chain = tk.option_chain(back_exp)
        T_front = front_dte / 365.0
        T_back = back_dte / 365.0

        # --- Long calendar, ATM ---
        f_leg = atm_leg(front_chain.calls, front_chain.puts, spot, now)
        b_leg = atm_leg(back_chain.calls, back_chain.puts, spot, now)
        if f_leg and b_leg:
            ff = forward_factor(f_leg.iv, b_leg.iv)
            rows.append({
                "ticker": ticker, "strategy": "Long Calendar (ATM)",
                "spot": round(spot, 2),
                "front_expiry": front_exp, "front_dte": front_dte,
                "back_expiry": back_exp, "back_dte": back_dte,
                "front_strike": f_leg.strike, "back_strike": b_leg.strike,
                "front_iv": round(f_leg.iv, 4), "back_iv": round(b_leg.iv, 4),
                "forward_factor": round(ff, 4) if not np.isnan(ff) else np.nan,
                "front_oi": int(f_leg.open_interest), "back_oi": int(b_leg.open_interest),
                "total_oi": int(f_leg.open_interest + b_leg.open_interest),
                "stale_quote": f_leg.stale or b_leg.stale,
            })

        # --- Long double calendar, +/-35 delta ---
        fc = delta_leg(front_chain.calls, spot, T_front, r, "call", target_delta, now)
        bc = delta_leg(back_chain.calls, spot, T_back, r, "call", target_delta, now)
        fp = delta_leg(front_chain.puts, spot, T_front, r, "put", target_delta, now)
        bp = delta_leg(back_chain.puts, spot, T_back, r, "put", target_delta, now)
        if fc and bc and fp and bp:
            ff_call = forward_factor(fc.iv, bc.iv)
            ff_put = forward_factor(fp.iv, bp.iv)
            ff_avg = np.nanmean([ff_call, ff_put])
            front_iv_avg = np.mean([fc.iv, fp.iv])
            back_iv_avg = np.mean([bc.iv, bp.iv])
            rows.append({
                "ticker": ticker, "strategy": "Long Double Calendar (35d)",
                "spot": round(spot, 2),
                "front_expiry": front_exp, "front_dte": front_dte,
                "back_expiry": back_exp, "back_dte": back_dte,
                "front_strike": f"{fc.strike}C/{fp.strike}P",
                "back_strike": f"{bc.strike}C/{bp.strike}P",
                "front_iv": round(front_iv_avg, 4), "back_iv": round(back_iv_avg, 4),
                "forward_factor": round(ff_avg, 4) if not np.isnan(ff_avg) else np.nan,
                "front_oi": int(fc.open_interest + fp.open_interest),
                "back_oi": int(bc.open_interest + bp.open_interest),
                "total_oi": int(fc.open_interest + fp.open_interest +
                                 bc.open_interest + bp.open_interest),
                "stale_quote": any([fc.stale, bc.stale, fp.stale, bp.stale]),
            })

    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {ticker}: {exc}", file=sys.stderr)

    return rows


def rank_results(df: pd.DataFrame, front_target: int, back_target: int,
                  ff_threshold: float) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["forward_factor"]).copy()
    df["meets_ff_threshold"] = df["forward_factor"] >= ff_threshold
    df["dte_distance"] = (
        (df["front_dte"] - front_target).abs() + (df["back_dte"] - back_target).abs()
    )
    df = df.sort_values(
        by=["meets_ff_threshold", "forward_factor", "dte_distance", "total_oi"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Screen S&P 500 options for calendar-spread backwardation.")
    p.add_argument("--top-n", type=int, default=TOP_N_DEFAULT,
                    help="How many of the most liquid S&P 500 names to scan.")
    p.add_argument("--front-dte", type=int, default=FRONT_TARGET_DTE)
    p.add_argument("--back-dte", type=int, default=BACK_TARGET_DTE)
    p.add_argument("--dte-tolerance", type=int, default=DTE_TOLERANCE)
    p.add_argument("--ff-threshold", type=float, default=FF_THRESHOLD,
                    help="Forward factor cutoff for priority ranking, e.g. 0.20 = 20%%.")
    p.add_argument("--delta", type=float, default=TARGET_DELTA,
                    help="Target |delta| for the double calendar strikes.")
    p.add_argument("--rate", type=float, default=RISK_FREE_RATE,
                    help="Risk-free rate used for delta calc.")
    p.add_argument("--tickers", nargs="*", default=None,
                    help="Override the universe with an explicit ticker list "
                         "(skips the liquidity ranking step).")
    p.add_argument("--max", type=int, default=None,
                    help="Cap the number of results printed.")
    p.add_argument("--output", type=str, default=None,
                    help="Optional CSV path to save full ranked results.")
    p.add_argument("--sleep", type=float, default=0.3,
                    help="Seconds to sleep between per-ticker option pulls "
                         "(politeness delay for Yahoo).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.tickers:
        universe = args.tickers
    else:
        tickers = get_sp500_tickers()
        universe = rank_by_liquidity(tickers, args.top_n)

    print(f"[info] screening {len(universe)} tickers: {', '.join(universe)}")

    all_rows: list[dict] = []
    for i, ticker in enumerate(universe, 1):
        print(f"[info] ({i}/{len(universe)}) {ticker}...")
        rows = screen_ticker(ticker, args.front_dte, args.back_dte,
                              args.dte_tolerance, args.rate, args.delta)
        all_rows.extend(rows)
        time.sleep(args.sleep)

    if not all_rows:
        print("[info] no candidates found.")
        return

    df = pd.DataFrame(all_rows)
    ranked = rank_results(df, args.front_dte, args.back_dte, args.ff_threshold)

    if ranked.empty:
        print("[info] no candidates with usable IV data.")
        return

    display_cols = [
        "ticker", "strategy", "spot", "forward_factor", "meets_ff_threshold",
        "front_expiry", "front_dte", "front_strike", "front_iv", "front_oi",
        "back_expiry", "back_dte", "back_strike", "back_iv", "back_oi",
        "total_oi", "stale_quote",
    ]
    out = ranked[display_cols]
    if args.max:
        out = out.head(args.max)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\n" + out.to_string(index=False))

    if args.output:
        ranked[display_cols].to_csv(args.output, index=False)
        print(f"\n[info] full ranked results written to {args.output}")


if __name__ == "__main__":
    main()
