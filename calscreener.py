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

Forward IV / forward factor:
    Variance is additive over time, so the implied volatility of the
    "forward" period sitting between the front and back expirations solves:

        IV_back^2 * T_back = IV_front^2 * T_front + Forward_IV^2 * (T_back - T_front)

        Forward_IV = sqrt(
            (IV_back^2 * T_back - IV_front^2 * T_front) / (T_back - T_front)
        )

    T is time to expiration in years (DTE / 365). Forward factor then
    compares the front leg to that forward vol rather than to the back leg
    directly:

        forward_factor = (IV_front - Forward_IV) / Forward_IV

    A positive forward factor means the front month is priced hotter than
    the market's own implied vol for the forward stretch between the two
    expirations -- backwardation, and the condition a long calendar wants.

    The variance-additivity equation can solve to a *negative* number under
    the square root when front-month IV is rich enough relative to the back
    month -- i.e. no real forward vol satisfies the no-arbitrage relationship.
    That is the most extreme form of backwardation, but it also means
    Forward_IV and forward_factor are undefined. Those rows are still
    included in the output (flagged via `negative_forward_variance`) but are
    NOT ranked by forward factor -- pandas' NaN sort puts them after every
    row with a real, computed forward factor, so you can review them by hand
    rather than trust an invented number.

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
MIN_VALID_IV = 0.02         # below this, treat Yahoo's IV as a data artifact
MAX_VALID_IV = 3.00         # above this, same -- reject rather than trust it
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


def implied_forward_iv(iv_front: float, iv_back: float, T_front: float,
                        T_back: float) -> tuple[float, float]:
    """Solve for the implied vol of the front-to-back forward period.

    Returns (forward_iv, forward_variance). forward_iv is NaN when
    forward_variance comes out <= 0 -- the equation has no real solution,
    which signals extreme backwardation rather than a data error.
    """
    if not iv_front or not iv_back or T_back <= T_front:
        return np.nan, np.nan
    var_front = iv_front ** 2 * T_front
    var_back = iv_back ** 2 * T_back
    var_forward = (var_back - var_front) / (T_back - T_front)
    if var_forward <= 0:
        return np.nan, var_forward
    return float(np.sqrt(var_forward)), var_forward


def forward_factor(iv_front: float, forward_iv: float) -> float:
    if not iv_front or not forward_iv or np.isnan(forward_iv) or forward_iv == 0:
        return np.nan
    return (iv_front - forward_iv) / forward_iv


def is_stale(last_trade_date, as_of: dt.datetime, max_age_days: int) -> bool:
    if last_trade_date is None or pd.isna(last_trade_date):
        return True
    ltd = pd.Timestamp(last_trade_date)
    if ltd.tzinfo is not None:
        ltd = ltd.tz_convert(None)
    return (as_of - ltd).days > max_age_days


def has_tradeable_quote(row) -> bool:
    """Reject contracts with no real two-sided market or an out-of-range IV.

    Yahoo's implied-vol model regularly spits out near-zero (or absurdly
    high) IV for thinly-traded/wide-spread contracts. Feeding one of those
    into a variance ratio blows up the forward factor into meaningless
    territory (billions of percent), so anything outside a sane IV band, or
    with no live bid/ask, is treated as unreliable and dropped rather than
    trusted.
    """
    iv = row.get("impliedVolatility")
    if iv is None or pd.isna(iv) or not (MIN_VALID_IV <= iv <= MAX_VALID_IV):
        return False
    bid, ask = row.get("bid", 0) or 0, row.get("ask", 0) or 0
    return bid > 0 and ask > 0


def filter_tradeable(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df.apply(has_tradeable_quote, axis=1)].copy()


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
    calls, puts = filter_tradeable(calls), filter_tradeable(puts)
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
    df = filter_tradeable(chain)
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
            fwd_iv, var_fwd = implied_forward_iv(f_leg.iv, b_leg.iv, T_front, T_back)
            ff = forward_factor(f_leg.iv, fwd_iv)
            rows.append({
                "ticker": ticker, "strategy": "Long Calendar (ATM)",
                "spot": round(spot, 2),
                "front_expiry": front_exp, "front_dte": front_dte,
                "back_expiry": back_exp, "back_dte": back_dte,
                "front_strike": f_leg.strike, "back_strike": b_leg.strike,
                "front_iv": round(f_leg.iv, 4), "back_iv": round(b_leg.iv, 4),
                "forward_iv": round(fwd_iv, 4) if not np.isnan(fwd_iv) else np.nan,
                "forward_factor": round(ff, 4) if not np.isnan(ff) else np.nan,
                "negative_forward_variance": (not np.isnan(var_fwd)) and var_fwd <= 0,
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
            fwd_iv_call, var_fwd_call = implied_forward_iv(fc.iv, bc.iv, T_front, T_back)
            fwd_iv_put, var_fwd_put = implied_forward_iv(fp.iv, bp.iv, T_front, T_back)
            negative_var = (
                ((not np.isnan(var_fwd_call)) and var_fwd_call <= 0)
                or ((not np.isnan(var_fwd_put)) and var_fwd_put <= 0)
            )
            if negative_var:
                # Extreme backwardation on at least one leg -- don't mask it
                # by averaging in the other leg's (possibly normal) factor.
                ff_avg, fwd_iv_avg = np.nan, np.nan
            else:
                ff_call = forward_factor(fc.iv, fwd_iv_call)
                ff_put = forward_factor(fp.iv, fwd_iv_put)
                ff_avg = np.nanmean([ff_call, ff_put])
                fwd_iv_avg = np.nanmean([fwd_iv_call, fwd_iv_put])
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
                "forward_iv": round(fwd_iv_avg, 4) if not np.isnan(fwd_iv_avg) else np.nan,
                "forward_factor": round(ff_avg, 4) if not np.isnan(ff_avg) else np.nan,
                "negative_forward_variance": negative_var,
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
    df = df.copy()
    # forward_factor is NaN when the term structure is so inverted that no
    # real forward vol solves the variance equation (negative_forward_variance
    # = True). Those rows are kept but never win the ranking on forward
    # factor -- sort_values() puts NaN last within each ascending/descending
    # group, so they surface after every row with a real forward factor.
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
        "negative_forward_variance", "forward_iv",
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
