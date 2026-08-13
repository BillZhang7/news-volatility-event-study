"""
Step 2: Detect extreme negative sentiment events (Plan B).

Plan B rule:
  daily sentiment shock = score below stock's 60-day mean by 1.5 std
  require at least 5 prior news-days in the lookback window
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    MIN_NEWS_DAYS_IN_LOOKBACK,
    SENTIMENT_LOOKBACK_DAYS,
    SENTIMENT_STD_THRESHOLD,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def detect_events(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per detected event with context stats.
    """
    daily = daily.sort_values(["ticker", "date"]).copy()
    events: list[dict] = []

    for ticker, grp in daily.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i, row in grp.iterrows():
            lookback = grp[grp["date"] < row["date"]]
            lookback = lookback[
                lookback["date"] >= row["date"] - pd.Timedelta(days=SENTIMENT_LOOKBACK_DAYS)
            ]
            if len(lookback) < MIN_NEWS_DAYS_IN_LOOKBACK:
                continue

            mu = lookback["sentiment_score"].mean()
            sigma = lookback["sentiment_score"].std(ddof=1)
            if sigma == 0 or np.isnan(sigma):
                continue

            threshold = mu - SENTIMENT_STD_THRESHOLD * sigma
            if row["sentiment_score"] < threshold:
                events.append(
                    {
                        "ticker": ticker,
                        "event_date": row["date"],
                        "sentiment_score": row["sentiment_score"],
                        "lookback_mean": mu,
                        "lookback_std": sigma,
                        "threshold": threshold,
                        "headline_count": row["headline_count"],
                        "z_score": (row["sentiment_score"] - mu) / sigma,
                    }
                )

    out = pd.DataFrame(events)
    if not out.empty:
        out = out.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    return out


def main() -> None:
    print("=" * 50)
    print("Step 2: Event detection (Plan B)")
    print("=" * 50)

    daily_path = DATA_DIR / "daily_sentiment.csv"
    if not daily_path.exists():
        raise FileNotFoundError("Run sentiment.py first.")

    daily = pd.read_csv(daily_path, parse_dates=["date"])
    events = detect_events(daily)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "events.csv"
    events.to_csv(path, index=False)

    print(f"Events detected: {len(events)}")
    if len(events) > 0:
        print(f"Date range: {events['event_date'].min().date()} -> {events['event_date'].max().date()}")
        print("\nSample events:")
        print(events.head(5).to_string(index=False))
    else:
        print("No events found — may need more news history from yfinance.")

    print(f"\nSaved: {path}")
    print("Step 2 complete.")


if __name__ == "__main__":
    main()
