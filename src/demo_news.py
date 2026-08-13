"""
Generate demo headlines when yfinance news is rate-limited.

NOT real news — lets the pipeline run offline. Re-run fetch_data.py later for live headlines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


NEGATIVE_TITLES = [
    "{t} faces regulatory probe over business practices",
    "{t} warns on outlook as demand slows",
    "{t} hit by lawsuit from investors",
    "{t} shares tumble on disappointing guidance",
    "{t} CEO under scrutiny after internal review",
    "{t} faces supply chain disruption concerns",
]

POSITIVE_TITLES = [
    "{t} beats earnings expectations",
    "{t} announces major product launch",
    "{t} stock rises on strong quarterly results",
    "{t} expands into new markets",
    "{t} wins large contract from enterprise clients",
]

NEUTRAL_TITLES = [
    "{t} scheduled to report quarterly earnings",
    "{t} executives speak at industry conference",
    "{t} announces date for investor day",
]


def generate_demo_news(
    tickers: list[str],
    start: str = "2021-07-01",
    end: str = "2024-12-31",
    events_per_ticker: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    rows: list[dict] = []

    for ticker in tickers:
        event_days = rng.choice(dates, size=min(events_per_ticker, len(dates)), replace=False)
        for day in sorted(event_days):
            roll = rng.random()
            if roll < 0.35:
                title = rng.choice(NEGATIVE_TITLES).format(t=ticker)
            elif roll < 0.7:
                title = rng.choice(POSITIVE_TITLES).format(t=ticker)
            else:
                title = rng.choice(NEUTRAL_TITLES).format(t=ticker)

            published = pd.Timestamp(day) + pd.Timedelta(hours=int(rng.integers(9, 17)))
            rows.append(
                {
                    "ticker": ticker,
                    "published_at": published,
                    "date": pd.Timestamp(day).normalize(),
                    "title": title,
                    "publisher": "demo_source",
                }
            )

    df = pd.DataFrame(rows).sort_values(["ticker", "published_at"]).reset_index(drop=True)
    return df
