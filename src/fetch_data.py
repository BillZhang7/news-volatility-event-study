"""
Step 0: Fetch price data and news headlines (yfinance, no API key).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import BENCHMARK, PRICE_END, PRICE_START, UNIVERSE
from demo_news import generate_demo_news

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def build_prices_from_cache(tickers: list[str]) -> pd.DataFrame:
    """Merge cached OHLCV CSV files into a close-price panel."""
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        frames[ticker] = df["Close"]
    if not frames:
        raise FileNotFoundError("No cached ticker CSV files in data/.")
    return pd.DataFrame(frames).sort_index()


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily close prices, one column per ticker."""
    frames: dict[str, pd.Series] = {}
    last_error: Exception | None = None

    for ticker in tickers:
        for attempt in range(1, 4):
            try:
                df = yf.Ticker(ticker).history(start=start, end=end)
                if df.empty:
                    raise ValueError(f"empty history for {ticker}")
                s = df["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                frames[ticker] = s
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(attempt * 3)
        else:
            cached = DATA_DIR / f"{ticker}.csv"
            if cached.exists():
                print(f"  {ticker}: using cached CSV")
                df = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
                frames[ticker] = df["Close"]
            else:
                print(f"  WARNING: failed {ticker} ({last_error})")

    if not frames:
        raise RuntimeError(f"Could not load any prices. Last error: {last_error}")

    prices = pd.DataFrame(frames).sort_index()
    return prices


def fetch_news_for_ticker(ticker: str, retries: int = 3) -> list[dict]:
    """Pull available headlines from yfinance."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(ticker).news or []
            rows: list[dict] = []
            for item in raw:
                title = item.get("title") or item.get("headline")
                if not title:
                    continue
                ts = item.get("providerPublishTime") or item.get("pubDate")
                if ts is None:
                    continue
                published = pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(None)
                rows.append(
                    {
                        "ticker": ticker,
                        "published_at": published,
                        "date": published.normalize(),
                        "title": title.strip(),
                        "publisher": item.get("publisher", ""),
                    }
                )
            return rows
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to fetch news for {ticker}: {last_error}") from last_error


def fetch_all_news(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or UNIVERSE
    all_rows: list[dict] = []
    print(f"Fetching news for {len(tickers)} tickers...")
    for ticker in tickers:
        print(f"  {ticker}...", end=" ")
        rows = fetch_news_for_ticker(ticker)
        print(f"{len(rows)} headlines")
        all_rows.extend(rows)
        time.sleep(0.5)
    if not all_rows:
        raise ValueError("No news returned. Check network or yfinance access.")
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["ticker", "published_at"]).reset_index(drop=True)
    return df


def main() -> None:
    print("=" * 50)
    print("Step 0: Fetch prices + news")
    print("=" * 50)

    tickers = UNIVERSE + [BENCHMARK]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPrices...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cached = [t for t in tickers if (DATA_DIR / f"{t}.csv").exists()]
    if len(cached) >= 8:
        print(f"  Using {len(cached)} cached ticker CSV files")
        prices = build_prices_from_cache(cached)
    else:
        try:
            prices = fetch_prices(tickers, PRICE_START, PRICE_END)
        except RuntimeError:
            print("  Live download failed — using any cached CSV files")
            prices = build_prices_from_cache(tickers)

    price_path = DATA_DIR / "prices.csv"
    prices.to_csv(price_path)
    print(f"  Saved {price_path} ({prices.shape[0]} days x {prices.shape[1]} tickers)")

    print("\nNews...")
    try:
        news = fetch_all_news(UNIVERSE)
        print("  Source: yfinance (live headlines)")
    except RuntimeError as exc:
        print(f"  Live news failed ({exc})")
        print("  Using DEMO headlines so the pipeline can run.")
        print("  Re-run Step 0 later on your machine for real yfinance news.")
        news = generate_demo_news([t for t in UNIVERSE if t in prices.columns])
    news_path = DATA_DIR / "news_raw.csv"
    news.to_csv(news_path, index=False)
    print(f"  Saved {news_path} ({len(news)} headlines)")
    print(f"  Date range: {news['date'].min().date()} -> {news['date'].max().date()}")

    meta = {
        "price_start": PRICE_START,
        "price_end": PRICE_END,
        "headline_count": len(news),
        "tickers": UNIVERSE,
        "news_source": "demo" if (news["publisher"] == "demo_source").any() else "live",
    }
    (DATA_DIR / "fetch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nStep 0 complete.")


if __name__ == "__main__":
    main()
