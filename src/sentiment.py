"""
Step 1: Score headline sentiment with FinBERT (fallback to VADER if needed).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import FINBERT_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class SentimentScorer:
    """
    Score = positive_prob - negative_prob  (higher = more positive)

    Uses FinBERT when available; falls back to VADER on unsupported setups.
    """

    def __init__(self) -> None:
        self.backend = "finbert"
        self._pipe = None
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                "sentiment-analysis",
                model=FINBERT_MODEL,
                tokenizer=FINBERT_MODEL,
                truncation=True,
                max_length=128,
            )
        except Exception:
            self.backend = "vader"
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> dict:
        if self.backend == "finbert":
            result = self._pipe(text)[0]
            label = result["label"].lower()
            conf = float(result["score"])
            if label == "positive":
                pos, neg = conf, 1 - conf
            elif label == "negative":
                pos, neg = 1 - conf, conf
            else:
                pos = neg = (1 - conf) / 2
        else:
            scores = self._vader.polarity_scores(text)
            pos = scores["pos"]
            neg = scores["neg"]

        sentiment = pos - neg
        return {
            "positive_prob": pos,
            "negative_prob": neg,
            "sentiment_score": sentiment,
            "backend": self.backend,
        }

    def score_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        if self.backend == "finbert":
            outputs = self._pipe(texts, batch_size=batch_size)
            rows: list[dict] = []
            for text, result in zip(texts, outputs):
                label = result["label"].lower()
                conf = float(result["score"])
                if label == "positive":
                    pos, neg = conf, 1 - conf
                elif label == "negative":
                    pos, neg = 1 - conf, conf
                else:
                    pos = neg = (1 - conf) / 2
                rows.append(
                    {
                        "positive_prob": pos,
                        "negative_prob": neg,
                        "sentiment_score": pos - neg,
                        "backend": self.backend,
                    }
                )
            return rows
        return [self.score_text(t) for t in texts]


def score_news(news: pd.DataFrame) -> pd.DataFrame:
    scorer = SentimentScorer()
    print(f"Sentiment backend: {scorer.backend}")

    texts = news["title"].tolist()
    scores = scorer.score_batch(texts)
    scored = news.copy()
    for key in ["positive_prob", "negative_prob", "sentiment_score", "backend"]:
        scored[key] = [s[key] for s in scores]
    return scored


def daily_sentiment(scored_news: pd.DataFrame) -> pd.DataFrame:
    """Average headline sentiment per (ticker, date)."""
    daily = (
        scored_news.groupby(["ticker", "date"], as_index=False)
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            negative_prob=("negative_prob", "mean"),
            headline_count=("title", "count"),
        )
        .sort_values(["ticker", "date"])
    )
    return daily


def main() -> None:
    print("=" * 50)
    print("Step 1: Sentiment scoring")
    print("=" * 50)

    news_path = DATA_DIR / "news_raw.csv"
    if not news_path.exists():
        raise FileNotFoundError(f"Run fetch_data.py first. Missing {news_path}")

    news = pd.read_csv(news_path, parse_dates=["published_at", "date"])
    print(f"Headlines: {len(news)}")

    scored = score_news(news)
    scored_path = DATA_DIR / "news_scored.csv"
    scored.to_csv(scored_path, index=False)

    daily = daily_sentiment(scored)
    daily_path = DATA_DIR / "daily_sentiment.csv"
    daily.to_csv(daily_path, index=False)

    print(f"Saved: {scored_path}")
    print(f"Saved: {daily_path}")
    print(f"Daily rows: {len(daily)}")
    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()
