# News Sentiment → Volatility Event Study

**Autonomous quant research project** — test whether extreme negative news shocks raise post-event stock volatility (H-A3), using stock-specific sentiment surprises (Plan B).

## Your research design

| Decision | Choice |
|----------|--------|
| Method | Event study |
| Hypothesis | H-A3: vol ↑ after extreme news (not necessarily predictable returns) |
| Events | Plan B: sentiment below 60d personal mean − 1.5σ |
| Window | [-5, +10] trading days |
| Universe | 15 US large caps |
| Sentiment | FinBERT (VADER fallback) |
| News | yfinance (no API key) |

See [RESEARCH.md](RESEARCH.md) for full rationale.

## Pipeline

```bash
python3 src/fetch_data.py    # prices + news
python3 src/sentiment.py     # FinBERT scoring
python3 src/events.py        # Plan B event detection
python3 src/event_study.py   # volatility event study
streamlit run streamlit_app.py
```

## Resume bullets

- Designed an autonomous event-study research project testing whether stock-specific negative news shocks (Plan B sentiment surprise) increase post-event realized volatility (H-A3)
- Built NLP + market data pipeline: yfinance headlines, FinBERT sentiment, event detection, volatility windows [-5,+10], in/out-of-sample summaries
- Delivered interactive Streamlit demo documenting assumptions and data limitations for reproducible alt-data quant research

## Deploy Streamlit Cloud (free)

1. Push this repo to GitHub (see below)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**
3. Repo: `YOUR_USERNAME/news-volatility-event-study`
4. **Main file:** `streamlit_app.py`
5. Deploy → public URL for your resume

> Committed `outputs/` files let the demo run without re-running the pipeline on Cloud.

## Push to GitHub

```bash
cd ~/Desktop/Quant/news-volatility-event-study
git init -b main   # first time only
git add -A && git commit -m "Initial commit"
```

Then on [github.com/new](https://github.com/new) create **empty** repo `news-volatility-event-study` (no README).

**GitHub Desktop:** File → Add Local Repository → select this folder → **Publish repository**.

Or Terminal:

```bash
bash push_to_github.sh https://github.com/BillZhang7/news-volatility-event-study.git
```

## Author

Bill — NYU Data Science + Mathematics

## License

MIT
