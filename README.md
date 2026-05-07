# AI Hedge Fund — India Edition

A fork of [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) customised for **Indian equity markets** (NSE / BSE). This project is for **educational purposes only** and is not intended for real trading or investment.

---

## What's different from the original

### Indian market calibration
- DCF valuations use **INR-denominated rates**: risk-free rate pegged to the 10-year G-Sec yield, equity risk premium calibrated to Indian markets.
- `yfinance` adapter (`src/tools/yfinance_api.py`) added to fetch price and fundamental data for `.NS` (NSE) and `.BO` (BSE) tickers — no FinancialDatasets.ai dependency for Indian stocks.
- Aswath Damodaran, Valuation Analyst, and Warren Buffett agents updated to handle INR figures and Indian accounting conventions.

### Parallel agent execution
- All analyst agents run **concurrently** via `ThreadPoolExecutor` instead of sequentially through LangGraph. On 8 agents this typically halves wall-clock time.
- Rich live progress table updates in real time as each agent finishes.

### Notion integration
- Every completed run is automatically pushed to a **Notion database** (one page per run) with the trading decision, confidence, token usage, and the full markdown report embedded.
- Requires `NOTION_API_KEY` and `NOTION_DATABASE_ID` in `.env`.

### Markdown report generation
- `src/utils/report.py` saves a structured markdown report to `reports/` after every run, including per-agent signals and token cost.

### Extended model registry
- `src/llm/api_models.json` updated with `claude-sonnet-4-6` and `claude-haiku-4-5` in addition to `claude-opus-4-7`.
- `ChatAnthropic` initialised with `max_tokens=8192` to prevent truncation on verbose reasoning agents.

### India-focused analyst defaults
The recommended analyst set for Indian stocks is:
- `aswath_damodaran` — India-calibrated DCF (INR rates)
- `rakesh_jhunjhunwala` — The Big Bull of India
- `mohnish_pabrai` — Dhandho value investing
- `technical_analyst`, `fundamentals_analyst`, `valuation_analyst`, `growth_analyst`, `sentiment_analyst` — free (no LLM tokens)

---

## Agents

| # | Agent | Style |
|---|-------|-------|
| 1 | Aswath Damodaran | Dean of Valuation — story + numbers + DCF |
| 2 | Ben Graham | Father of Value Investing — margin of safety |
| 3 | Bill Ackman | Activist investor |
| 4 | Cathie Wood | Disruptive growth / innovation |
| 5 | Charlie Munger | Quality businesses at fair prices |
| 6 | Michael Burry | Deep value contrarian |
| 7 | Mohnish Pabrai | Dhandho — low risk, high uncertainty |
| 8 | Nassim Taleb | Tail risk / antifragility |
| 9 | Peter Lynch | Ten-baggers in everyday businesses |
| 10 | Phil Fisher | Scuttlebutt growth research |
| 11 | Rakesh Jhunjhunwala | Big Bull of India — macro + growth |
| 12 | Stanley Druckenmiller | Macro / asymmetric opportunities |
| 13 | Warren Buffett | Wonderful companies at fair prices |
| 14 | Valuation Analyst | DCF, owner earnings, EV/EBITDA, residual income |
| 15 | Sentiment Analyst | Insider trading + news sentiment |
| 16 | Fundamentals Analyst | Financial statement analysis |
| 17 | Technical Analyst | Price action, momentum, volatility regimes |
| 18 | Growth Analyst | Revenue / earnings / FCF growth trends |
| 19 | News Sentiment Analyst | LLM-based news analysis |
| 20 | Risk Manager | Volatility-adjusted position sizing |
| 21 | Portfolio Manager | Final trade decision |

---

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a SEBI-registered financial advisor for investment decisions
- Past performance does not indicate future results

---

## Table of Contents
- [How to Install](#how-to-install)
- [How to Run](#how-to-run)
- [Notion Integration](#notion-integration)
- [License](#license)

---

## How to Install

### 1. Clone the repository

```bash
git clone https://github.com/aman468/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. Install dependencies

```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

poetry install
```

### 3. Set up API keys

```bash
cp .env.example .env   # or edit .env directly
```

Minimum required keys:

```env
# LLM provider — at least one required
ANTHROPIC_API_KEY=your-key       # recommended: claude-sonnet-4-6
OPENAI_API_KEY=your-key
DEEPSEEK_API_KEY=your-key

# Financial data (required for non-Indian tickers)
FINANCIAL_DATASETS_API_KEY=your-key

# Notion report push (optional)
NOTION_API_KEY=your-notion-integration-key
NOTION_DATABASE_ID=your-database-id
```

For Indian stocks (`.NS` / `.BO`) the `yfinance` adapter is used automatically — no `FINANCIAL_DATASETS_API_KEY` needed.

---

## How to Run

### Command line

```bash
# NSE stock, last 12 months, recommended India agents
poetry run python -m src.main \
  --tickers RELIANCE.NS \
  --start-date 2025-01-01 \
  --end-date 2026-01-01 \
  --model claude-sonnet-4-6 \
  --analysts aswath_damodaran,rakesh_jhunjhunwala,mohnish_pabrai,technical_analyst,fundamentals_analyst,valuation_analyst,growth_analyst,sentiment_analyst \
  --report-dir reports \
  --show-reasoning
```

Multiple tickers:

```bash
poetry run python -m src.main \
  --tickers INFY.NS,TCS.NS,HDFCBANK.NS \
  --start-date 2025-01-01 \
  --end-date 2026-01-01 \
  --model claude-sonnet-4-6 \
  --analysts aswath_damodaran,rakesh_jhunjhunwala,mohnish_pabrai,technical_analyst,fundamentals_analyst,valuation_analyst,growth_analyst,sentiment_analyst \
  --report-dir reports
```

BSE tickers use the `.BO` suffix:

```bash
--tickers RELIANCE.BO,WIPRO.BO
```

### Ticker format

| Exchange | Suffix | Example |
|----------|--------|---------|
| NSE | `.NS` | `JIOFIN.NS` |
| BSE | `.BO` | `JIOFIN.BO` |
| US (NYSE/NASDAQ) | none | `AAPL` |

### Output

After each run you get:
1. A live Rich progress table showing all agents updating in parallel
2. A trading decision table printed to the terminal
3. A markdown report saved to `reports/<ticker>_<timestamp>.md`
4. A Notion page in your analysis database (if `NOTION_API_KEY` is configured)

---

## Notion Integration

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) and create an integration.
2. Copy the **Internal Integration Secret** → set as `NOTION_API_KEY` in `.env`.
3. Create a Notion database (or use the one pre-configured at `NOTION_DATABASE_ID`).
4. Share the database with your integration.

Each run creates one Notion page with: ticker, date range, model, action, confidence, token usage, and the full markdown report.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Original project by [virattt](https://github.com/virattt/ai-hedge-fund).
