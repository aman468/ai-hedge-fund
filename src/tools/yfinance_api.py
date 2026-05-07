"""yfinance adapter — same interface as src/tools/api.py.

Works with any Yahoo Finance symbol, including:
  NSE (India):  RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS
  BSE (India):  500325.BO, 532540.BO
  US:           AAPL, MSFT, NVDA (fallback when no FINANCIAL_DATASETS_API_KEY)
"""

import datetime
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance not installed. Run: poetry add yfinance")

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

_cache = get_cache()

# ── Row-name lookup tables ────────────────────────────────────────────────────
# Maps our field names → candidate row labels in yfinance statements.
# First match wins.

_INCOME_MAP: dict[str, list[str]] = {
    "revenue":              ["Total Revenue"],
    "net_income":           ["Net Income"],
    "operating_income":     ["Operating Income", "Total Operating Income As Reported"],
    "ebit":                 ["EBIT"],
    "ebitda":               ["EBITDA"],
    "interest_expense":     ["Interest Expense", "Interest Expense Non Operating"],
    "earnings_per_share":   ["Basic EPS", "Diluted EPS"],
    "outstanding_shares":   ["Diluted Average Shares", "Basic Average Shares"],
    "dividends_and_other_cash_distributions": ["Common Stock Dividends"],
}

_BALANCE_MAP: dict[str, list[str]] = {
    "total_assets":        ["Total Assets"],
    "total_liabilities":   ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "current_assets":      ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "total_debt":          ["Total Debt"],
    "cash_and_equivalents": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ],
}

_CASHFLOW_MAP: dict[str, list[str]] = {
    "free_cash_flow":              ["Free Cash Flow"],
    "capital_expenditure":         ["Capital Expenditure"],
    "depreciation_and_amortization": [
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
    ],
    # Net equity activity: negative = buyback, positive = issuance
    "issuance_or_purchase_of_equity_shares": [
        "Net Common Stock Issuance",
        "Repurchase Of Capital Stock",
        "Common Stock Issuance",
    ],
}

# Fields computed from multiple statements rather than a direct row lookup
_DERIVED = {"working_capital", "book_value_per_share"}


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _safe(val) -> float | None:
    """Convert to float, return None for NaN / inf / missing."""
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _find(df: pd.DataFrame | None, candidates: list[str]) -> pd.Series | None:
    """Return the first matching row from df by row-label candidates."""
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _col(series: pd.Series | None, idx: int = 0) -> float | None:
    """Safe value extraction from a Series at column index idx."""
    if series is None or len(series) <= idx:
        return None
    return _safe(series.iloc[idx])


def _ttm_flow(df: pd.DataFrame | None) -> pd.Series | None:
    """Sum the last 4 quarterly columns (flow statements: income / cash flow)."""
    if df is None or df.empty:
        return None
    cols = df.columns[: min(4, df.shape[1])]
    return df[cols].sum(axis=1)


def _latest_bs(df: pd.DataFrame | None) -> pd.Series | None:
    """Most recent quarterly balance sheet column."""
    if df is None or df.empty:
        return None
    return df.iloc[:, 0]


def _growth(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None or prev == 0:
        return None
    return (curr - prev) / abs(prev)


def _fetch_ticker_data(ticker: str):
    """Fetch and return all yfinance statement DataFrames for a ticker."""
    t = yf.Ticker(ticker)
    info = t.info or {}
    return (
        t,
        info,
        t.financials,           # annual income statement
        t.balance_sheet,        # annual balance sheet
        t.cashflow,             # annual cash flow
        t.quarterly_financials,
        t.quarterly_balance_sheet,
        t.quarterly_cashflow,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str = None,
) -> list[Price]:
    cache_key = f"yf_{ticker}_{start_date}_{end_date}"
    if cached := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

    try:
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
    except Exception as exc:
        logger.warning("yfinance price fetch failed for %s: %s", ticker, exc)
        return []

    if hist.empty:
        return []

    prices = [
        Price(
            open=float(row["Open"]),
            close=float(row["Close"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            volume=int(row["Volume"]),
            time=ts.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        for ts, row in hist.iterrows()
    ]
    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    cache_key = f"yf_{ticker}_{period}_{end_date}_{limit}"
    if cached := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached]

    try:
        _, info, ann_fin, ann_bs, ann_cf, qtr_fin, qtr_bs, qtr_cf = _fetch_ticker_data(ticker)
    except Exception as exc:
        logger.warning("yfinance financial fetch failed for %s: %s", ticker, exc)
        return []

    currency = info.get("currency", "USD")

    def _build_ttm() -> FinancialMetrics:
        ttm_inc = _ttm_flow(qtr_fin) if (qtr_fin is not None and not qtr_fin.empty) else (ann_fin.iloc[:, 0] if (ann_fin is not None and not ann_fin.empty) else None)
        ttm_cf  = _ttm_flow(qtr_cf)  if (qtr_cf  is not None and not qtr_cf.empty)  else (ann_cf.iloc[:,  0] if (ann_cf  is not None and not ann_cf.empty)  else None)
        ttm_bs  = _latest_bs(qtr_bs) if (qtr_bs  is not None and not qtr_bs.empty)  else (ann_bs.iloc[:,  0] if (ann_bs  is not None and not ann_bs.empty)  else None)

        market_cap = _safe(info.get("marketCap"))
        shares     = _safe(info.get("sharesOutstanding"))
        fcf        = _safe(info.get("freeCashflow"))
        eps        = _safe(info.get("trailingEps"))
        bvps       = _safe(info.get("bookValue"))

        # Prior-year values for growth (annual col 1)
        rev_curr   = _safe(info.get("totalRevenue"))
        rev_prev   = _col(_find(ann_fin, ["Total Revenue"]), 1)
        ni_curr    = _safe(info.get("netIncomeToCommon"))
        ni_prev    = _col(_find(ann_fin, ["Net Income"]), 1)
        eps_prev   = _col(_find(ann_fin, ["Basic EPS", "Diluted EPS"]), 1)
        fcf_prev   = _col(_find(ann_cf,  ["Free Cash Flow"]), 1)
        oi_curr    = _col(_find(ann_fin, ["Operating Income"]), 0)
        oi_prev    = _col(_find(ann_fin, ["Operating Income"]), 1)
        ebitda_c   = _col(_find(ann_fin, ["EBITDA"]), 0)
        ebitda_p   = _col(_find(ann_fin, ["EBITDA"]), 1)

        # Book value growth: need equity per share for two periods
        eq_curr  = _col(_find(ann_bs, ["Stockholders Equity", "Common Stock Equity"]), 0)
        eq_prev  = _col(_find(ann_bs, ["Stockholders Equity", "Common Stock Equity"]), 1)
        sh_curr  = _col(_find(ann_fin, ["Diluted Average Shares", "Basic Average Shares"]), 0)
        sh_prev  = _col(_find(ann_fin, ["Diluted Average Shares", "Basic Average Shares"]), 1)
        bvps_calc = (eq_curr / sh_curr) if (eq_curr and sh_curr and sh_curr > 0) else None
        bvps_prev = (eq_prev / sh_prev) if (eq_prev and sh_prev and sh_prev > 0) else None

        # ROIC approximation: EBIT*(1-t) / invested_capital
        total_debt = _col(_find(ann_bs, ["Total Debt"]), 0)
        eq_book    = (market_cap / info.get("priceToBook")) if (market_cap and _safe(info.get("priceToBook"))) else eq_curr
        inv_cap    = ((eq_book or 0) + (total_debt or 0)) or None
        roic       = ((ebitda_c * 0.75) / inv_cap) if (ebitda_c and inv_cap and inv_cap > 0) else None

        fcf_yield    = (fcf / market_cap) if (fcf and market_cap and market_cap > 0) else None
        fcf_per_share = (fcf / shares)    if (fcf and shares and shares > 0)         else None

        return FinancialMetrics(
            ticker=ticker,
            report_period=datetime.datetime.now().strftime("%Y-%m-%d"),
            period="ttm",
            currency=currency,
            market_cap=market_cap,
            enterprise_value=_safe(info.get("enterpriseValue")),
            price_to_earnings_ratio=_safe(info.get("trailingPE")),
            price_to_book_ratio=_safe(info.get("priceToBook")),
            price_to_sales_ratio=_safe(info.get("priceToSalesTrailing12Months")),
            enterprise_value_to_ebitda_ratio=_safe(info.get("enterpriseToEbitda")),
            enterprise_value_to_revenue_ratio=_safe(info.get("enterpriseToRevenue")),
            free_cash_flow_yield=fcf_yield,
            peg_ratio=_safe(info.get("pegRatio")),
            gross_margin=_safe(info.get("grossMargins")),
            operating_margin=_safe(info.get("operatingMargins")),
            net_margin=_safe(info.get("profitMargins")),
            return_on_equity=_safe(info.get("returnOnEquity")),
            return_on_assets=_safe(info.get("returnOnAssets")),
            return_on_invested_capital=roic,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=_safe(info.get("currentRatio")),
            quick_ratio=_safe(info.get("quickRatio")),
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=_safe(info.get("debtToEquity")),
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=_safe(info.get("revenueGrowth")) or _growth(rev_curr, rev_prev),
            earnings_growth=_safe(info.get("earningsGrowth")) or _growth(ni_curr, ni_prev),
            book_value_growth=_growth(bvps_calc or bvps, bvps_prev),
            earnings_per_share_growth=_growth(eps, eps_prev),
            free_cash_flow_growth=_growth(fcf, fcf_prev),
            operating_income_growth=_growth(oi_curr, oi_prev),
            ebitda_growth=_growth(ebitda_c, ebitda_p),
            payout_ratio=_safe(info.get("payoutRatio")),
            earnings_per_share=eps,
            book_value_per_share=bvps_calc or bvps,
            free_cash_flow_per_share=fcf_per_share,
        )

    def _build_annual(col_idx: int) -> FinancialMetrics | None:
        if ann_fin is None or ann_fin.empty or ann_fin.shape[1] <= col_idx:
            return None
        date = ann_fin.columns[col_idx]
        report_period = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

        rev  = _col(_find(ann_fin, ["Total Revenue"]), col_idx)
        ni   = _col(_find(ann_fin, ["Net Income"]), col_idx)
        oi   = _col(_find(ann_fin, ["Operating Income"]), col_idx)
        ebitda = _col(_find(ann_fin, ["EBITDA"]), col_idx)
        ebit   = _col(_find(ann_fin, ["EBIT"]), col_idx)
        eps    = _col(_find(ann_fin, ["Basic EPS", "Diluted EPS"]), col_idx)
        shares = _col(_find(ann_fin, ["Diluted Average Shares", "Basic Average Shares"]), col_idx)
        int_exp = _col(_find(ann_fin, ["Interest Expense", "Interest Expense Non Operating"]), col_idx)

        assets  = _col(_find(ann_bs, ["Total Assets"]), col_idx)
        liab    = _col(_find(ann_bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"]), col_idx)
        cur_a   = _col(_find(ann_bs, ["Current Assets"]), col_idx)
        cur_l   = _col(_find(ann_bs, ["Current Liabilities"]), col_idx)
        debt    = _col(_find(ann_bs, ["Total Debt"]), col_idx)
        cash    = _col(_find(ann_bs, _BALANCE_MAP["cash_and_equivalents"]), col_idx)
        equity  = _col(_find(ann_bs, ["Stockholders Equity", "Common Stock Equity"]), col_idx)

        fcf      = _col(_find(ann_cf, ["Free Cash Flow"]), col_idx)

        # Prior year for growth rates
        p = col_idx + 1
        rev_p   = _col(_find(ann_fin, ["Total Revenue"]), p)
        ni_p    = _col(_find(ann_fin, ["Net Income"]), p)
        oi_p    = _col(_find(ann_fin, ["Operating Income"]), p)
        ebitda_p = _col(_find(ann_fin, ["EBITDA"]), p)
        eps_p   = _col(_find(ann_fin, ["Basic EPS", "Diluted EPS"]), p)
        fcf_p   = _col(_find(ann_cf,  ["Free Cash Flow"]), p)
        eq_p    = _col(_find(ann_bs,  ["Stockholders Equity", "Common Stock Equity"]), p)
        sh_p    = _col(_find(ann_fin, ["Diluted Average Shares", "Basic Average Shares"]), p)

        bvps      = (equity / shares) if (equity and shares and shares > 0) else None
        bvps_prev = (eq_p / sh_p)     if (eq_p   and sh_p   and sh_p   > 0) else None
        fcf_ps    = (fcf   / shares)   if (fcf    and shares and shares > 0) else None
        roe       = (ni    / equity)   if (ni     and equity and equity != 0) else None
        roa       = (ni    / assets)   if (ni     and assets and assets != 0) else None
        d_e       = (debt  / equity)   if (debt   and equity and equity != 0) else None
        d_a       = (debt  / assets)   if (debt   and assets and assets != 0) else None
        net_mg    = (ni    / rev)      if (ni     and rev    and rev    != 0) else None
        op_mg     = (oi    / rev)      if (oi     and rev    and rev    != 0) else None
        cur_ratio = (cur_a / cur_l)    if (cur_a  and cur_l  and cur_l  != 0) else None
        cash_ratio = (cash / cur_l)    if (cash   and cur_l  and cur_l  != 0) else None
        at        = (rev   / assets)   if (rev    and assets and assets != 0) else None
        int_cov   = (ebit  / abs(int_exp)) if (ebit and int_exp and int_exp != 0) else None

        return FinancialMetrics(
            ticker=ticker,
            report_period=report_period,
            period="annual",
            currency=currency,
            market_cap=None,
            enterprise_value=None,
            price_to_earnings_ratio=None,
            price_to_book_ratio=None,
            price_to_sales_ratio=None,
            enterprise_value_to_ebitda_ratio=None,
            enterprise_value_to_revenue_ratio=None,
            free_cash_flow_yield=None,
            peg_ratio=None,
            gross_margin=None,
            operating_margin=op_mg,
            net_margin=net_mg,
            return_on_equity=roe,
            return_on_assets=roa,
            return_on_invested_capital=None,
            asset_turnover=at,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=cur_ratio,
            quick_ratio=None,
            cash_ratio=cash_ratio,
            operating_cash_flow_ratio=None,
            debt_to_equity=d_e,
            debt_to_assets=d_a,
            interest_coverage=int_cov,
            revenue_growth=_growth(rev, rev_p),
            earnings_growth=_growth(ni, ni_p),
            book_value_growth=_growth(bvps, bvps_prev),
            earnings_per_share_growth=_growth(eps, eps_p),
            free_cash_flow_growth=_growth(fcf, fcf_p),
            operating_income_growth=_growth(oi, oi_p),
            ebitda_growth=_growth(ebitda, ebitda_p),
            payout_ratio=None,
            earnings_per_share=eps,
            book_value_per_share=bvps,
            free_cash_flow_per_share=fcf_ps,
        )

    metrics: list[FinancialMetrics] = []

    if period == "ttm":
        metrics.append(_build_ttm())
        n_ann = ann_fin.shape[1] if (ann_fin is not None and not ann_fin.empty) else 0
        for i in range(min(limit - 1, n_ann)):
            m = _build_annual(i)
            if m:
                metrics.append(m)
    else:
        n_ann = ann_fin.shape[1] if (ann_fin is not None and not ann_fin.empty) else 0
        for i in range(min(limit, n_ann)):
            m = _build_annual(i)
            if m:
                metrics.append(m)

    if not metrics:
        return []

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
    return metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    try:
        _, info, ann_fin, ann_bs, ann_cf, qtr_fin, qtr_bs, qtr_cf = _fetch_ticker_data(ticker)
    except Exception as exc:
        logger.warning("yfinance search_line_items failed for %s: %s", ticker, exc)
        return []

    currency = info.get("currency", "USD")

    def _lookup_field(
        field: str,
        inc: pd.Series | None,
        bs: pd.Series | None,
        cf: pd.Series | None,
        shares_out: float | None = None,
    ) -> float | None:
        if field in _INCOME_MAP and inc is not None:
            for row_name in _INCOME_MAP[field]:
                if row_name in (inc.index if hasattr(inc, "index") else []):
                    return _safe(inc[row_name])
        if field in _BALANCE_MAP and bs is not None:
            for row_name in _BALANCE_MAP[field]:
                if row_name in (bs.index if hasattr(bs, "index") else []):
                    return _safe(bs[row_name])
        if field in _CASHFLOW_MAP and cf is not None:
            for row_name in _CASHFLOW_MAP[field]:
                if row_name in (cf.index if hasattr(cf, "index") else []):
                    return _safe(cf[row_name])
        if field == "working_capital" and bs is not None:
            cur_a = next((_safe(bs[r]) for r in _BALANCE_MAP["current_assets"]   if r in bs.index), None)
            cur_l = next((_safe(bs[r]) for r in _BALANCE_MAP["current_liabilities"] if r in bs.index), None)
            return (cur_a - cur_l) if (cur_a is not None and cur_l is not None) else None
        if field == "book_value_per_share" and bs is not None:
            eq = next((_safe(bs[r]) for r in ["Stockholders Equity", "Common Stock Equity"] if r in bs.index), None)
            sh = shares_out
            if eq is not None and sh and sh > 0:
                return eq / sh
            # fallback to info
            return _safe(info.get("bookValue"))
        return None

    results: list[LineItem] = []

    if period == "ttm":
        has_qtr = qtr_fin is not None and not qtr_fin.empty
        ttm_inc = _ttm_flow(qtr_fin) if has_qtr else (_latest_bs(ann_fin) if (ann_fin is not None and not ann_fin.empty) else None)
        ttm_cf  = _ttm_flow(qtr_cf)  if (qtr_cf  is not None and not qtr_cf.empty)  else (_latest_bs(ann_cf) if (ann_cf is not None and not ann_cf.empty) else None)
        ttm_bs  = _latest_bs(qtr_bs) if (qtr_bs  is not None and not qtr_bs.empty)  else (_latest_bs(ann_bs) if (ann_bs is not None and not ann_bs.empty) else None)
        shares  = _safe(info.get("sharesOutstanding"))

        ttm_date = datetime.datetime.now().strftime("%Y-%m-%d")
        kwargs = {f: _lookup_field(f, ttm_inc, ttm_bs, ttm_cf, shares) for f in line_items}
        results.append(LineItem(ticker=ticker, report_period=ttm_date, period="ttm", currency=currency, **kwargs))

        # Fill remaining slots with annual periods
        n_ann = ann_fin.shape[1] if (ann_fin is not None and not ann_fin.empty) else 0
        for i in range(min(limit - 1, n_ann)):
            results.append(_build_annual_line_item(i, ticker, currency, line_items, ann_fin, ann_bs, ann_cf, info))

    else:
        n_ann = ann_fin.shape[1] if (ann_fin is not None and not ann_fin.empty) else 0
        for i in range(min(limit, n_ann)):
            results.append(_build_annual_line_item(i, ticker, currency, line_items, ann_fin, ann_bs, ann_cf, info))

    return [r for r in results if r is not None]


def _build_annual_line_item(
    col_idx: int,
    ticker: str,
    currency: str,
    line_items: list[str],
    ann_fin: pd.DataFrame | None,
    ann_bs: pd.DataFrame | None,
    ann_cf: pd.DataFrame | None,
    info: dict,
) -> LineItem | None:
    if ann_fin is None or ann_fin.empty or ann_fin.shape[1] <= col_idx:
        return None

    date = ann_fin.columns[col_idx]
    report_period = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

    inc = ann_fin.iloc[:, col_idx]
    bs  = ann_bs.iloc[:,  col_idx] if (ann_bs is not None and not ann_bs.empty and ann_bs.shape[1] > col_idx)  else None
    cf  = ann_cf.iloc[:,  col_idx] if (ann_cf is not None and not ann_cf.empty and ann_cf.shape[1] > col_idx)  else None

    # shares for this period (from income statement)
    shares = next(
        (_safe(inc[r]) for r in ["Diluted Average Shares", "Basic Average Shares"] if r in inc.index),
        _safe(info.get("sharesOutstanding")),
    )

    kwargs: dict = {}
    for field in line_items:
        val = None
        if field in _INCOME_MAP:
            val = next((_safe(inc[r]) for r in _INCOME_MAP[field] if r in inc.index), None)
        elif field in _BALANCE_MAP and bs is not None:
            val = next((_safe(bs[r]) for r in _BALANCE_MAP[field] if r in bs.index), None)
        elif field in _CASHFLOW_MAP and cf is not None:
            val = next((_safe(cf[r]) for r in _CASHFLOW_MAP[field] if r in cf.index), None)
        elif field == "working_capital" and bs is not None:
            cur_a = next((_safe(bs[r]) for r in _BALANCE_MAP["current_assets"]      if r in bs.index), None)
            cur_l = next((_safe(bs[r]) for r in _BALANCE_MAP["current_liabilities"] if r in bs.index), None)
            val   = (cur_a - cur_l) if (cur_a is not None and cur_l is not None) else None
        elif field == "book_value_per_share":
            eq  = next((_safe(bs[r]) for r in ["Stockholders Equity", "Common Stock Equity"] if bs is not None and r in bs.index), None)
            val = (eq / shares) if (eq is not None and shares and shares > 0) else None
        kwargs[field] = val  # always set (may be None) so agents can access the attribute

    return LineItem(ticker=ticker, report_period=report_period, period="annual", currency=currency, **kwargs)


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    try:
        df = yf.Ticker(ticker).insider_transactions
        if df is None or df.empty:
            return []

        # Normalize date column (yfinance uses various column names)
        date_col = next((c for c in ["Start Date", "Date", "startDate"] if c in df.columns), None)
        if date_col is None:
            return []

        df = df.copy()
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_date"])
        df = df[df["_date"] <= pd.Timestamp(end_date)]
        if start_date:
            df = df[df["_date"] >= pd.Timestamp(start_date)]
        df = df.head(limit)

        trades = []
        for _, row in df.iterrows():
            date_str = row["_date"].strftime("%Y-%m-%dT%H:%M:%S")
            name_col  = next((c for c in ["Insider Trading", "Insider", "Name"] if c in row.index), None)
            title_col = next((c for c in ["Position", "Title"] if c in row.index), None)
            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=str(row[name_col]) if name_col else None,
                title=str(row[title_col]) if title_col else None,
                is_board_director=None,
                transaction_date=date_str,
                transaction_shares=_safe(row.get("Shares")),
                transaction_price_per_share=None,
                transaction_value=_safe(row.get("Value")),
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=None,
                filing_date=date_str,
            ))
        return trades
    except Exception as exc:
        logger.warning("yfinance insider trades failed for %s: %s", ticker, exc)
        return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    try:
        news_raw = yf.Ticker(ticker).news or []
        result: list[CompanyNews] = []

        for item in news_raw:
            # yfinance ≥0.2.x wraps content in a "content" dict; handle both formats
            content = item.get("content", item)
            ts = (
                content.get("pubDate")
                or item.get("providerPublishTime")
                or content.get("providerPublishTime")
                or 0
            )
            if isinstance(ts, (int, float)):
                date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")
            else:
                # Already a string
                date_str = str(ts)[:19]

            if date_str[:10] > end_date:
                continue
            if start_date and date_str[:10] < start_date:
                continue

            title   = content.get("title") or item.get("title", "")
            provider = content.get("provider") or {}
            source   = (provider.get("displayName") if isinstance(provider, dict) else str(provider)) or "Yahoo Finance"
            url_obj  = content.get("canonicalUrl") or {}
            url      = (url_obj.get("url") if isinstance(url_obj, dict) else str(url_obj)) or item.get("link", "")

            result.append(CompanyNews(
                ticker=ticker,
                title=title,
                author=None,
                source=source,
                date=date_str,
                url=url,
                sentiment=None,
            ))

            if len(result) >= limit:
                break

        return result
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        return []


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    try:
        return _safe(yf.Ticker(ticker).info.get("marketCap"))
    except Exception as exc:
        logger.warning("yfinance market_cap failed for %s: %s", ticker, exc)
        return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert list of Price objects to a DataFrame (format-independent)."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str = None,
) -> pd.DataFrame:
    return prices_to_df(get_prices(ticker, start_date, end_date))
