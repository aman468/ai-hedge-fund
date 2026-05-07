"""Push hedge fund analysis reports to Notion database."""

from __future__ import annotations

import os
from datetime import datetime

import httpx

_NOTION_VERSION = "2022-06-28"
_NOTION_BASE = "https://api.notion.com/v1"
_CHUNK = 1900  # Notion code-block character limit
_MAX_REPORT_CHARS = 19000  # cap so we don't exceed 100-block limit


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('NOTION_API_KEY', '')}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }


def push_to_notion(
    result: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    selected_analysts: list[str],
    report_path: str,
    token_usage: dict | None = None,
    report_content: str = "",
) -> str | None:
    """Create a Notion page in the analysis-runs database. Returns the page URL or None."""
    database_id = os.getenv("NOTION_DATABASE_ID", "")
    api_key = os.getenv("NOTION_API_KEY", "")
    if not database_id or not api_key or api_key == "your-notion-api-key":
        return None

    decisions = result.get("decisions") or {}

    actions = [str(v.get("action", "")).upper() for v in decisions.values() if isinstance(v, dict)]
    unique = set(a for a in actions if a)
    overall_action = unique.pop() if len(unique) == 1 else ("MIXED" if unique else "HOLD")

    confidences = [v.get("confidence", 0) for v in decisions.values() if isinstance(v, dict)]
    avg_confidence = round(sum(confidences) / len(confidences)) if confidences else 0

    ticker_str = ", ".join(tickers)
    run_title = f"{ticker_str} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    inp = (token_usage or {}).get("input", 0)
    out = (token_usage or {}).get("output", 0)

    properties = {
        "Run": {"title": [{"text": {"content": run_title}}]},
        "Tickers": {"rich_text": [{"text": {"content": ticker_str}}]},
        "Start Date": {"date": {"start": start_date}},
        "End Date": {"date": {"start": end_date}},
        "Model": {"rich_text": [{"text": {"content": model_name}}]},
        "Analysts": {"rich_text": [{"text": {"content": ", ".join(selected_analysts)}}]},
        "Action": {"select": {"name": overall_action}},
        "Confidence": {"number": avg_confidence},
        "Input Tokens": {"number": inp},
        "Output Tokens": {"number": out},
        "Report Path": {"rich_text": [{"text": {"content": report_path}}]},
        "Status": {"select": {"name": "Success"}},
    }

    body = {
        "parent": {"database_id": database_id},
        "properties": properties,
        "children": _build_blocks(decisions, tickers, report_content),
    }

    try:
        resp = httpx.post(f"{_NOTION_BASE}/pages", headers=_headers(), json=body, timeout=30)
        resp.raise_for_status()
        return resp.json().get("url")
    except httpx.HTTPStatusError as exc:
        print(f"[Notion] HTTP {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:
        print(f"[Notion] Push failed: {exc}")
    return None


def _build_blocks(decisions: dict, tickers: list[str], report_content: str) -> list:
    blocks: list[dict] = []

    blocks.append(_h2("Trading Decisions"))
    for ticker in tickers:
        dec = (decisions.get(ticker) or {}) if isinstance(decisions, dict) else {}
        action = str(dec.get("action", "—")).upper()
        qty = dec.get("quantity", 0)
        conf = dec.get("confidence", 0)
        reason = str(dec.get("reasoning", ""))[:600]
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"text": {"content": f"{ticker}  "}, "annotations": {"bold": True, "code": True}},
                    {"text": {"content": f"{action}  |  qty {qty}  |  {conf}% confidence\n"}},
                    {"text": {"content": reason}, "annotations": {"color": "gray"}},
                ]
            },
        })

    if report_content:
        blocks.append(_h2("Full Report"))
        trimmed = report_content[:_MAX_REPORT_CHARS]
        for i in range(0, len(trimmed), _CHUNK):
            blocks.append({
                "object": "block",
                "type": "code",
                "code": {
                    "rich_text": [{"text": {"content": trimmed[i: i + _CHUNK]}}],
                    "language": "markdown",
                },
            })

    return blocks


def _h2(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": text}}]},
    }
