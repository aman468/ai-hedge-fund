"""Markdown report generator for hedge fund analysis runs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


_SIGNAL_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}


def save_report(
    result: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    selected_analysts: list[str],
    report_dir: str = "reports",
    token_usage: dict | None = None,
) -> str:
    """Format analysis result as markdown and write to report_dir. Returns file path."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ticker_slug = "_".join(t.replace(".", "-") for t in tickers)
    filepath = os.path.join(report_dir, f"{ticker_slug}_{timestamp}.md")

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Hedge Fund Analysis Report",
        "",
        "| | |",
        "|---|---|",
        f"| **Ticker(s)** | {', '.join(tickers)} |",
        f"| **Date Range** | {start_date} → {end_date} |",
        f"| **Model** | {model_name} |",
        f"| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| **Analysts run** | {', '.join(selected_analysts)} |",
        "",
        "---",
        "",
    ]

    decisions = result.get("decisions") or {}
    analyst_signals = result.get("analyst_signals") or {}

    # ── Trading decisions ─────────────────────────────────────────────────────
    lines += ["## Trading Decisions", ""]
    lines += ["| Ticker | Action | Quantity | Confidence | Reasoning |"]
    lines += ["|--------|--------|:--------:|:----------:|-----------|"]

    for ticker in tickers:
        dec = (decisions.get(ticker) or {}) if isinstance(decisions, dict) else {}
        action = str(dec.get("action", "—")).upper()
        qty = dec.get("quantity", 0)
        conf = dec.get("confidence", 0)
        reason = str(dec.get("reasoning", "")).replace("\n", " ")[:150]
        emoji = _SIGNAL_EMOJI.get(action, "⚪")
        lines.append(f"| {ticker} | {emoji} **{action}** | {qty} | {conf}% | {reason} |")

    lines += ["", "---", ""]

    # ── Per-agent analysis ────────────────────────────────────────────────────
    lines += ["## Agent Analysis", ""]

    for agent_key, agent_data in analyst_signals.items():
        display = (
            agent_key
            .replace("_agent", "")
            .replace("_analyst", " Analyst")
            .replace("_", " ")
            .title()
        )

        for ticker, sig in (agent_data or {}).items():
            signal = str(sig.get("signal", "—")).upper()
            confidence = sig.get("confidence", 0)
            reasoning = sig.get("reasoning", "")
            emoji = _SIGNAL_EMOJI.get(signal, "⚪")

            lines += [
                f"### {display} &nbsp; {emoji} {signal} &nbsp; ({confidence}%)",
                f"*Ticker: {ticker}*",
                "",
            ]

            if isinstance(reasoning, dict):
                for k, v in reasoning.items():
                    label = k.replace("_", " ").title()
                    if isinstance(v, dict):
                        sig_val = v.get("signal", "")
                        detail = v.get("details", "")
                        if not detail:
                            detail = json.dumps(v, indent=2)
                        header = f"**{label}**" + (f" — {sig_val.upper()}" if sig_val else "")
                        lines += [header, f"> {detail}", ""]
                    else:
                        lines.append(f"**{label}:** {v}  ")
                lines += [""]
            else:
                # Long string: preserve paragraphs
                for para in str(reasoning).split("\n"):
                    lines.append(para.strip())
                lines += [""]

            lines += ["---", ""]

    # ── Token usage ───────────────────────────────────────────────────────────
    if token_usage and token_usage.get("calls", 0) > 0:
        inp = token_usage["input"]
        out = token_usage["output"]
        calls = token_usage["calls"]
        lines += [
            "## Token Usage",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| LLM calls | {calls} |",
            f"| Input tokens | {inp:,} |",
            f"| Output tokens | {out:,} |",
            f"| Total tokens | {inp + out:,} |",
            "",
        ]

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return filepath
