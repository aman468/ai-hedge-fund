import copy
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from colorama import Fore, Style, init

from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.utils.llm import reset_token_usage, print_token_summary, get_token_usage
from src.utils.report import save_report
from src.utils.notion_push import push_to_notion
from src.cli.input import parse_cli_inputs

import json

load_dotenv()
init(autoreset=True)

_MAX_PARALLEL_ANALYSTS = 8


def parse_hedge_fund_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


def _run_analyst(key: str, func, state: AgentState) -> dict:
    """Call one analyst agent on a deep-copied state snapshot."""
    return func(copy.deepcopy(state))


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
) -> dict:
    progress.start()
    reset_token_usage()

    try:
        analyst_nodes = get_analyst_nodes()
        analysts_to_run = selected_analysts or list(analyst_nodes.keys())

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": show_reasoning,
                "model_name": model_name,
                "model_provider": model_provider,
            },
        }

        # ── Phase 1: analysts run in parallel ─────────────────────────────────
        merged_messages = list(initial_state["messages"])
        merged_signals: dict = {}

        workers = min(len(analysts_to_run), _MAX_PARALLEL_ANALYSTS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_analyst, key, analyst_nodes[key][1], initial_state): key
                for key in analysts_to_run
                if key in analyst_nodes
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    out = future.result()
                    merged_messages.extend(out.get("messages", []))
                    merged_signals.update(out.get("data", {}).get("analyst_signals", {}))
                except Exception as exc:
                    print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} {key} failed: {exc}")

        # ── Phase 2: risk management ───────────────────────────────────────────
        risk_state: AgentState = {
            "messages": merged_messages,
            "data": {**initial_state["data"], "analyst_signals": merged_signals},
            "metadata": initial_state["metadata"],
        }
        risk_out = risk_management_agent(risk_state)
        merged_messages = merged_messages + risk_out.get("messages", [])
        merged_signals.update(risk_out.get("data", {}).get("analyst_signals", {}))

        # ── Phase 3: portfolio manager ─────────────────────────────────────────
        pm_state: AgentState = {
            "messages": merged_messages,
            "data": {**initial_state["data"], "analyst_signals": merged_signals},
            "metadata": initial_state["metadata"],
        }
        pm_out = portfolio_management_agent(pm_state)
        final_messages = merged_messages + pm_out.get("messages", [])

        return {
            "decisions": parse_hedge_fund_response(final_messages[-1].content),
            "analyst_signals": merged_signals,
        }

    finally:
        progress.stop()
        print_token_summary(model_name)


if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run the hedge fund trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=True,
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers
    portfolio = {
        "cash": inputs.initial_cash,
        "margin_requirement": inputs.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers},
    }

    result = run_hedge_fund(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        portfolio=portfolio,
        show_reasoning=inputs.show_reasoning,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
    )
    print_trading_output(result)

    token_usage = get_token_usage()
    report_path = save_report(
        result=result,
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        model_name=inputs.model_name,
        selected_analysts=inputs.selected_analysts,
        report_dir=inputs.report_dir,
        token_usage=token_usage,
    )
    print(f"\nReport saved  → {report_path}")

    # ── Push to Notion ─────────────────────────────────────────────────────────
    try:
        with open(report_path, encoding="utf-8") as fh:
            report_content = fh.read()
    except OSError:
        report_content = ""

    notion_url = push_to_notion(
        result=result,
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        model_name=inputs.model_name,
        selected_analysts=inputs.selected_analysts,
        report_path=report_path,
        token_usage=token_usage,
        report_content=report_content,
    )
    if notion_url:
        print(f"Notion page   → {notion_url}")
    else:
        print(f"{Fore.YELLOW}[Notion] Skipped — set NOTION_API_KEY in .env to enable{Style.RESET_ALL}")
