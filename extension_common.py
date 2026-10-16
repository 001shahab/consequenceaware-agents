"""
extension_common.py
===================================================================
Shared utilities for the CA2I revision experiments (reviewer response).

Everything here is deterministic and runs WITHOUT network or LLM access:
it reloads the exact data CSVs and LLM-generated strategy JSON files that
were produced in the original experiment, and re-runs the *same* backtesting
engine (consequence_assessment.backtest_strategy). This guarantees that every
number reported in the revised paper is reproducible by a reviewer offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
import consequence_assessment as ca
from strategy_generator import TradingStrategy

DATA_DIR = Path(config.DATA_DIR)
LOGS_DIR = Path(config.LOGS_DIR)
FIG_DIR = Path(config.FIGURES_DIR)

ANN = config.TRADING_DAYS_PER_YEAR          # 252
RF_ANNUAL = config.RISK_FREE_RATE           # 0.02


# ───────────────────────────────────────────────────────────────────────────
# Data / strategy loaders
# ───────────────────────────────────────────────────────────────────────────
def load_prices(ticker: str, phase: str) -> pd.DataFrame:
    """Load a saved OHLCV CSV (phase in {train, test, full}) as a DatetimeIndexed
    DataFrame, exactly as the backtester expects (uses the 'Close' column)."""
    path = DATA_DIR / f"{ticker}_{phase}.csv"
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.sort_index()


def load_strategies(ticker: str) -> List[TradingStrategy]:
    """Re-hydrate the saved (LLM-generated) candidate strategies for a ticker."""
    with open(LOGS_DIR / f"strategies_{ticker}.json") as f:
        blob = json.load(f)
    out = []
    for s in blob["strategies"]:
        out.append(TradingStrategy(
            name=s["name"],
            strategy_type=s["strategy_type"],
            description=s.get("description", ""),
            parameters=s["parameters"],
            entry_rule=s.get("entry_rule", ""),
            exit_rule=s.get("exit_rule", ""),
            reasoning=s.get("reasoning", ""),
            ticker=ticker,
            generation_method=s.get("generation_method", "llm"),
        ))
    return out


def warmup_tail(train: pd.DataFrame, n: int = 250) -> pd.DataFrame:
    """Replicate experiment_runner: prepend the last n training bars as warmup
    so long-lookback indicators produce signals on the short test window."""
    return train.tail(n)


# ───────────────────────────────────────────────────────────────────────────
# Backtest wrappers returning daily-return series (needed for portfolio
# baselines, transaction costs, and bootstrap)
# ───────────────────────────────────────────────────────────────────────────
def backtest(strategy: TradingStrategy, data: pd.DataFrame,
             warmup: pd.DataFrame | None = None) -> ca.BacktestResult:
    return ca.backtest_strategy(strategy, data, warmup_data=warmup)


def daily_returns(result: ca.BacktestResult) -> pd.Series:
    if result.daily_returns is not None:
        return result.daily_returns
    return result.equity_curve.pct_change().dropna()


def sharpe_from_returns(r: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    """Annualised Sharpe ratio identical to the engine's definition."""
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() <= 1e-12:
        return 0.0
    ann_vol = r.std() * np.sqrt(ANN)
    return float((r.mean() * ANN - rf_annual) / ann_vol)


def max_drawdown_from_equity(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    return float(dd.min())


# ───────────────────────────────────────────────────────────────────────────
# Candidate-set helpers
# ───────────────────────────────────────────────────────────────────────────
def evaluate_candidates(ticker: str) -> Dict:
    """Re-run the full per-asset pipeline deterministically and return, for the
    given ticker, the training and test BacktestResult for every candidate
    strategy plus the buy-and-hold benchmark and aligned daily-return series.
    """
    train = load_prices(ticker, "train")
    test = load_prices(ticker, "test")
    wu = warmup_tail(train)
    strategies = load_strategies(ticker)

    train_res, test_res = [], []
    for s in strategies:
        train_res.append(backtest(s, train))
        test_res.append(backtest(s, test, warmup=wu))

    bh = ca.backtest_buy_and_hold(ticker, test)

    return {
        "ticker": ticker,
        "strategies": strategies,
        "train": train_res,
        "test": test_res,
        "bh": bh,
        "train_prices": train,
        "test_prices": test,
    }


def aligned_test_return_matrix(ev: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """Build a (days x K) matrix of candidate test daily returns aligned on the
    test calendar, used for portfolio baselines (ensemble, mean-variance,
    risk-parity)."""
    cols = {}
    for s, r in zip(ev["strategies"], ev["test"]):
        cols[s.name] = daily_returns(r)
    mat = pd.DataFrame(cols).dropna(how="all").fillna(0.0)
    return mat, list(cols.keys())


def aligned_train_return_matrix(ev: Dict) -> pd.DataFrame:
    cols = {}
    for s, r in zip(ev["strategies"], ev["train"]):
        cols[s.name] = daily_returns(r)
    return pd.DataFrame(cols).dropna(how="all").fillna(0.0)
