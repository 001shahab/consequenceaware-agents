"""
extension_cache.py
===================================================================
Shared, cached candidate evaluations for the second-round revision
experiments (extension_08--10). Every candidate in every evaluation cell is
backtested once (training fit, walk-forward CV scores, test result) and the
result is pickled, so the downstream analyses are mutually consistent and
cheap to rerun.

Two settings are cached:
  * "deployment": the actual GPT-5.4 candidates (K = 5) on the held-out
    Aug--Oct 2025 window, one cell per equity (n = 10);
  * "rolling": the multi-regime evaluation of extension_07 (8 quarters x 10
    equities = 80 cells), K = 10 rule-based candidates, for each of the 10
    candidate-generation seeds.

Each cell stores, per candidate: the strategy specification and text, the
in-sample Sharpe, the CV lower-confidence-bound score and median fold
drawdown, and the training and test daily-return series.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import config
import consequence_assessment as ca
import data_pipeline as dp
import extension_common as ec
import extension_07_improved_selector as e7

warnings.filterwarnings("ignore")
CACHE = Path(config.DATA_DIR) / "revision2_cache.pkl"
K_ROLL = e7.ROLLING_K
SEEDS = list(range(e7.N_SEEDS))


def _cell(strategies: List, train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    wu = train.tail(e7.WARMUP_BARS)
    cands = []
    for s in strategies:
        tr = ca.backtest_strategy(s, train)
        te = ca.backtest_strategy(s, test, warmup_data=wu)
        cv = e7.cv_scores(s, train)
        cands.append({
            "name": s.name, "type": s.strategy_type, "params": dict(s.parameters),
            "text": " ".join([s.description, s.entry_rule, s.exit_rule, s.reasoning]),
            "insample_sr": tr.sharpe_ratio, "insample_dd": tr.max_drawdown,
            "lcb": cv["lcb"], "cv_mean": cv["mean"], "cv_std": cv["std"], "cvdd": cv["dd"],
            "train_ret": ec.daily_returns(tr), "test_ret": ec.daily_returns(te),
            "test_trades": te.total_trades,
        })
    return {"cands": cands}


def build() -> Dict:
    out = {"deployment": {}, "rolling": {}}
    for t in config.TICKERS:
        train, test = ec.load_prices(t, "train"), ec.load_prices(t, "test")
        out["deployment"][t] = _cell(ec.load_strategies(t), train, test)
    full = {t: ec.load_prices(t, "full") for t in config.TICKERS}
    for seed in SEEDS:
        for w in e7.WINDOWS:
            for t in config.TICKERS:
                f = full[t]
                train = f[f.index <= pd.Timestamp(w["train_end"])]
                test = f[(f.index >= pd.Timestamp(w["test_start"])) &
                         (f.index <= pd.Timestamp(w["test_end"]))]
                if len(train) < e7.WARMUP_BARS + 40 or len(test) < 20:
                    continue
                pool = e7.make_pool(t, dp.compute_market_summary(train), K_ROLL, seed_offset=seed)
                out["rolling"][(seed, w["label"], t)] = _cell(pool, train, test)
        print(f"  cached seed {seed}")
    return out


def load() -> Dict:
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    data = build()
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    return data


def ensemble(rets: List[pd.Series]) -> pd.Series:
    """Equal-weight, daily-rebalanced combination; empty list = stay in cash."""
    if not rets:
        return pd.Series(dtype=float)
    return pd.DataFrame({i: r for i, r in enumerate(rets)}).dropna(how="all").fillna(0.0).mean(axis=1)


if __name__ == "__main__":
    load()
    print(f"Cache ready: {CACHE}")
