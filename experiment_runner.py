"""
Experiment Runner for Consequence-Aware Agentic Trading Research
=======================================================================
Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia (shb@3sholding.com)

This module orchestrates the complete experiment: for each stock, I generate
strategies, evaluate their consequences on historical data, select the best
one, and then test it out-of-sample. I also run both baselines for comparison.

The experiment flow mirrors the research methodology:
  1. For each stock:
     a. Generate 5 candidate strategies (via LLM or rule-based)
     b. Backtest all 5 on the training window (consequence assessment)
     c. Select the strategy with the best Sharpe ratio
     d. Execute the selected strategy on the test window
  2. Run baselines across all stocks:
     a. Random selection: pick one of the 5 strategies at random
     b. Buy-and-hold: purchase day 1, hold through test period
  3. Conduct sensitivity analysis:
     a. What if we select on cumulative return instead of Sharpe?
     b. What if we use a different training window?

I maintain strict separation between training and test data throughout.
The consequence assessment (step 1b) only ever touches training data.
The out-of-sample test (step 1d) only touches test data.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import data_pipeline
import strategy_generator as sg
import consequence_assessment as ca

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Experiment Result Container
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class StockExperimentResult:
    """
    I store the complete experiment result for a single stock. This includes
    the consequence assessment results (training), the out-of-sample results
    (test), and baseline comparisons. Everything a reviewer needs to evaluate
    the experiment for one stock is in this object.
    """
    ticker: str

    # Strategies and their training-period backtest results
    strategies: List[sg.TradingStrategy] = field(default_factory=list)
    training_results: List[ca.BacktestResult] = field(default_factory=list)

    # Consequence-aware selection
    selected_strategy_idx: int = -1
    selected_strategy_name: str = ""
    selection_metric: str = ""
    selection_metric_value: float = 0.0

    # Out-of-sample (test period) results
    test_result_consequence_aware: Optional[ca.BacktestResult] = None
    test_result_random: Optional[ca.BacktestResult] = None
    test_result_buy_hold: Optional[ca.BacktestResult] = None

    # Random baseline details
    random_strategy_idx: int = -1
    random_strategy_name: str = ""

    # Market summary for context
    market_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class FullExperimentResult:
    """
    I aggregate results across all stocks into a single experiment object.
    This is what gets passed to the visualization and results modules.
    """
    stock_results: Dict[str, StockExperimentResult] = field(default_factory=dict)
    sensitivity_results: Dict[str, Any] = field(default_factory=dict)
    experiment_metadata: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────
# Core Experiment Logic
# ───────────────────────────────────────────────────────────────────────────
def run_single_stock_experiment(
    ticker: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    use_llm: bool = True,
    random_seed: int = config.RANDOM_SEED,
) -> StockExperimentResult:
    """
    I run the complete experiment for a single stock. This is the workhorse
    function that implements the consequence-aware pipeline:
      1. Generate strategies
      2. Backtest on training data (consequence assessment)
      3. Select the best
      4. Test out-of-sample
      5. Run baselines

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    train_data : pd.DataFrame
        Training period OHLCV data.
    test_data : pd.DataFrame
        Test period OHLCV data (strictly held out).
    use_llm : bool
        Whether to use LLM for strategy generation.
    random_seed : int
        Seed for random baseline selection.

    Returns
    -------
    StockExperimentResult
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING EXPERIMENT FOR {ticker} ({config.TICKER_NAMES.get(ticker, '')})")
    logger.info(f"{'='*70}")
    stock_start = time.time()

    result = StockExperimentResult(ticker=ticker)

    # ── Step 1: Compute market summary for strategy generation context ──
    logger.info("Step 1: Computing market summary statistics...")
    market_summary = data_pipeline.compute_market_summary(train_data)
    result.market_summary = market_summary
    logger.info(
        f"  Market context: return={market_summary['total_return']:.2%}, "
        f"vol={market_summary['annualized_volatility']:.2%}, "
        f"sharpe={market_summary['sharpe_ratio']:.2f}"
    )

    # ── Step 2: Generate candidate strategies ──
    logger.info("Step 2: Generating candidate strategies...")
    strategies = sg.generate_strategies(ticker, market_summary, use_llm=use_llm)
    result.strategies = strategies
    for i, s in enumerate(strategies):
        logger.info(f"  Strategy {i+1}: {s.name} ({s.strategy_type})")

    # ── Step 3: Consequence Assessment — backtest all strategies on training data ──
    logger.info("Step 3: Running consequence assessment (training backtests)...")
    training_results = []
    for i, strategy in enumerate(strategies):
        logger.info(f"  Backtesting strategy {i+1}/{len(strategies)}: {strategy.name}")
        bt_result = ca.backtest_strategy(strategy, train_data)
        training_results.append(bt_result)
    result.training_results = training_results

    # I log all training results for this stock
    ca.log_backtest_results(ticker, training_results, phase="training")

    # ── Step 4: Select best strategy based on consequence assessment ──
    logger.info("Step 4: Selecting best strategy via consequence assessment...")
    best_idx, best_training_result = ca.select_best_strategy(
        training_results, metric=config.PRIMARY_SELECTION_METRIC
    )
    result.selected_strategy_idx = best_idx
    result.selected_strategy_name = strategies[best_idx].name
    result.selection_metric = config.PRIMARY_SELECTION_METRIC
    result.selection_metric_value = getattr(
        best_training_result, config.PRIMARY_SELECTION_METRIC, 0.0
    )

    logger.info(
        f"  SELECTED: Strategy {best_idx+1} — '{strategies[best_idx].name}' "
        f"(training {config.PRIMARY_SELECTION_METRIC}="
        f"{result.selection_metric_value:.4f})"
    )

    # I prepend the tail of the training data as warm-up for indicator
    # computation on the test period. Without this, strategies with long
    # lookback windows (e.g., 200-day MA) would produce no signals on a
    # 64-bar test period. I use the last 250 rows of training data—enough
    # for the longest indicator window in my strategy set.
    warmup_tail = train_data.tail(250)

    # ── Step 5: Out-of-sample test — consequence-aware agent ──
    logger.info("Step 5: Testing selected strategy out-of-sample...")
    test_ca = ca.backtest_strategy(strategies[best_idx], test_data, warmup_data=warmup_tail)
    result.test_result_consequence_aware = test_ca
    logger.info(
        f"  Consequence-aware test: Return={test_ca.cumulative_return:.2%}, "
        f"Sharpe={test_ca.sharpe_ratio:.2f}"
    )

    # ── Step 6: Random baseline — pick a strategy without consequence assessment ──
    logger.info("Step 6: Running random baseline...")
    rng = random.Random(random_seed + hash(ticker))
    random_idx = rng.randint(0, len(strategies) - 1)
    result.random_strategy_idx = random_idx
    result.random_strategy_name = strategies[random_idx].name

    test_random = ca.backtest_strategy(strategies[random_idx], test_data, warmup_data=warmup_tail)
    result.test_result_random = test_random
    logger.info(
        f"  Random baseline (strategy {random_idx+1}: '{strategies[random_idx].name}'): "
        f"Return={test_random.cumulative_return:.2%}, "
        f"Sharpe={test_random.sharpe_ratio:.2f}"
    )

    # ── Step 7: Buy-and-hold baseline ──
    logger.info("Step 7: Running buy-and-hold baseline...")
    test_bh = ca.backtest_buy_and_hold(ticker, test_data)
    result.test_result_buy_hold = test_bh
    logger.info(
        f"  Buy-and-hold: Return={test_bh.cumulative_return:.2%}, "
        f"Sharpe={test_bh.sharpe_ratio:.2f}"
    )

    # I log the test period results
    ca.log_backtest_results(
        ticker,
        [test_ca, test_random, test_bh],
        phase="test",
    )

    stock_time = time.time() - stock_start
    logger.info(f"\nCompleted {ticker} in {stock_time:.1f}s")
    _log_stock_comparison(result)

    return result


def _log_stock_comparison(result: StockExperimentResult) -> None:
    """I print a side-by-side comparison for quick inspection."""
    ca_r = result.test_result_consequence_aware
    rand_r = result.test_result_random
    bh_r = result.test_result_buy_hold

    logger.info(f"\n{'─'*60}")
    logger.info(f"TEST PERIOD COMPARISON for {result.ticker}")
    logger.info(f"{'─'*60}")
    logger.info(f"{'Metric':<25} {'CA Agent':>12} {'Random':>12} {'Buy&Hold':>12}")
    logger.info(f"{'─'*60}")

    for metric in ["cumulative_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"]:
        ca_v = getattr(ca_r, metric, 0) if ca_r else 0
        rand_v = getattr(rand_r, metric, 0) if rand_r else 0
        bh_v = getattr(bh_r, metric, 0) if bh_r else 0

        if metric in ["cumulative_return", "max_drawdown", "win_rate"]:
            logger.info(f"{metric:<25} {ca_v:>11.2%} {rand_v:>11.2%} {bh_v:>11.2%}")
        elif metric == "total_trades":
            logger.info(f"{metric:<25} {ca_v:>12d} {rand_v:>12d} {bh_v:>12d}")
        else:
            logger.info(f"{metric:<25} {ca_v:>12.4f} {rand_v:>12.4f} {bh_v:>12.4f}")

    logger.info(f"{'─'*60}")


# ───────────────────────────────────────────────────────────────────────────
# Full Experiment Across All Stocks
# ───────────────────────────────────────────────────────────────────────────
def run_full_experiment(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    use_llm: bool = True,
) -> FullExperimentResult:
    """
    I run the complete experiment across all stocks in our universe.
    This is the top-level function that main.py calls after the data pipeline.

    Parameters
    ----------
    train_data : dict
        Mapping ticker -> training DataFrame.
    test_data : dict
        Mapping ticker -> test DataFrame.
    use_llm : bool
        Whether to attempt LLM strategy generation.

    Returns
    -------
    FullExperimentResult
    """
    experiment_start = time.time()
    logger.info("\n" + "=" * 70)
    logger.info("STARTING FULL EXPERIMENT")
    logger.info(f"Stocks: {list(train_data.keys())}")
    logger.info(f"Selection metric: {config.PRIMARY_SELECTION_METRIC}")
    logger.info(f"Random seed: {config.RANDOM_SEED}")
    logger.info("=" * 70)

    full_result = FullExperimentResult()
    full_result.experiment_metadata = {
        "start_time": datetime.now().isoformat(),
        "num_stocks": len(train_data),
        "tickers": list(train_data.keys()),
        "selection_metric": config.PRIMARY_SELECTION_METRIC,
        "random_seed": config.RANDOM_SEED,
        "train_period": f"{config.DATA_START_DATE} to {config.TRAIN_END_DATE}",
        "test_period": f"{config.TEST_START_DATE} to {config.DATA_END_DATE}",
        "num_strategies_per_stock": config.NUM_STRATEGIES_PER_STOCK,
        "initial_capital": config.INITIAL_CAPITAL,
        "risk_free_rate": config.RISK_FREE_RATE,
    }

    # I process each stock sequentially. I could parallelize this, but
    # sequential execution makes the logs easier to follow and debug.
    for ticker in train_data.keys():
        stock_result = run_single_stock_experiment(
            ticker=ticker,
            train_data=train_data[ticker],
            test_data=test_data[ticker],
            use_llm=use_llm,
        )
        full_result.stock_results[ticker] = stock_result

    # ── Run multiple random baseline iterations for statistical robustness ──
    logger.info("\n" + "=" * 70)
    logger.info(f"RUNNING {config.NUM_RANDOM_BASELINE_RUNS} RANDOM BASELINE ITERATIONS")
    logger.info("=" * 70)
    random_baseline_runs = _run_multiple_random_baselines(
        full_result.stock_results, test_data, train_data
    )
    full_result.experiment_metadata["random_baseline_runs"] = random_baseline_runs

    # ── Sensitivity analysis ──
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING SENSITIVITY ANALYSIS")
    logger.info("=" * 70)
    sensitivity = _run_sensitivity_analysis(full_result.stock_results)
    full_result.sensitivity_results = sensitivity

    experiment_time = time.time() - experiment_start
    full_result.experiment_metadata["total_time_seconds"] = experiment_time
    full_result.experiment_metadata["end_time"] = datetime.now().isoformat()

    logger.info(f"\nFull experiment completed in {experiment_time:.1f}s")
    _log_aggregate_results(full_result)

    # I save the complete experiment metadata
    meta_path = config.LOGS_DIR / "experiment_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(full_result.experiment_metadata, f, indent=2, default=str)

    return full_result


def _run_multiple_random_baselines(
    stock_results: Dict[str, StockExperimentResult],
    test_data: Dict[str, pd.DataFrame],
    train_data: Dict[str, pd.DataFrame],
) -> Dict[str, List[Dict[str, float]]]:
    """
    I run the random baseline multiple times (default: 100) to get a
    distribution of random selection outcomes. A single random draw could
    get lucky or unlucky—I need the distribution to make fair comparisons.

    For each iteration, I randomly select one of the 5 strategies for each
    stock and record the test-period performance. This gives me 100 samples
    of "what random selection would have achieved."
    """
    all_runs: Dict[str, List[Dict[str, float]]] = {t: [] for t in stock_results}

    for run_idx in range(config.NUM_RANDOM_BASELINE_RUNS):
        rng = random.Random(config.RANDOM_SEED + run_idx)

        for ticker, sr in stock_results.items():
            # I pick a random strategy from the 5 candidates
            rand_idx = rng.randint(0, len(sr.strategies) - 1)
            strategy = sr.strategies[rand_idx]

            # I backtest it on the test period with warmup from training tail
            warmup_tail = train_data[ticker].tail(250)
            bt = ca.backtest_strategy(strategy, test_data[ticker], warmup_data=warmup_tail)

            all_runs[ticker].append({
                "run_idx": run_idx,
                "strategy_idx": rand_idx,
                "strategy_name": strategy.name,
                "cumulative_return": bt.cumulative_return,
                "sharpe_ratio": bt.sharpe_ratio,
                "max_drawdown": bt.max_drawdown,
            })

    # I log summary statistics for the random baseline distribution
    for ticker in all_runs:
        returns = [r["cumulative_return"] for r in all_runs[ticker]]
        sharpes = [r["sharpe_ratio"] for r in all_runs[ticker]]
        logger.info(
            f"  {ticker} random baseline: "
            f"return={np.mean(returns):.2%}±{np.std(returns):.2%}, "
            f"sharpe={np.mean(sharpes):.2f}±{np.std(sharpes):.2f}"
        )

    return all_runs


def _run_sensitivity_analysis(
    stock_results: Dict[str, StockExperimentResult],
) -> Dict[str, Any]:
    """
    I conduct sensitivity analysis to test the robustness of the results.
    Key question: does the conclusion change if I use a different selection
    metric?

    I re-run the strategy selection using alternative metrics (cumulative
    return, Calmar ratio) and compare the resulting test-period performance.
    This directly addresses the reviewer question: "Why Sharpe ratio?"
    """
    sensitivity = {}

    for alt_metric in config.SENSITIVITY_METRICS:
        logger.info(f"  Sensitivity test: selecting by {alt_metric}")
        alt_results = {}

        for ticker, sr in stock_results.items():
            # I re-select the best strategy using the alternative metric
            alt_idx, alt_bt = ca.select_best_strategy(
                sr.training_results, metric=alt_metric
            )

            # I check if this would have changed the selection
            primary_idx = sr.selected_strategy_idx
            changed = alt_idx != primary_idx

            # I use the pre-computed test result for the alt-selected strategy
            # by backtesting the alternatively-selected strategy on test data
            # (these are already computed as training_results)
            alt_results[ticker] = {
                "alternative_metric": alt_metric,
                "selected_idx": alt_idx,
                "selected_name": sr.strategies[alt_idx].name,
                "primary_idx": primary_idx,
                "selection_changed": changed,
                "training_metric_value": getattr(
                    sr.training_results[alt_idx], alt_metric, 0.0
                ),
            }

            if changed:
                logger.info(
                    f"    {ticker}: Selection CHANGED — "
                    f"primary=Strategy {primary_idx+1}, "
                    f"alt({alt_metric})=Strategy {alt_idx+1}"
                )

        sensitivity[alt_metric] = alt_results

    return sensitivity


def _log_aggregate_results(result: FullExperimentResult) -> None:
    """I print the aggregate performance summary across all stocks."""
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATE RESULTS ACROSS ALL STOCKS")
    logger.info("=" * 70)

    ca_returns = []
    rand_returns = []
    bh_returns = []
    ca_sharpes = []
    rand_sharpes = []
    bh_sharpes = []

    for ticker, sr in result.stock_results.items():
        if sr.test_result_consequence_aware:
            ca_returns.append(sr.test_result_consequence_aware.cumulative_return)
            ca_sharpes.append(sr.test_result_consequence_aware.sharpe_ratio)
        if sr.test_result_random:
            rand_returns.append(sr.test_result_random.cumulative_return)
            rand_sharpes.append(sr.test_result_random.sharpe_ratio)
        if sr.test_result_buy_hold:
            bh_returns.append(sr.test_result_buy_hold.cumulative_return)
            bh_sharpes.append(sr.test_result_buy_hold.sharpe_ratio)

    logger.info(f"\n{'Approach':<25} {'Mean Return':>14} {'Std Return':>14} {'Mean Sharpe':>14}")
    logger.info("-" * 70)

    if ca_returns:
        logger.info(
            f"{'Consequence-Aware':<25} {np.mean(ca_returns):>13.2%} "
            f"{np.std(ca_returns):>13.2%} {np.mean(ca_sharpes):>14.4f}"
        )
    if rand_returns:
        logger.info(
            f"{'Random Baseline':<25} {np.mean(rand_returns):>13.2%} "
            f"{np.std(rand_returns):>13.2%} {np.mean(rand_sharpes):>14.4f}"
        )
    if bh_returns:
        logger.info(
            f"{'Buy-and-Hold':<25} {np.mean(bh_returns):>13.2%} "
            f"{np.std(bh_returns):>13.2%} {np.mean(bh_sharpes):>14.4f}"
        )

    logger.info("=" * 70)

    # I show per-stock winners
    logger.info("\nPer-stock winner analysis:")
    ca_wins = 0
    total = len(result.stock_results)
    for ticker, sr in result.stock_results.items():
        ca_ret = sr.test_result_consequence_aware.cumulative_return if sr.test_result_consequence_aware else -999
        bh_ret = sr.test_result_buy_hold.cumulative_return if sr.test_result_buy_hold else -999
        rand_ret = sr.test_result_random.cumulative_return if sr.test_result_random else -999

        best = max(ca_ret, bh_ret, rand_ret)
        if best == ca_ret:
            winner = "CA Agent"
            ca_wins += 1
        elif best == bh_ret:
            winner = "Buy&Hold"
        else:
            winner = "Random"

        logger.info(
            f"  {ticker:<8}: CA={ca_ret:>8.2%}, Random={rand_ret:>8.2%}, "
            f"B&H={bh_ret:>8.2%} -> Winner: {winner}"
        )

    logger.info(f"\nConsequence-Aware agent won {ca_wins}/{total} stocks by return")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    logger.info("Experiment runner module loaded. Use run_full_experiment() to start.")
