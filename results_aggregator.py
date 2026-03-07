"""
Results Aggregation Module for Consequence-Aware Agentic Trading Research
=======================================================================
Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia (shb@3sholding.com)

I aggregate all experimental results into structured CSV files and compute
statistical significance tests. This module produces the tables and numbers
that go directly into the paper.

My statistical philosophy: I report everything, favorable or not. If the
consequence-aware agent underperforms, the t-test will show it. I do NOT
cherry-pick metrics or windows to make the results look better. That would
be scientific misconduct, and it would also make the paper less interesting—
negative results with honest analysis are publishable and valuable.

Outputs:
  1. Detailed results CSV (one row per stock × strategy)
  2. Summary statistics CSV (aggregate means, stds)
  3. Statistical significance tests (paired t-tests)
  4. Sensitivity analysis results
  5. Complete experiment log in JSON Lines format
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import config
from experiment_runner import FullExperimentResult, StockExperimentResult

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Detailed Results CSV
# ───────────────────────────────────────────────────────────────────────────
def generate_detailed_results_csv(
    result: FullExperimentResult,
    save_dir: Path = config.DATA_DIR,
) -> Path:
    """
    I generate a comprehensive CSV with one row per stock × strategy. This
    includes both training (backtest) and test period metrics, the strategy
    description, and whether it was selected by the consequence-aware agent.

    A reviewer can load this CSV into R/Python/Excel and independently verify
    every number I report in the paper.
    """
    logger.info("Generating detailed results CSV...")
    rows = []

    for ticker, sr in result.stock_results.items():
        for i, (strategy, train_result) in enumerate(
            zip(sr.strategies, sr.training_results)
        ):
            row = {
                "ticker": ticker,
                "company": config.TICKER_NAMES.get(ticker, ""),
                "strategy_number": i + 1,
                "strategy_name": strategy.name,
                "strategy_type": strategy.strategy_type,
                "strategy_description": strategy.description,
                "generation_method": strategy.generation_method,
                "was_selected": i == sr.selected_strategy_idx,
                "selection_metric": config.PRIMARY_SELECTION_METRIC,

                # Training period metrics (consequence assessment)
                "train_cumulative_return": train_result.cumulative_return,
                "train_annualized_return": train_result.annualized_return,
                "train_sharpe_ratio": train_result.sharpe_ratio,
                "train_max_drawdown": train_result.max_drawdown,
                "train_win_rate": train_result.win_rate,
                "train_calmar_ratio": train_result.calmar_ratio,
                "train_total_trades": train_result.total_trades,
                "train_profit_factor": train_result.profit_factor,
                "train_sortino_ratio": train_result.sortino_ratio,
                "train_volatility": train_result.volatility,

                # Strategy parameters (serialized for reference)
                "parameters_json": json.dumps(strategy.parameters),
            }

            # I add test period metrics only for the selected and random strategies
            if i == sr.selected_strategy_idx and sr.test_result_consequence_aware:
                row["test_return_ca"] = sr.test_result_consequence_aware.cumulative_return
                row["test_sharpe_ca"] = sr.test_result_consequence_aware.sharpe_ratio
                row["test_max_dd_ca"] = sr.test_result_consequence_aware.max_drawdown

            rows.append(row)

        # I add buy-and-hold as a separate row for completeness
        if sr.test_result_buy_hold:
            bh_row = {
                "ticker": ticker,
                "company": config.TICKER_NAMES.get(ticker, ""),
                "strategy_number": 0,
                "strategy_name": "Buy and Hold",
                "strategy_type": "buy_and_hold",
                "strategy_description": "Buy on day 1, hold through entire period",
                "generation_method": "baseline",
                "was_selected": False,
                "selection_metric": "N/A",
                "train_cumulative_return": None,
                "train_sharpe_ratio": None,
                "test_return_ca": sr.test_result_buy_hold.cumulative_return,
                "test_sharpe_ca": sr.test_result_buy_hold.sharpe_ratio,
                "test_max_dd_ca": sr.test_result_buy_hold.max_drawdown,
            }
            rows.append(bh_row)

    df = pd.DataFrame(rows)
    path = save_dir / "detailed_results.csv"
    df.to_csv(path, index=False)
    logger.info(f"  Saved detailed results CSV ({len(df)} rows) to {path}")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Summary Statistics
# ───────────────────────────────────────────────────────────────────────────
def generate_summary_statistics(
    result: FullExperimentResult,
    save_dir: Path = config.DATA_DIR,
) -> pd.DataFrame:
    """
    I compute aggregate summary statistics across all stocks for each approach.
    This is the core results table for the paper. I report mean, standard
    deviation, median, min, and max for the primary metrics.
    """
    logger.info("Computing summary statistics...")

    # I collect test-period metrics for each approach
    approaches = {
        "Consequence-Aware": [],
        "Random Selection": [],
        "Buy-and-Hold": [],
    }

    for ticker, sr in result.stock_results.items():
        if sr.test_result_consequence_aware:
            approaches["Consequence-Aware"].append({
                "ticker": ticker,
                "return": sr.test_result_consequence_aware.cumulative_return,
                "sharpe": sr.test_result_consequence_aware.sharpe_ratio,
                "max_dd": sr.test_result_consequence_aware.max_drawdown,
                "win_rate": sr.test_result_consequence_aware.win_rate,
                "calmar": sr.test_result_consequence_aware.calmar_ratio,
                "volatility": sr.test_result_consequence_aware.volatility,
                "sortino": sr.test_result_consequence_aware.sortino_ratio,
            })
        if sr.test_result_random:
            approaches["Random Selection"].append({
                "ticker": ticker,
                "return": sr.test_result_random.cumulative_return,
                "sharpe": sr.test_result_random.sharpe_ratio,
                "max_dd": sr.test_result_random.max_drawdown,
                "win_rate": sr.test_result_random.win_rate,
                "calmar": sr.test_result_random.calmar_ratio,
                "volatility": sr.test_result_random.volatility,
                "sortino": sr.test_result_random.sortino_ratio,
            })
        if sr.test_result_buy_hold:
            approaches["Buy-and-Hold"].append({
                "ticker": ticker,
                "return": sr.test_result_buy_hold.cumulative_return,
                "sharpe": sr.test_result_buy_hold.sharpe_ratio,
                "max_dd": sr.test_result_buy_hold.max_drawdown,
                "win_rate": sr.test_result_buy_hold.win_rate,
                "calmar": sr.test_result_buy_hold.calmar_ratio,
                "volatility": sr.test_result_buy_hold.volatility,
                "sortino": sr.test_result_buy_hold.sortino_ratio,
            })

    # I compute summary statistics for each approach
    summary_rows = []
    for approach_name, data_list in approaches.items():
        if not data_list:
            continue
        df_approach = pd.DataFrame(data_list)
        for metric in ["return", "sharpe", "max_dd", "win_rate", "calmar", "volatility", "sortino"]:
            values = df_approach[metric].values
            summary_rows.append({
                "approach": approach_name,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            })

    summary_df = pd.DataFrame(summary_rows)
    path = save_dir / "summary_statistics.csv"
    summary_df.to_csv(path, index=False)
    logger.info(f"  Saved summary statistics to {path}")

    # I also print a nicely formatted table to the log
    _print_summary_table(approaches)

    return summary_df


def _print_summary_table(approaches: Dict[str, List[Dict]]) -> None:
    """I log a human-readable summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS TABLE (for paper)")
    logger.info("=" * 80)

    header = f"{'Metric':<20}"
    for approach in approaches:
        header += f" {approach:>20}"
    logger.info(header)
    logger.info("-" * 80)

    metrics = ["return", "sharpe", "max_dd", "volatility"]
    metric_labels = {
        "return": "Cumulative Return",
        "sharpe": "Sharpe Ratio",
        "max_dd": "Max Drawdown",
        "volatility": "Annualized Vol",
    }

    for metric in metrics:
        row = f"{metric_labels.get(metric, metric):<20}"
        for approach_name, data_list in approaches.items():
            if data_list:
                values = [d[metric] for d in data_list]
                mean_v = np.mean(values)
                std_v = np.std(values)
                if metric in ["return", "max_dd", "volatility"]:
                    row += f" {mean_v:>8.2%}±{std_v:<8.2%}"
                else:
                    row += f" {mean_v:>8.3f}±{std_v:<8.3f}"
            else:
                row += f" {'N/A':>20}"
        logger.info(row)

    logger.info("=" * 80)


# ───────────────────────────────────────────────────────────────────────────
# Statistical Significance Tests
# ───────────────────────────────────────────────────────────────────────────
def run_statistical_tests(
    result: FullExperimentResult,
    save_dir: Path = config.DATA_DIR,
) -> Dict[str, Any]:
    """
    I run paired t-tests to assess whether the consequence-aware agent's
    performance is statistically significantly different from the baselines.

    I use paired tests because each stock is a natural pair: the same stock
    is traded by all three approaches, so we're comparing matched observations.

    I report two-tailed p-values because I don't want to assume a direction
    a priori. If the consequence-aware agent is worse, I want the test to
    detect that too.

    I also compute the 100-run random baseline statistics for a more robust
    comparison. A single random draw can get lucky, but the average over 100
    draws is a fair representation of "no consequence assessment."
    """
    logger.info("Running statistical significance tests...")
    test_results = {}

    # I collect paired observations
    ca_returns = []
    rand_returns = []
    bh_returns = []
    ca_sharpes = []
    rand_sharpes = []
    bh_sharpes = []
    tickers_order = []

    for ticker, sr in result.stock_results.items():
        tickers_order.append(ticker)
        ca_returns.append(
            sr.test_result_consequence_aware.cumulative_return if sr.test_result_consequence_aware else 0
        )
        rand_returns.append(
            sr.test_result_random.cumulative_return if sr.test_result_random else 0
        )
        bh_returns.append(
            sr.test_result_buy_hold.cumulative_return if sr.test_result_buy_hold else 0
        )
        ca_sharpes.append(
            sr.test_result_consequence_aware.sharpe_ratio if sr.test_result_consequence_aware else 0
        )
        rand_sharpes.append(
            sr.test_result_random.sharpe_ratio if sr.test_result_random else 0
        )
        bh_sharpes.append(
            sr.test_result_buy_hold.sharpe_ratio if sr.test_result_buy_hold else 0
        )

    ca_returns = np.array(ca_returns)
    rand_returns = np.array(rand_returns)
    bh_returns = np.array(bh_returns)
    ca_sharpes = np.array(ca_sharpes)
    rand_sharpes = np.array(rand_sharpes)
    bh_sharpes = np.array(bh_sharpes)

    # ── Test 1: CA vs Random Selection (Returns) ──
    if len(ca_returns) > 1:
        t_stat, p_value = stats.ttest_rel(ca_returns, rand_returns)
        test_results["ca_vs_random_return"] = {
            "test": "paired_t_test",
            "metric": "cumulative_return",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < config.SIGNIFICANCE_LEVEL,
            "ca_mean": float(np.mean(ca_returns)),
            "random_mean": float(np.mean(rand_returns)),
            "difference_mean": float(np.mean(ca_returns - rand_returns)),
            "n_observations": len(ca_returns),
        }
        logger.info(
            f"  CA vs Random (return): t={t_stat:.3f}, p={p_value:.4f} "
            f"{'*** SIGNIFICANT ***' if p_value < config.SIGNIFICANCE_LEVEL else '(not significant)'}"
        )

    # ── Test 2: CA vs Buy-and-Hold (Returns) ──
    if len(ca_returns) > 1:
        t_stat, p_value = stats.ttest_rel(ca_returns, bh_returns)
        test_results["ca_vs_buyhold_return"] = {
            "test": "paired_t_test",
            "metric": "cumulative_return",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < config.SIGNIFICANCE_LEVEL,
            "ca_mean": float(np.mean(ca_returns)),
            "buyhold_mean": float(np.mean(bh_returns)),
            "difference_mean": float(np.mean(ca_returns - bh_returns)),
            "n_observations": len(ca_returns),
        }
        logger.info(
            f"  CA vs Buy&Hold (return): t={t_stat:.3f}, p={p_value:.4f} "
            f"{'*** SIGNIFICANT ***' if p_value < config.SIGNIFICANCE_LEVEL else '(not significant)'}"
        )

    # ── Test 3: CA vs Random Selection (Sharpe) ──
    if len(ca_sharpes) > 1:
        t_stat, p_value = stats.ttest_rel(ca_sharpes, rand_sharpes)
        test_results["ca_vs_random_sharpe"] = {
            "test": "paired_t_test",
            "metric": "sharpe_ratio",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < config.SIGNIFICANCE_LEVEL,
            "ca_mean": float(np.mean(ca_sharpes)),
            "random_mean": float(np.mean(rand_sharpes)),
            "difference_mean": float(np.mean(ca_sharpes - rand_sharpes)),
            "n_observations": len(ca_sharpes),
        }
        logger.info(
            f"  CA vs Random (Sharpe): t={t_stat:.3f}, p={p_value:.4f} "
            f"{'*** SIGNIFICANT ***' if p_value < config.SIGNIFICANCE_LEVEL else '(not significant)'}"
        )

    # ── Test 4: CA vs Buy-and-Hold (Sharpe) ──
    if len(ca_sharpes) > 1:
        t_stat, p_value = stats.ttest_rel(ca_sharpes, bh_sharpes)
        test_results["ca_vs_buyhold_sharpe"] = {
            "test": "paired_t_test",
            "metric": "sharpe_ratio",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < config.SIGNIFICANCE_LEVEL,
            "ca_mean": float(np.mean(ca_sharpes)),
            "buyhold_mean": float(np.mean(bh_sharpes)),
            "difference_mean": float(np.mean(ca_sharpes - bh_sharpes)),
            "n_observations": len(ca_sharpes),
        }
        logger.info(
            f"  CA vs Buy&Hold (Sharpe): t={t_stat:.3f}, p={p_value:.4f} "
            f"{'*** SIGNIFICANT ***' if p_value < config.SIGNIFICANCE_LEVEL else '(not significant)'}"
        )

    # ── Test 5: CA vs Average Random Baseline (100 runs) ──
    random_runs = result.experiment_metadata.get("random_baseline_runs", {})
    if random_runs:
        avg_random_returns = []
        for ticker in tickers_order:
            if ticker in random_runs:
                runs = random_runs[ticker]
                avg_ret = np.mean([r["cumulative_return"] for r in runs])
                avg_random_returns.append(avg_ret)
            else:
                avg_random_returns.append(0.0)

        avg_random_returns = np.array(avg_random_returns)
        if len(avg_random_returns) > 1:
            t_stat, p_value = stats.ttest_rel(ca_returns, avg_random_returns)
            test_results["ca_vs_avg_random_return"] = {
                "test": "paired_t_test",
                "metric": "cumulative_return",
                "comparison": "CA vs. mean of 100 random selections",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < config.SIGNIFICANCE_LEVEL,
                "ca_mean": float(np.mean(ca_returns)),
                "avg_random_mean": float(np.mean(avg_random_returns)),
                "difference_mean": float(np.mean(ca_returns - avg_random_returns)),
                "n_observations": len(ca_returns),
            }
            logger.info(
                f"  CA vs Avg Random (return, 100 runs): t={t_stat:.3f}, "
                f"p={p_value:.4f} "
                f"{'*** SIGNIFICANT ***' if p_value < config.SIGNIFICANCE_LEVEL else '(not significant)'}"
            )

    # ── Effect Size (Cohen's d) ──
    # I compute Cohen's d for the primary comparison because p-values alone
    # don't tell you the practical size of the effect.
    if len(ca_returns) > 1:
        diff = ca_returns - rand_returns
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0
        test_results["effect_size_ca_vs_random"] = {
            "metric": "cumulative_return",
            "cohens_d": float(cohens_d),
            "interpretation": (
                "negligible" if abs(cohens_d) < 0.2 else
                "small" if abs(cohens_d) < 0.5 else
                "medium" if abs(cohens_d) < 0.8 else
                "large"
            ),
        }
        logger.info(
            f"  Effect size (Cohen's d, CA vs Random return): {cohens_d:.3f} "
            f"({test_results['effect_size_ca_vs_random']['interpretation']})"
        )

    # I save all test results to JSON
    path = save_dir / "statistical_tests.json"
    with open(path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    logger.info(f"  Saved statistical test results to {path}")

    # I also save a concise CSV summary of the tests
    test_summary_rows = []
    for test_name, test_data in test_results.items():
        if "t_statistic" in test_data:
            test_summary_rows.append({
                "test_name": test_name,
                "metric": test_data.get("metric", ""),
                "t_statistic": test_data["t_statistic"],
                "p_value": test_data["p_value"],
                "significant_at_5pct": test_data.get("significant", False),
                "ca_mean": test_data.get("ca_mean", ""),
                "comparison_mean": test_data.get("random_mean", test_data.get("buyhold_mean", test_data.get("avg_random_mean", ""))),
                "difference": test_data.get("difference_mean", ""),
            })

    if test_summary_rows:
        test_df = pd.DataFrame(test_summary_rows)
        test_csv_path = save_dir / "statistical_tests_summary.csv"
        test_df.to_csv(test_csv_path, index=False)
        logger.info(f"  Saved statistical test summary CSV to {test_csv_path}")

    return test_results


# ───────────────────────────────────────────────────────────────────────────
# Per-Stock Results Table
# ───────────────────────────────────────────────────────────────────────────
def generate_per_stock_comparison(
    result: FullExperimentResult,
    save_dir: Path = config.DATA_DIR,
) -> Path:
    """
    I create a clean comparison CSV with one row per stock showing the
    test-period performance of all three approaches side by side. This is
    the table that goes directly into the paper.
    """
    logger.info("Generating per-stock comparison table...")
    rows = []

    for ticker, sr in result.stock_results.items():
        row = {
            "ticker": ticker,
            "company": config.TICKER_NAMES.get(ticker, ""),
            "selected_strategy": sr.selected_strategy_name,
            "selected_strategy_type": sr.strategies[sr.selected_strategy_idx].strategy_type if sr.strategies else "",
            "train_sharpe_selected": sr.selection_metric_value,
        }

        # Consequence-Aware results
        if sr.test_result_consequence_aware:
            r = sr.test_result_consequence_aware
            row.update({
                "ca_test_return": r.cumulative_return,
                "ca_test_sharpe": r.sharpe_ratio,
                "ca_test_max_dd": r.max_drawdown,
                "ca_test_trades": r.total_trades,
                "ca_test_win_rate": r.win_rate,
            })

        # Random baseline results
        if sr.test_result_random:
            r = sr.test_result_random
            row.update({
                "random_strategy": sr.random_strategy_name,
                "random_test_return": r.cumulative_return,
                "random_test_sharpe": r.sharpe_ratio,
                "random_test_max_dd": r.max_drawdown,
            })

        # Buy-and-hold results
        if sr.test_result_buy_hold:
            r = sr.test_result_buy_hold
            row.update({
                "bh_test_return": r.cumulative_return,
                "bh_test_sharpe": r.sharpe_ratio,
                "bh_test_max_dd": r.max_drawdown,
            })

        # I determine the winner for each stock
        returns = {
            "CA": row.get("ca_test_return", -999),
            "Random": row.get("random_test_return", -999),
            "B&H": row.get("bh_test_return", -999),
        }
        row["winner_by_return"] = max(returns, key=returns.get)

        rows.append(row)

    df = pd.DataFrame(rows)
    path = save_dir / "per_stock_comparison.csv"
    df.to_csv(path, index=False)
    logger.info(f"  Saved per-stock comparison to {path}")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Public API — Generate All Results
# ───────────────────────────────────────────────────────────────────────────
def generate_all_results(
    result: FullExperimentResult,
    save_dir: Path = config.DATA_DIR,
) -> Dict[str, Any]:
    """
    I generate every results artifact in one call. This is the entry point
    that main.py uses after the experiment completes.
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING RESULTS AND STATISTICAL ANALYSIS")
    logger.info("=" * 60)

    outputs = {}

    outputs["detailed_csv"] = str(generate_detailed_results_csv(result, save_dir))
    outputs["summary_stats"] = generate_summary_statistics(result, save_dir)
    outputs["statistical_tests"] = run_statistical_tests(result, save_dir)
    outputs["per_stock_comparison"] = str(generate_per_stock_comparison(result, save_dir))

    # I save a complete experiment summary in JSON Lines format
    _save_jsonl_log(result, save_dir)

    logger.info("\nAll results generated successfully")
    return outputs


def _save_jsonl_log(
    result: FullExperimentResult,
    save_dir: Path,
) -> None:
    """
    I write every significant event to a JSON Lines file. Each line is a
    self-contained JSON object with a timestamp, event type, and payload.
    This format is easy to parse with standard tools (jq, pandas.read_json).
    """
    jsonl_path = config.JSON_LOG_FILE
    logger.info(f"Saving structured log to {jsonl_path}")

    events = []

    # Experiment metadata
    events.append({
        "timestamp": datetime.now().isoformat(),
        "event": "experiment_metadata",
        "data": result.experiment_metadata,
    })

    # Per-stock results
    for ticker, sr in result.stock_results.items():
        # Training results summary
        train_summaries = [r.to_summary_dict() for r in sr.training_results]
        events.append({
            "timestamp": datetime.now().isoformat(),
            "event": "training_backtest_complete",
            "ticker": ticker,
            "data": {
                "num_strategies": len(train_summaries),
                "results": train_summaries,
                "selected_idx": sr.selected_strategy_idx,
                "selected_name": sr.selected_strategy_name,
            },
        })

        # Test results
        events.append({
            "timestamp": datetime.now().isoformat(),
            "event": "test_results",
            "ticker": ticker,
            "data": {
                "consequence_aware": sr.test_result_consequence_aware.to_summary_dict() if sr.test_result_consequence_aware else None,
                "random_baseline": sr.test_result_random.to_summary_dict() if sr.test_result_random else None,
                "buy_and_hold": sr.test_result_buy_hold.to_summary_dict() if sr.test_result_buy_hold else None,
            },
        })

    with open(jsonl_path, "w") as f:
        for event in events:
            f.write(json.dumps(event, default=str) + "\n")

    logger.info(f"  Wrote {len(events)} events to structured log")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    logger.info("Results aggregator loaded. Use generate_all_results() with experiment results.")
