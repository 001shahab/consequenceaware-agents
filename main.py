"""
Main Experiment Orchestrator — Consequence-Aware Agentic Trading Research
=======================================================================
Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia (shb@3sholding.com)

This is the entry point that runs the entire experiment end-to-end. I designed
it so that a reviewer can clone the repo, install dependencies, and run:

    python main.py

...and get the complete set of results, figures, and statistical tests without
any manual intervention. Reproducibility is not optional—it's the minimum
standard for publishable research.

Pipeline stages:
  1. Data Pipeline: Fetch, validate, and split OHLCV data for all stocks.
  2. Experiment: For each stock, generate strategies, assess consequences,
     select the best, and test out-of-sample. Run baselines.
  3. Visualization: Generate publication-ready figures.
  4. Results: Aggregate metrics, compute statistical significance, save CSVs.
  5. Summary: Print key findings to the console and log.

I time every stage and log extensively. If something fails, the logs should
tell you exactly where and why.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

import config
import data_pipeline
import experiment_runner
import visualization
import results_aggregator

# ───────────────────────────────────────────────────────────────────────────
# Logging Setup
# ───────────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    """
    I configure dual logging: human-readable to console + file, and
    structured JSON to a separate file. The console output keeps the
    researcher informed during the run; the file log is the permanent record.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # I clear any existing handlers to avoid duplicate output
    root_logger.handlers = []

    # Console handler — I want to see what's happening in real time
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    root_logger.addHandler(console_handler)

    # File handler — the permanent audit trail
    file_handler = logging.FileHandler(config.EXPERIMENT_LOG_FILE, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    root_logger.addHandler(file_handler)


logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ───────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """
    I keep command-line arguments minimal. The experiment should run with
    sensible defaults from config.py, but I allow overrides for convenience.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Consequence-Aware Agentic Trading Experiment\n"
            "Author: Prof. Shahab Anbarjafari, 3S Holding OÜ"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM API calls; use deterministic rule-based strategy generation",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Override the default ticker list (e.g., --tickers AAPL MSFT TSLA)",
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help=f"Random seed (default: {config.RANDOM_SEED})",
    )
    return parser.parse_args()


# ───────────────────────────────────────────────────────────────────────────
# Main Experiment Pipeline
# ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    I orchestrate the full experiment pipeline. Each stage is timed and
    logged independently so that failures are immediately traceable.
    """
    args = parse_args()
    setup_logging()

    # I set the global random seed at the very beginning
    np.random.seed(args.seed)
    config.RANDOM_SEED = args.seed

    total_start = time.time()

    logger.info("=" * 70)
    logger.info("CONSEQUENCE-AWARE AGENTIC TRADING — EXPERIMENT START")
    logger.info("=" * 70)
    logger.info(f"Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Use LLM: {not args.no_llm}")
    logger.info(f"Tickers: {args.tickers or config.TICKERS}")
    logger.info(f"Training period: {config.DATA_START_DATE} to {config.TRAIN_END_DATE}")
    logger.info(f"Test period: {config.TEST_START_DATE} to {config.DATA_END_DATE}")
    logger.info(f"Output directory: {config.RESULTS_DIR}")
    logger.info(f"Selection metric: {config.PRIMARY_SELECTION_METRIC}")
    logger.info("=" * 70)

    # ── Stage 1: Data Pipeline ──────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: DATA PIPELINE")
    logger.info("=" * 70)
    stage_start = time.time()

    tickers = args.tickers if args.tickers else None
    train_data, test_data, quality_reports = data_pipeline.run_pipeline(tickers)

    stage_time = time.time() - stage_start
    logger.info(f"Stage 1 completed in {stage_time:.1f}s")

    # I verify we have data for all tickers before proceeding
    assert len(train_data) > 0, "No training data available — cannot proceed"
    assert len(test_data) > 0, "No test data available — cannot proceed"
    for ticker in train_data:
        assert ticker in test_data, f"Missing test data for {ticker}"

    # ── Stage 2: Experiment ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: RUNNING EXPERIMENT")
    logger.info("=" * 70)
    stage_start = time.time()

    use_llm = not args.no_llm
    full_result = experiment_runner.run_full_experiment(
        train_data, test_data, use_llm=use_llm
    )

    stage_time = time.time() - stage_start
    logger.info(f"Stage 2 completed in {stage_time:.1f}s")

    # ── Stage 3: Visualization ──────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: GENERATING VISUALIZATIONS")
    logger.info("=" * 70)
    stage_start = time.time()

    figure_manifest = visualization.generate_all_figures(full_result)

    stage_time = time.time() - stage_start
    logger.info(f"Stage 3 completed in {stage_time:.1f}s")

    # ── Stage 4: Results Aggregation ────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 4: AGGREGATING RESULTS")
    logger.info("=" * 70)
    stage_start = time.time()

    results_output = results_aggregator.generate_all_results(full_result)

    stage_time = time.time() - stage_start
    logger.info(f"Stage 4 completed in {stage_time:.1f}s")

    # ── Stage 5: Final Summary ──────────────────────────────────────────
    total_time = time.time() - total_start
    _print_final_summary(full_result, total_time, results_output)

    logger.info(f"\nExperiment completed successfully in {total_time:.1f}s")
    logger.info(f"All outputs saved to: {config.RESULTS_DIR}")
    logger.info("=" * 70)


def _print_final_summary(
    result: experiment_runner.FullExperimentResult,
    total_time: float,
    results_output: dict,
) -> None:
    """
    I print a comprehensive final summary that captures the key findings.
    This is what I'd present at a lab meeting or include in the abstract.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE — KEY FINDINGS")
    logger.info("=" * 70)

    # I compute aggregate metrics
    ca_returns = []
    rand_returns = []
    bh_returns = []
    ca_sharpes = []
    ca_wins = 0

    for ticker, sr in result.stock_results.items():
        if sr.test_result_consequence_aware:
            ca_returns.append(sr.test_result_consequence_aware.cumulative_return)
            ca_sharpes.append(sr.test_result_consequence_aware.sharpe_ratio)
        if sr.test_result_random:
            rand_returns.append(sr.test_result_random.cumulative_return)
        if sr.test_result_buy_hold:
            bh_returns.append(sr.test_result_buy_hold.cumulative_return)

        # I count wins for the CA agent
        ca_r = sr.test_result_consequence_aware.cumulative_return if sr.test_result_consequence_aware else -999
        bh_r = sr.test_result_buy_hold.cumulative_return if sr.test_result_buy_hold else -999
        if ca_r > bh_r:
            ca_wins += 1

    n_stocks = len(result.stock_results)

    logger.info(f"\n1. PERFORMANCE SUMMARY (Test Period)")
    logger.info(f"   Consequence-Aware Agent:")
    logger.info(f"     Mean Return:  {np.mean(ca_returns):.2%} ± {np.std(ca_returns):.2%}")
    logger.info(f"     Mean Sharpe:  {np.mean(ca_sharpes):.3f}")
    logger.info(f"   Random Baseline:")
    logger.info(f"     Mean Return:  {np.mean(rand_returns):.2%} ± {np.std(rand_returns):.2%}")
    logger.info(f"   Buy-and-Hold:")
    logger.info(f"     Mean Return:  {np.mean(bh_returns):.2%} ± {np.std(bh_returns):.2%}")

    logger.info(f"\n2. WIN RATE")
    logger.info(f"   CA Agent beat Buy-and-Hold in {ca_wins}/{n_stocks} stocks")

    logger.info(f"\n3. STATISTICAL SIGNIFICANCE")
    stat_tests = results_output.get("statistical_tests", {})
    for test_name, test_data in stat_tests.items():
        if "p_value" in test_data:
            sig = "YES" if test_data.get("significant", False) else "NO"
            logger.info(
                f"   {test_name}: p={test_data['p_value']:.4f} (significant: {sig})"
            )

    logger.info(f"\n4. EXECUTION")
    logger.info(f"   Total time: {total_time:.1f}s")
    logger.info(f"   Stocks processed: {n_stocks}")
    logger.info(f"   Strategies per stock: {config.NUM_STRATEGIES_PER_STOCK}")
    logger.info(f"   Total backtests: {n_stocks * config.NUM_STRATEGIES_PER_STOCK}")

    logger.info(f"\n5. OUTPUT FILES")
    logger.info(f"   Results directory: {config.RESULTS_DIR}")
    logger.info(f"   Figures: {config.FIGURES_DIR}")
    logger.info(f"   Data/CSVs: {config.DATA_DIR}")
    logger.info(f"   Logs: {config.LOGS_DIR}")

    logger.info(f"\n6. LIMITATIONS & NEXT STEPS")
    logger.info(f"   - Test period is 3 months; longer OOS window would strengthen results")
    logger.info(f"   - Long-only strategies; short selling could improve in bear markets")
    logger.info(f"   - No transaction costs modeled; real-world friction would affect active strategies more")
    logger.info(f"   - Strategy space limited to 5 technical analysis types; richer strategy libraries may help")
    logger.info(f"   - Single random baseline draw; see 100-run random baseline for robustness")


if __name__ == "__main__":
    main()
