"""
Visualization Module for Consequence-Aware Agentic Trading Research
=======================================================================
Author: Prof. Shahab Anbarjafari, 3S Holding OÜ, Tartu, Estonia (shb@3sholding.com)

I generate all the figures for the paper and supplementary materials. My
design principle: every figure must be self-contained and interpretable
by a reader who hasn't seen the code. That means clear labels, legends,
and titles on every plot.

I use matplotlib for static publication figures (journal-ready at 300 DPI)
and ensure consistent styling across all plots. The color scheme is
designed to be colorblind-friendly and to print well in grayscale.

Figures generated:
  1. Per-stock equity curves (consequence-aware vs. baselines)
  2. Sharpe ratio distribution comparison
  3. Drawdown comparison charts
  4. Cumulative return comparison bar chart
  5. Summary dashboard with aggregate metrics
  6. Training-period strategy comparison heatmap
  7. Sensitivity analysis visualization
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # I use non-interactive backend for server/CI compatibility
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

import config
from experiment_runner import FullExperimentResult, StockExperimentResult
from consequence_assessment import BacktestResult

logger = logging.getLogger(__name__)

# I configure matplotlib defaults for publication quality
plt.rcParams.update({
    "figure.dpi": config.FIGURE_DPI,
    "savefig.dpi": config.FIGURE_DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ───────────────────────────────────────────────────────────────────────────
# Per-Stock Equity Curves
# ───────────────────────────────────────────────────────────────────────────
def plot_equity_curves_per_stock(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> List[Path]:
    """
    I plot the test-period equity curves for each stock, comparing the
    consequence-aware agent against both baselines. Each stock gets its
    own figure for clarity—cramming all ten into one figure would be
    unreadable.

    Returns a list of saved figure paths.
    """
    logger.info("Generating per-stock equity curve plots...")
    saved_paths = []

    for ticker, sr in result.stock_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        # I plot each approach with distinct colors and line styles
        if sr.test_result_consequence_aware and sr.test_result_consequence_aware.equity_curve is not None:
            ec = sr.test_result_consequence_aware.equity_curve
            ax.plot(
                ec.index, ec.values,
                color=config.COLOR_CONSEQUENCE_AWARE,
                linewidth=2.0, label="Consequence-Aware Agent",
                zorder=3,
            )

        if sr.test_result_random and sr.test_result_random.equity_curve is not None:
            ec = sr.test_result_random.equity_curve
            ax.plot(
                ec.index, ec.values,
                color=config.COLOR_RANDOM_BASELINE,
                linewidth=1.5, linestyle="--", label="Random Selection",
                zorder=2,
            )

        if sr.test_result_buy_hold and sr.test_result_buy_hold.equity_curve is not None:
            ec = sr.test_result_buy_hold.equity_curve
            ax.plot(
                ec.index, ec.values,
                color=config.COLOR_BUY_HOLD,
                linewidth=1.5, linestyle=":", label="Buy & Hold",
                zorder=2,
            )

        # I add the initial capital reference line
        ax.axhline(
            y=config.INITIAL_CAPITAL, color="gray", linestyle="-",
            alpha=0.4, linewidth=0.8, label="Initial Capital",
        )

        ax.set_title(
            f"{ticker} ({config.TICKER_NAMES.get(ticker, '')}) — "
            f"Test Period Equity Curves",
            fontweight="bold",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="best", framealpha=0.9)
        ax.tick_params(axis="x", rotation=30)

        # I add a text box with key metrics for quick reference
        ca_ret = sr.test_result_consequence_aware.cumulative_return if sr.test_result_consequence_aware else 0
        textstr = (
            f"CA Return: {ca_ret:.2%}\n"
            f"Strategy: {sr.selected_strategy_name}"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7)
        ax.text(
            0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=props,
        )

        fig.tight_layout()
        path = save_dir / f"equity_curve_{ticker}.{config.FIGURE_FORMAT}"
        fig.savefig(path)
        plt.close(fig)
        saved_paths.append(path)
        logger.info(f"  Saved equity curve for {ticker}")

    return saved_paths


# ───────────────────────────────────────────────────────────────────────────
# Combined Equity Curves (All Stocks in One Figure)
# ───────────────────────────────────────────────────────────────────────────
def plot_equity_curves_combined(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    I create a grid of subplots with all ten stocks' equity curves in a
    single figure. This is useful for the paper's main body where space
    is limited. The individual per-stock plots go in the appendix.
    """
    logger.info("Generating combined equity curve plot...")
    n_stocks = len(result.stock_results)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for idx, (ticker, sr) in enumerate(result.stock_results.items()):
        ax = axes[idx]

        if sr.test_result_consequence_aware and sr.test_result_consequence_aware.equity_curve is not None:
            ec = sr.test_result_consequence_aware.equity_curve
            ax.plot(ec.index, ec.values, color=config.COLOR_CONSEQUENCE_AWARE,
                    linewidth=1.5, label="CA Agent")

        if sr.test_result_random and sr.test_result_random.equity_curve is not None:
            ec = sr.test_result_random.equity_curve
            ax.plot(ec.index, ec.values, color=config.COLOR_RANDOM_BASELINE,
                    linewidth=1.0, linestyle="--", label="Random")

        if sr.test_result_buy_hold and sr.test_result_buy_hold.equity_curve is not None:
            ec = sr.test_result_buy_hold.equity_curve
            ax.plot(ec.index, ec.values, color=config.COLOR_BUY_HOLD,
                    linewidth=1.0, linestyle=":", label="B&H")

        ax.set_title(ticker, fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)

        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    # I hide unused subplots
    for idx in range(n_stocks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Test Period Equity Curves: Consequence-Aware Agent vs. Baselines",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    path = save_dir / f"equity_curves_combined.{config.FIGURE_FORMAT}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved combined equity curves")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Sharpe Ratio Distribution
# ───────────────────────────────────────────────────────────────────────────
def plot_sharpe_distribution(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    I compare the distribution of test-period Sharpe ratios across the three
    approaches. This is one of the most important figures in the paper—it
    shows whether consequence assessment systematically improves risk-adjusted
    returns, or if the mean improvement is driven by one or two outlier stocks.
    """
    logger.info("Generating Sharpe ratio distribution plot...")

    ca_sharpes = []
    rand_sharpes = []
    bh_sharpes = []
    tickers = []

    for ticker, sr in result.stock_results.items():
        tickers.append(ticker)
        ca_sharpes.append(
            sr.test_result_consequence_aware.sharpe_ratio if sr.test_result_consequence_aware else 0
        )
        rand_sharpes.append(
            sr.test_result_random.sharpe_ratio if sr.test_result_random else 0
        )
        bh_sharpes.append(
            sr.test_result_buy_hold.sharpe_ratio if sr.test_result_buy_hold else 0
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Grouped bar chart of Sharpe ratios per stock
    x = np.arange(len(tickers))
    width = 0.25
    axes[0].bar(x - width, ca_sharpes, width, color=config.COLOR_CONSEQUENCE_AWARE,
                label="CA Agent", alpha=0.85, edgecolor="white")
    axes[0].bar(x, rand_sharpes, width, color=config.COLOR_RANDOM_BASELINE,
                label="Random", alpha=0.85, edgecolor="white")
    axes[0].bar(x + width, bh_sharpes, width, color=config.COLOR_BUY_HOLD,
                label="Buy & Hold", alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tickers, rotation=45, ha="right")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].set_title("Test Period Sharpe Ratios by Stock", fontweight="bold")
    axes[0].legend()
    axes[0].axhline(y=0, color="black", linewidth=0.5)

    # Right: Box plot / violin plot showing distribution
    data_for_box = pd.DataFrame({
        "Consequence-Aware": ca_sharpes,
        "Random Selection": rand_sharpes,
        "Buy & Hold": bh_sharpes,
    })
    bp = axes[1].boxplot(
        [ca_sharpes, rand_sharpes, bh_sharpes],
        labels=["CA Agent", "Random", "Buy & Hold"],
        patch_artist=True,
        widths=0.5,
    )
    colors = [config.COLOR_CONSEQUENCE_AWARE, config.COLOR_RANDOM_BASELINE, config.COLOR_BUY_HOLD]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # I overlay individual data points for transparency (n=10 is small enough)
    for i, (sharpes, color) in enumerate(zip(
        [ca_sharpes, rand_sharpes, bh_sharpes], colors
    )):
        jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(sharpes))
        axes[1].scatter(
            np.full(len(sharpes), i + 1) + jitter, sharpes,
            color=color, edgecolors="black", linewidths=0.5,
            zorder=3, s=40, alpha=0.8,
        )

    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_title("Sharpe Ratio Distributions", fontweight="bold")
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle(
        "Test Period Sharpe Ratio Comparison",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    path = save_dir / f"sharpe_distribution.{config.FIGURE_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved Sharpe distribution plot")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Drawdown Comparison
# ───────────────────────────────────────────────────────────────────────────
def plot_drawdown_comparison(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    I compare maximum drawdowns across approaches and stocks. Drawdown is
    arguably more important than return for real-world trading—a strategy
    that returns 20% but experiences a 40% drawdown is psychologically and
    financially devastating.
    """
    logger.info("Generating drawdown comparison plot...")

    tickers = list(result.stock_results.keys())
    ca_dd = []
    rand_dd = []
    bh_dd = []

    for ticker in tickers:
        sr = result.stock_results[ticker]
        ca_dd.append(
            abs(sr.test_result_consequence_aware.max_drawdown) if sr.test_result_consequence_aware else 0
        )
        rand_dd.append(
            abs(sr.test_result_random.max_drawdown) if sr.test_result_random else 0
        )
        bh_dd.append(
            abs(sr.test_result_buy_hold.max_drawdown) if sr.test_result_buy_hold else 0
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(tickers))
    width = 0.25

    bars1 = ax.bar(x - width, ca_dd, width, color=config.COLOR_CONSEQUENCE_AWARE,
                   label="CA Agent", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x, rand_dd, width, color=config.COLOR_RANDOM_BASELINE,
                   label="Random", alpha=0.85, edgecolor="white")
    bars3 = ax.bar(x + width, bh_dd, width, color=config.COLOR_BUY_HOLD,
                   label="Buy & Hold", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Maximum Drawdown (absolute)")
    ax.set_title(
        "Test Period Maximum Drawdowns — Lower is Better",
        fontweight="bold",
    )
    ax.legend()

    # I add value labels on bars for readability
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2., height + 0.002,
                    f"{height:.1%}", ha="center", va="bottom", fontsize=7,
                )

    fig.tight_layout()
    path = save_dir / f"drawdown_comparison.{config.FIGURE_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved drawdown comparison plot")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Cumulative Return Comparison
# ───────────────────────────────────────────────────────────────────────────
def plot_cumulative_return_comparison(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    I create a bar chart comparing cumulative returns across all stocks
    for each approach. This is the "headline" figure that immediately
    answers: does consequence assessment improve returns?
    """
    logger.info("Generating cumulative return comparison plot...")

    tickers = list(result.stock_results.keys())
    ca_ret = []
    rand_ret = []
    bh_ret = []

    for ticker in tickers:
        sr = result.stock_results[ticker]
        ca_ret.append(
            sr.test_result_consequence_aware.cumulative_return if sr.test_result_consequence_aware else 0
        )
        rand_ret.append(
            sr.test_result_random.cumulative_return if sr.test_result_random else 0
        )
        bh_ret.append(
            sr.test_result_buy_hold.cumulative_return if sr.test_result_buy_hold else 0
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Per-stock comparison
    x = np.arange(len(tickers))
    width = 0.25
    axes[0].bar(x - width, [r * 100 for r in ca_ret], width,
                color=config.COLOR_CONSEQUENCE_AWARE, label="CA Agent", alpha=0.85)
    axes[0].bar(x, [r * 100 for r in rand_ret], width,
                color=config.COLOR_RANDOM_BASELINE, label="Random", alpha=0.85)
    axes[0].bar(x + width, [r * 100 for r in bh_ret], width,
                color=config.COLOR_BUY_HOLD, label="Buy & Hold", alpha=0.85)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tickers, rotation=45, ha="right")
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].set_title("Test Period Returns by Stock", fontweight="bold")
    axes[0].legend()
    axes[0].axhline(y=0, color="black", linewidth=0.5)

    # Right: Aggregate summary with mean ± std
    means = [np.mean(ca_ret) * 100, np.mean(rand_ret) * 100, np.mean(bh_ret) * 100]
    stds = [np.std(ca_ret) * 100, np.std(rand_ret) * 100, np.std(bh_ret) * 100]
    labels = ["CA Agent", "Random", "Buy & Hold"]
    colors = [config.COLOR_CONSEQUENCE_AWARE, config.COLOR_RANDOM_BASELINE, config.COLOR_BUY_HOLD]

    bars = axes[1].bar(labels, means, yerr=stds, capsize=5,
                       color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("Mean Cumulative Return (%)")
    axes[1].set_title("Aggregate Mean Returns (±1 std)", fontweight="bold")
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    for bar, mean in zip(bars, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
            f"{mean:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    fig.suptitle(
        "Cumulative Return Comparison: Consequence-Aware vs. Baselines",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    path = save_dir / f"cumulative_return_comparison.{config.FIGURE_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved cumulative return comparison plot")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Summary Dashboard
# ───────────────────────────────────────────────────────────────────────────
def plot_summary_dashboard(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    I create a comprehensive dashboard figure that summarizes the entire
    experiment in one view. This combines multiple metrics: returns, Sharpe
    ratios, drawdowns, and win rates.

    This figure is designed to go on the first page of the paper or in
    the executive summary.
    """
    logger.info("Generating summary dashboard...")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    tickers = list(result.stock_results.keys())

    # I collect all metrics
    metrics = {"ca": {}, "random": {}, "bh": {}}
    for metric_name in ["cumulative_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
        for approach, key in [("ca", "test_result_consequence_aware"),
                              ("random", "test_result_random"),
                              ("bh", "test_result_buy_hold")]:
            values = []
            for ticker in tickers:
                sr = result.stock_results[ticker]
                res = getattr(sr, key, None)
                values.append(getattr(res, metric_name, 0) if res else 0)
            metrics[approach][metric_name] = values

    # Panel 1: Returns bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(tickers))
    w = 0.25
    ax1.bar(x - w, [r * 100 for r in metrics["ca"]["cumulative_return"]], w,
            color=config.COLOR_CONSEQUENCE_AWARE, label="CA Agent")
    ax1.bar(x, [r * 100 for r in metrics["random"]["cumulative_return"]], w,
            color=config.COLOR_RANDOM_BASELINE, label="Random")
    ax1.bar(x + w, [r * 100 for r in metrics["bh"]["cumulative_return"]], w,
            color=config.COLOR_BUY_HOLD, label="B&H")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, rotation=45, fontsize=8)
    ax1.set_ylabel("Return (%)")
    ax1.set_title("Cumulative Returns", fontweight="bold")
    ax1.legend(fontsize=8)

    # Panel 2: Sharpe ratios
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - w, metrics["ca"]["sharpe_ratio"], w,
            color=config.COLOR_CONSEQUENCE_AWARE, label="CA Agent")
    ax2.bar(x, metrics["random"]["sharpe_ratio"], w,
            color=config.COLOR_RANDOM_BASELINE, label="Random")
    ax2.bar(x + w, metrics["bh"]["sharpe_ratio"], w,
            color=config.COLOR_BUY_HOLD, label="B&H")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, rotation=45, fontsize=8)
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Risk-Adjusted Returns", fontweight="bold")
    ax2.legend(fontsize=8)

    # Panel 3: Max Drawdowns
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x - w, [abs(d) * 100 for d in metrics["ca"]["max_drawdown"]], w,
            color=config.COLOR_CONSEQUENCE_AWARE, label="CA Agent")
    ax3.bar(x, [abs(d) * 100 for d in metrics["random"]["max_drawdown"]], w,
            color=config.COLOR_RANDOM_BASELINE, label="Random")
    ax3.bar(x + w, [abs(d) * 100 for d in metrics["bh"]["max_drawdown"]], w,
            color=config.COLOR_BUY_HOLD, label="B&H")
    ax3.set_xticks(x)
    ax3.set_xticklabels(tickers, rotation=45, fontsize=8)
    ax3.set_ylabel("Max Drawdown (%)")
    ax3.set_title("Maximum Drawdowns (Lower is Better)", fontweight="bold")
    ax3.legend(fontsize=8)

    # Panel 4: Aggregate comparison radar-style summary
    ax4 = fig.add_subplot(gs[1, 1])
    agg_data = {
        "Mean Return": [
            np.mean(metrics["ca"]["cumulative_return"]) * 100,
            np.mean(metrics["random"]["cumulative_return"]) * 100,
            np.mean(metrics["bh"]["cumulative_return"]) * 100,
        ],
        "Mean Sharpe": [
            np.mean(metrics["ca"]["sharpe_ratio"]),
            np.mean(metrics["random"]["sharpe_ratio"]),
            np.mean(metrics["bh"]["sharpe_ratio"]),
        ],
        "Mean |DD|": [
            np.mean([abs(d) for d in metrics["ca"]["max_drawdown"]]) * 100,
            np.mean([abs(d) for d in metrics["random"]["max_drawdown"]]) * 100,
            np.mean([abs(d) for d in metrics["bh"]["max_drawdown"]]) * 100,
        ],
    }

    table_data = []
    for metric_name, values in agg_data.items():
        table_data.append([metric_name] + [f"{v:.2f}" for v in values])

    table = ax4.table(
        cellText=table_data,
        colLabels=["Metric", "CA Agent", "Random", "Buy & Hold"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.axis("off")
    ax4.set_title("Aggregate Summary", fontweight="bold", pad=20)

    # Panel 5: Strategy selection summary
    ax5 = fig.add_subplot(gs[2, :])
    selection_data = []
    for ticker in tickers:
        sr = result.stock_results[ticker]
        selection_data.append({
            "Ticker": ticker,
            "Selected": sr.selected_strategy_name,
            "Train Sharpe": f"{sr.selection_metric_value:.2f}",
            "Test Return": f"{sr.test_result_consequence_aware.cumulative_return:.2%}" if sr.test_result_consequence_aware else "N/A",
            "B&H Return": f"{sr.test_result_buy_hold.cumulative_return:.2%}" if sr.test_result_buy_hold else "N/A",
        })

    cell_text = [[d["Ticker"], d["Selected"], d["Train Sharpe"],
                  d["Test Return"], d["B&H Return"]] for d in selection_data]
    table2 = ax5.table(
        cellText=cell_text,
        colLabels=["Ticker", "Selected Strategy", "Train Sharpe", "Test Return", "B&H Return"],
        cellLoc="center",
        loc="center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.8)
    ax5.axis("off")
    ax5.set_title("Strategy Selection Details", fontweight="bold", pad=20)

    fig.suptitle(
        "Consequence-Aware Agentic Trading — Experiment Summary Dashboard",
        fontsize=15, fontweight="bold", y=0.98,
    )

    path = save_dir / f"summary_dashboard.{config.FIGURE_FORMAT}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved summary dashboard")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Training Period Strategy Heatmap
# ───────────────────────────────────────────────────────────────────────────
def plot_training_heatmap(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    I create a heatmap showing the training-period Sharpe ratio for every
    strategy across every stock. This helps the reader understand the
    consequence assessment landscape—how much variation is there between
    strategies for each stock?
    """
    logger.info("Generating training strategy heatmap...")

    tickers = list(result.stock_results.keys())
    strategy_names = config.STRATEGY_TYPES

    # I build the Sharpe ratio matrix
    sharpe_matrix = np.zeros((len(tickers), len(strategy_names)))
    for i, ticker in enumerate(tickers):
        sr = result.stock_results[ticker]
        for j, tr in enumerate(sr.training_results):
            if j < len(strategy_names):
                sharpe_matrix[i, j] = tr.sharpe_ratio

    fig, ax = plt.subplots(figsize=(10, 8))
    im = sns.heatmap(
        sharpe_matrix,
        annot=True, fmt=".2f",
        xticklabels=[s.replace("_", " ").title() for s in strategy_names],
        yticklabels=tickers,
        cmap="RdYlGn",
        center=0,
        ax=ax,
        cbar_kws={"label": "Sharpe Ratio"},
    )

    # I mark the selected strategy for each stock
    for i, ticker in enumerate(tickers):
        sr = result.stock_results[ticker]
        j = sr.selected_strategy_idx
        if 0 <= j < len(strategy_names):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor="blue", linewidth=3))

    ax.set_title(
        "Training Period Sharpe Ratios\n(Blue border = selected strategy)",
        fontweight="bold",
    )
    ax.set_xlabel("Strategy Type")
    ax.set_ylabel("Stock")

    fig.tight_layout()
    path = save_dir / f"training_heatmap.{config.FIGURE_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved training heatmap")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Sensitivity Analysis Visualization
# ───────────────────────────────────────────────────────────────────────────
def plot_sensitivity_analysis(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> Path:
    """
    I visualize the sensitivity analysis results: how often does the
    selection change when using different metrics? This is important for
    establishing that our results aren't fragile.
    """
    logger.info("Generating sensitivity analysis plot...")

    if not result.sensitivity_results:
        logger.warning("No sensitivity results to plot")
        return save_dir / "sensitivity_analysis.png"

    tickers = list(result.stock_results.keys())
    metrics = list(result.sensitivity_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    # I create a matrix: for each alt metric × ticker, did selection change?
    change_matrix = np.zeros((len(metrics), len(tickers)))
    for i, metric in enumerate(metrics):
        for j, ticker in enumerate(tickers):
            if ticker in result.sensitivity_results[metric]:
                changed = result.sensitivity_results[metric][ticker].get("selection_changed", False)
                change_matrix[i, j] = 1.0 if changed else 0.0

    sns.heatmap(
        change_matrix,
        annot=True, fmt=".0f",
        xticklabels=tickers,
        yticklabels=[m.replace("_", " ").title() for m in metrics],
        cmap="YlOrRd",
        vmin=0, vmax=1,
        ax=ax,
        cbar_kws={"label": "Selection Changed (1=Yes)"},
    )

    total_changes = int(change_matrix.sum())
    total_cells = change_matrix.size
    ax.set_title(
        f"Sensitivity Analysis: Selection Stability\n"
        f"({total_changes}/{total_cells} selections changed with alternative metrics)",
        fontweight="bold",
    )

    fig.tight_layout()
    path = save_dir / f"sensitivity_analysis.{config.FIGURE_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved sensitivity analysis plot")
    return path


# ───────────────────────────────────────────────────────────────────────────
# Public API — Generate All Figures
# ───────────────────────────────────────────────────────────────────────────
def generate_all_figures(
    result: FullExperimentResult,
    save_dir: Path = config.FIGURES_DIR,
) -> Dict[str, Any]:
    """
    I generate every figure in one call. This is the entry point that
    main.py uses. I return a manifest of all generated figure paths.
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PUBLICATION FIGURES")
    logger.info("=" * 60)

    manifest = {}

    manifest["equity_curves_per_stock"] = plot_equity_curves_per_stock(result, save_dir)
    manifest["equity_curves_combined"] = str(plot_equity_curves_combined(result, save_dir))
    manifest["sharpe_distribution"] = str(plot_sharpe_distribution(result, save_dir))
    manifest["drawdown_comparison"] = str(plot_drawdown_comparison(result, save_dir))
    manifest["cumulative_returns"] = str(plot_cumulative_return_comparison(result, save_dir))
    manifest["summary_dashboard"] = str(plot_summary_dashboard(result, save_dir))
    manifest["training_heatmap"] = str(plot_training_heatmap(result, save_dir))
    manifest["sensitivity_analysis"] = str(plot_sensitivity_analysis(result, save_dir))

    n_figures = sum(
        len(v) if isinstance(v, list) else 1
        for v in manifest.values()
    )
    logger.info(f"\nGenerated {n_figures} figures total")

    # I save the manifest for reference
    import json
    manifest_serializable = {
        k: [str(p) for p in v] if isinstance(v, list) else str(v)
        for k, v in manifest.items()
    }
    manifest_path = save_dir / "figure_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_serializable, f, indent=2)

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    logger.info("Visualization module loaded. Use generate_all_figures() with experiment results.")
