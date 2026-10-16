"""
extension_06_figures.py
===================================================================
Regenerates two figures for the revised manuscript from the reproducible
backtests:

1. equity_curves_combined.png  -- adds explicit y-axis labels ("Portfolio
   value (USD)") to every panel (R1.11) and annotates that flat segments of
   the ACTIVE strategies correspond to periods held in cash (no open position),
   whereas buy-and-hold is fully invested throughout and is therefore never
   flat. This directly resolves the reviewer's "constant for extended periods"
   query: the constant segments belong to the active CA2I/Random strategies in
   cash, not to buy-and-hold.

2. advanced_baselines_bar.png -- mean test Sharpe of CA2I against the advanced
   baselines from extension_01.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import consequence_assessment as ca
import extension_common as ec

warnings.filterwarnings("ignore")
FIG = Path(config.FIGURES_DIR)
DATA = Path(config.DATA_DIR)


def equity_curves_combined():
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=False)
    axes = axes.flatten()
    for ax, t in zip(axes, config.TICKERS):
        ev = ec.evaluate_candidates(t)
        train, test = ev["train"], ev["test"]
        idx_ca2i = int(np.argmax([r.sharpe_ratio for r in train]))
        idx_rnd = int(np.argmax([0 for _ in train]))  # placeholder
        # random = expected: plot the mean equity across candidates
        cand_eq = [r.equity_curve / r.equity_curve.iloc[0] for r in test]
        rnd_eq = sum(cand_eq) / len(cand_eq) * config.INITIAL_CAPITAL
        ca2i_eq = test[idx_ca2i].equity_curve
        bh_eq = ev["bh"].equity_curve

        ax.plot(ca2i_eq.index, ca2i_eq.values, color=config.COLOR_CONSEQUENCE_AWARE,
                lw=1.8, label="CA$^2$I")
        ax.plot(rnd_eq.index, rnd_eq.values, color=config.COLOR_RANDOM_BASELINE,
                lw=1.2, ls="--", label="Random (mean)")
        ax.plot(bh_eq.index, bh_eq.values, color=config.COLOR_BUY_HOLD,
                lw=1.2, ls=":", label="Buy & Hold")
        ax.set_title(t, fontsize=11)
        ax.set_ylabel("Portfolio value (USD)", fontsize=8)
        ax.set_xlabel("Date (Aug--Oct 2025)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Out-of-sample equity curves ($V_0=\\$100{,}000$). Flat segments "
                 "of the active strategies denote periods held in cash; "
                 "buy-and-hold is fully invested throughout.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "equity_curves_combined.png", dpi=config.FIGURE_DPI)
    plt.close(fig)
    print("Saved equity_curves_combined.png (with y-axis labels)")


def advanced_baselines_bar():
    rows = list(csv.DictReader(open(DATA / "advanced_baselines_summary.csv")))
    order = ["BuyHold", "Oracle", "CA2I", "ReturnGreedy", "Ensemble",
             "RiskParity", "Calmar", "MeanVariance", "MinVol"]
    label_map = {"CA2I": "CA$^2$I", "ReturnGreedy": "Return-greedy",
                 "Calmar": "Calmar", "MinVol": "Min-Vol", "Ensemble": "Ensemble (1/K)",
                 "MeanVariance": "Mean-Variance", "RiskParity": "Risk-Parity",
                 "BuyHold": "Buy & Hold", "Oracle": "Oracle (UB)"}
    d = {r["method"]: float(r["mean_sharpe"]) for r in rows}
    methods = [m for m in order if m in d]
    vals = [d[m] for m in methods]
    colors = []
    for m in methods:
        if m == "CA2I": colors.append(config.COLOR_CONSEQUENCE_AWARE)
        elif m in ("BuyHold", "Oracle"): colors.append("#9E9E9E")
        else: colors.append(config.COLOR_RANDOM_BASELINE)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar([label_map[m] for m in methods], vals, color=colors, edgecolor="k")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Mean test Sharpe ratio (10 equities)")
    ax.set_title("CA$^2$I vs.\\ advanced baselines (single out-of-sample window)")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.08),
                f"{v:+.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "advanced_baselines_bar.png", dpi=config.FIGURE_DPI)
    plt.close(fig)
    print("Saved advanced_baselines_bar.png")


if __name__ == "__main__":
    equity_curves_combined()
    advanced_baselines_bar()
