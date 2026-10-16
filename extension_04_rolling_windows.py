"""
extension_04_rolling_windows.py
===================================================================
Reviewer comments addressed: R2.2 (non-stationarity / regime shifts) and
R2.4 (repeated rolling-window evaluation across multiple market regimes).

A single 64-day window cannot speak to regime robustness, and pooling only
ten paired observations gives a low-power test. Here we re-run the FULL
generate -> assess -> select -> test loop over EIGHT non-overlapping quarterly
out-of-sample windows spanning 2023Q4--2025Q3 (bull, correction, and
sideways regimes). To keep the experiment fully reproducible offline (no API
or network), strategy candidates are produced by the deterministic rule-based
generator on each window's own training data; CA2I then selects the
max-training-Sharpe candidate exactly as in the main experiment.

For every (window, ticker) pair we record the test Sharpe of:
  * CA2I (max training Sharpe),
  * Random selection (expected value = mean over the K candidates),
  * Buy & Hold.

We then (a) report per-window mean Sharpe, (b) pool all window x ticker pairs
(n = 8 x 10 = 80) and run paired t-test / Wilcoxon / bootstrap on the pooled
CA2I - Random difference, and (c) save a regime-robustness bar figure.

Outputs:
  results/data/rolling_windows.csv
  results/data/rolling_windows_pooled.json
  results/figures/rolling_window_robustness.png
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import consequence_assessment as ca
import data_pipeline as dp
import strategy_generator as sg
import extension_common as ec

warnings.filterwarnings("ignore")
OUT = Path(config.DATA_DIR)
FIG = Path(config.FIGURES_DIR)

WINDOWS = [
    {"label": "2023Q4", "train_end": "2023-09-30", "test_start": "2023-10-01", "test_end": "2023-12-31"},
    {"label": "2024Q1", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-03-31"},
    {"label": "2024Q2", "train_end": "2024-03-31", "test_start": "2024-04-01", "test_end": "2024-06-30"},
    {"label": "2024Q3", "train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2024-09-30"},
    {"label": "2024Q4", "train_end": "2024-09-30", "test_start": "2024-10-01", "test_end": "2024-12-31"},
    {"label": "2025Q1", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-03-31"},
    {"label": "2025Q2", "train_end": "2025-03-31", "test_start": "2025-04-01", "test_end": "2025-06-30"},
    {"label": "2025Q3", "train_end": "2025-06-30", "test_start": "2025-07-01", "test_end": "2025-09-30"},
]


def slice_window(full: pd.DataFrame, w):
    train = full[full.index <= pd.Timestamp(w["train_end"])]
    test = full[(full.index >= pd.Timestamp(w["test_start"])) &
                (full.index <= pd.Timestamp(w["test_end"]))]
    return train, test


def bootstrap_ci(diff, n_boot=10000, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    full = {t: ec.load_prices(t, "full") for t in config.TICKERS}
    rows = []
    pooled_ca2i, pooled_rnd = [], []
    per_window = {w["label"]: {"ca2i": [], "rnd": [], "bh": []} for w in WINDOWS}

    for w in WINDOWS:
        for t in config.TICKERS:
            train, test = slice_window(full[t], w)
            if len(train) < 220 or len(test) < 20:
                continue
            summary = dp.compute_market_summary(train)
            strategies = sg.generate_strategies_rule_based(t, summary)
            train_res = [ca.backtest_strategy(s, train) for s in strategies]
            wu = train.tail(250)
            test_res = [ca.backtest_strategy(s, test, warmup_data=wu) for s in strategies]

            idx = int(np.argmax([r.sharpe_ratio for r in train_res]))
            ca2i_sr = test_res[idx].sharpe_ratio
            rnd_sr = float(np.mean([r.sharpe_ratio for r in test_res]))
            bh_sr = ca.backtest_buy_and_hold(t, test).sharpe_ratio

            rows.append({"window": w["label"], "ticker": t,
                         "ca2i_sharpe": ca2i_sr, "random_sharpe": rnd_sr,
                         "buyhold_sharpe": bh_sr})
            pooled_ca2i.append(ca2i_sr); pooled_rnd.append(rnd_sr)
            per_window[w["label"]]["ca2i"].append(ca2i_sr)
            per_window[w["label"]]["rnd"].append(rnd_sr)
            per_window[w["label"]]["bh"].append(bh_sr)

    with open(OUT / "rolling_windows.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    ca2i = np.array(pooled_ca2i); rnd = np.array(pooled_rnd)
    diff = ca2i - rnd
    t_stat, t_p = stats.ttest_rel(ca2i, rnd)
    w_stat, w_p = stats.wilcoxon(ca2i, rnd)
    lo, hi = bootstrap_ci(diff)
    n_pos, n_neg = int(np.sum(diff > 0)), int(np.sum(diff < 0))
    sign_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue

    pooled = {
        "n_pairs": int(len(diff)),
        "mean_ca2i": float(ca2i.mean()), "mean_random": float(rnd.mean()),
        "mean_diff": float(diff.mean()),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "wilcoxon_W": float(w_stat), "wilcoxon_p": float(w_p),
        "sign_pos": n_pos, "sign_neg": n_neg, "sign_p": float(sign_p),
        "boot_ci_low": lo, "boot_ci_high": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "per_window_mean": {k: {"ca2i": float(np.mean(v["ca2i"])),
                                "random": float(np.mean(v["rnd"])),
                                "buyhold": float(np.mean(v["bh"]))}
                            for k, v in per_window.items()},
    }
    with open(OUT / "rolling_windows_pooled.json", "w") as f:
        json.dump(pooled, f, indent=2)

    # figure
    labels = [w["label"] for w in WINDOWS]
    x = np.arange(len(labels)); width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, [pooled["per_window_mean"][l]["ca2i"] for l in labels], width,
           label="CA$^2$I", color=config.COLOR_CONSEQUENCE_AWARE)
    ax.bar(x, [pooled["per_window_mean"][l]["random"] for l in labels], width,
           label="Random (expected)", color=config.COLOR_RANDOM_BASELINE)
    ax.bar(x + width, [pooled["per_window_mean"][l]["buyhold"] for l in labels], width,
           label="Buy & Hold", color=config.COLOR_BUY_HOLD)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Mean test Sharpe ratio (10 equities)")
    ax.set_xlabel("Out-of-sample quarter")
    ax.set_title("Regime robustness: CA$^2$I across eight rolling out-of-sample windows")
    ax.axhline(0, color="k", lw=0.6); ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "rolling_window_robustness.png", dpi=config.FIGURE_DPI)

    print("\n=== ROLLING-WINDOW REGIME ROBUSTNESS (R2.2, R2.4) ===")
    print(f"{'window':>8} | {'CA2I':>7} {'Random':>8} {'B&H':>7}")
    for l in labels:
        pw = pooled["per_window_mean"][l]
        print(f"{l:>8} | {pw['ca2i']:>+7.3f} {pw['random']:>+8.3f} {pw['buyhold']:>+7.3f}")
    print(f"\nPooled (n={pooled['n_pairs']} window x stock pairs):")
    print(f"  mean CA2I={pooled['mean_ca2i']:+.3f}  mean Random={pooled['mean_random']:+.3f}  "
          f"diff={pooled['mean_diff']:+.3f}")
    print(f"  paired t : t={t_stat:+.3f}, p={t_p:.4f}")
    print(f"  Wilcoxon : W={w_stat:.1f}, p={w_p:.4f}")
    print(f"  sign     : {n_pos}+/{n_neg}-, p={sign_p:.4f}")
    print(f"  bootstrap 95% CI of mean diff: [{lo:+.3f}, {hi:+.3f}]  "
          f"excludes 0? {pooled['ci_excludes_zero']}")
    print(f"\nSaved CSV, JSON, and figure (rolling_window_robustness.png).")


if __name__ == "__main__":
    main()
