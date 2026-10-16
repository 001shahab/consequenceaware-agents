"""
extension_11_summary_dashboard.py
===================================================================
Second-round reviewer comment R1.2: "an explicit mapping should be provided
showing which strategy family each of the named strategies in the Figure 1
table belongs to."

Rebuilds the deployment-window summary dashboard (Fig. 1) so that
  * the strategy-selection table carries an explicit family column,
  * the random reference is the *expected* value of uniform selection
    (mean over the K = 5 candidates), matching Table II, and
  * the full name -> family mapping of all 50 GPT-5.4 candidates is exported.

Outputs:
  results/figures/summary_dashboard.png
  results/data/strategy_family_mapping.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import extension_cache as xc
from extension_08_ensemble_comparison import panel

OUT = Path(config.DATA_DIR)
FIG = Path(config.FIGURES_DIR)

FAMILY = {
    "ma_crossover": "Moving-average crossover (MA)",
    "rsi_mean_reversion": "RSI mean reversion (RSI)",
    "bollinger_breakout": "Bollinger-band breakout (BB)",
    "macd_signal": "MACD signal (MACD)",
    "momentum": "Price momentum (MOM)",
}
SHORT = {"ma_crossover": "MA", "rsi_mean_reversion": "RSI", "bollinger_breakout": "BB",
         "macd_signal": "MACD", "momentum": "MOM"}
COLORS = {"CA2I": "#2196F3", "Random": "#FF9800", "B&H": "#4CAF50"}


def main():
    dep = xc.load()["deployment"]
    per = pd.read_csv(OUT / "per_stock_comparison.csv").set_index("ticker")
    tickers = list(per.index)

    rows, mapping = [], []
    for t in tickers:
        cands = dep[t]["cands"]
        sel = int(np.argmax([c["insample_sr"] for c in cands]))
        assert cands[sel]["name"] == per.loc[t, "selected_strategy"], t
        panels = [panel(c["test_ret"].values) for c in cands]
        for i, c in enumerate(cands):
            mapping.append({"ticker": t, "strategy": c["name"], "family": FAMILY[c["type"]],
                            "train_sharpe": c["insample_sr"], "selected_by_ca2i": i == sel})
        rows.append({
            "ticker": t, "strategy": cands[sel]["name"], "family": SHORT[cands[sel]["type"]],
            "train_sr": cands[sel]["insample_sr"],
            "ca_ret": per.loc[t, "ca_test_return"], "ca_sr": per.loc[t, "ca_test_sharpe"],
            "ca_dd": abs(per.loc[t, "ca_test_max_dd"]),
            "rnd_ret": np.mean([p["cum_return"] for p in panels]),
            "rnd_sr": np.mean([p["sharpe"] for p in panels]),
            "rnd_dd": np.mean([abs(p["max_dd"]) for p in panels]),
            "bh_ret": per.loc[t, "bh_test_return"], "bh_sr": per.loc[t, "bh_test_sharpe"],
            "bh_dd": abs(per.loc[t, "bh_test_max_dd"]),
        })
    df = pd.DataFrame(rows)
    pd.DataFrame(mapping).to_csv(OUT / "strategy_family_mapping.csv", index=False)
    print(df[["ticker", "strategy", "family", "rnd_ret", "rnd_sr", "rnd_dd"]].round(4).to_string(index=False))
    print("means:", df[["ca_ret", "ca_sr", "ca_dd", "rnd_ret", "rnd_sr", "rnd_dd", "bh_ret", "bh_sr", "bh_dd"]]
          .mean().round(4).to_dict())

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.42, 1.2], hspace=0.45, wspace=0.25)
    x, w = np.arange(len(tickers)), 0.26
    specs = [("ret", "(a) Test cumulative return (%)", 100), ("sr", "(b) Test Sharpe ratio", 1),
             ("dd", "(c) Test maximum drawdown (%; lower is better)", 100)]
    for j, (key, title, scale) in enumerate(specs):
        ax = fig.add_subplot(gs[0, j])
        for o, (lab, pre) in enumerate((("CA2I", "ca"), ("Random", "rnd"), ("B&H", "bh"))):
            name = {"CA2I": r"CA$^2$I", "Random": "Random (expected)", "B&H": "Buy-and-hold"}[lab]
            ax.bar(x + (o - 1) * w, df[f"{pre}_{key}"] * scale, w, label=name, color=COLORS[lab])
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, fontsize=8)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    agg = [["Mean cumulative return (%)", f"{100*df.ca_ret.mean():+.2f}", f"{100*df.rnd_ret.mean():+.2f}",
            f"{100*df.bh_ret.mean():+.2f}"],
           ["Mean Sharpe ratio", f"{df.ca_sr.mean():+.2f}", f"{df.rnd_sr.mean():+.2f}", f"{df.bh_sr.mean():+.2f}"],
           ["Mean maximum drawdown (%)", f"{100*df.ca_dd.mean():.2f}", f"{100*df.rnd_dd.mean():.2f}",
            f"{100*df.bh_dd.mean():.2f}"]]
    tb = ax.table(cellText=agg, colLabels=["Metric", "CA$^2$I", "Random (expected)", "Buy-and-hold"],
                  cellLoc="center", colWidths=[0.38, 0.18, 0.24, 0.2], bbox=[0.12, 0.0, 0.76, 0.86])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    ax.set_title("(d) Aggregate summary (ten equities)", fontsize=11, fontweight="bold", pad=2)

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    cell = [[r.ticker, r.strategy, r.family, f"{r.train_sr:.2f}", f"{100*r.ca_ret:+.2f}%",
             f"{100*r.bh_ret:+.2f}%"] for r in df.itertuples()]
    tb = ax.table(cellText=cell, colLabels=["Ticker", "Selected strategy (GPT-5.4 name)", "Family",
                                            "Train Sharpe", "Test return", "B&H return"],
                  cellLoc="center", colWidths=[0.09, 0.34, 0.1, 0.13, 0.13, 0.13], bbox=[0.04, 0.0, 0.92, 0.93])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    ax.set_title("(e) Strategy selected by the Consequence Assessment Module, with its strategy family",
                 fontsize=11, fontweight="bold", pad=2)
    fig.text(0.5, 0.075, "Families: MA = moving-average crossover; RSI = relative strength index mean "
             "reversion; BB = Bollinger-band breakout; MACD = moving-average convergence/divergence signal; "
             "MOM = price momentum.", ha="center", fontsize=8.5)
    fig.savefig(FIG / "summary_dashboard.png", dpi=200, bbox_inches="tight")
    print("Saved summary_dashboard.png and strategy_family_mapping.csv")


if __name__ == "__main__":
    main()
