"""
extension_10_candidate_diversity.py
===================================================================
Second-round reviewer comment R2.5: "How to ensure sufficient diversity among
candidates, given that parameter variations within a fixed family may yield
highly correlated strategies?"

Diversity is measured on *training-period* daily returns (the information
available before execution), using:
  * mean pairwise Pearson correlation of candidate returns;
  * effective number of independent candidates (participation ratio of the
    correlation-matrix eigenvalues, ENB = (sum l)^2 / sum l^2; 1 = all
    candidates identical, K = fully uncorrelated);
  * market-exposure overlap (Jaccard index of invested days).
For the K = 10 rolling pools (two parameterisations per family) we contrast
within-family and across-family pairs, which directly tests the reviewer's
concern. We also report whether the CA2I+ ensemble draws its members from
distinct families, and an ablation that enforces family diversity
(at most one member per family) in the top-3 ensemble.

Outputs:
  results/data/candidate_diversity.json
  results/figures/candidate_diversity.png
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import extension_cache as xc
import extension_07_improved_selector as e7
from extension_08_ensemble_comparison import panel, test_matrix, order_by_score, paired

OUT = Path(config.DATA_DIR)
FIG = Path(config.FIGURES_DIR)


def train_matrix(cands: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame({i: c["train_ret"] for i, c in enumerate(cands)}).dropna(how="all").fillna(0.0)


def enb(corr: np.ndarray) -> float:
    lam = np.clip(np.linalg.eigvalsh(np.nan_to_num(corr)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else 1.0


def diversity(cands: List[Dict]) -> Dict:
    R = train_matrix(cands)
    C = R.corr().to_numpy(copy=True)
    np.fill_diagonal(C, 1.0)
    C = np.nan_to_num(C)
    k = len(cands)
    invested = (R.values != 0)
    within, across, jacc = [], [], []
    for i, j in itertools.combinations(range(k), 2):
        (within if cands[i]["type"] == cands[j]["type"] else across).append(C[i, j])
        union = np.logical_or(invested[:, i], invested[:, j]).sum()
        jacc.append(np.logical_and(invested[:, i], invested[:, j]).sum() / union if union else 0.0)
    order = order_by_score(cands)
    top = order[:e7.TOP_M]
    return {
        "mean_corr": float(np.mean(within + across)),
        "within_family_corr": float(np.mean(within)) if within else float("nan"),
        "across_family_corr": float(np.mean(across)) if across else float("nan"),
        "enb": enb(C), "k": k, "enb_ratio": enb(C) / k,
        "overlap_jaccard": float(np.mean(jacc)),
        "n_families": len({c["type"] for c in cands}),
        "top3_distinct_families": len({cands[i]["type"] for i in top}),
        "top3_mean_corr": float(np.mean([C[i, j] for i, j in itertools.combinations(top, 2)])) if len(top) > 1 else 1.0,
    }


def family_diverse_top(cands: List[Dict], m: int) -> List[int]:
    chosen, seen = [], set()
    for i in order_by_score(cands):
        if cands[i]["type"] not in seen:
            chosen.append(i); seen.add(cands[i]["type"])
        if len(chosen) == m:
            break
    return chosen


def main():
    data = xc.load()
    dep = {t: diversity(c["cands"]) for t, c in data["deployment"].items()}
    roll_raw = {k: diversity(c["cands"]) for k, c in data["rolling"].items()}

    def avg(rows, key):
        return float(np.nanmean([r[key] for r in rows.values()]))

    keys = ["mean_corr", "within_family_corr", "across_family_corr", "enb", "enb_ratio",
            "overlap_jaccard", "n_families", "top3_distinct_families", "top3_mean_corr"]
    out = {"deployment": {k: avg(dep, k) for k in keys if k not in ("within_family_corr",)},
           "deployment_per_ticker": dep,
           "rolling": {k: avg(roll_raw, k) for k in keys}}

    w = np.array([r["within_family_corr"] for r in roll_raw.values()])
    a = np.array([r["across_family_corr"] for r in roll_raw.values()])
    out["rolling_within_vs_across"] = paired(w, a, "within-family vs across-family corr (rolling cells)")

    # does diversity predict the benefit of ensembling? (cell level, rolling)
    gains, enbs = [], []
    fam_plus, plus = [], []
    for key, cell in data["rolling"].items():
        cands = cell["cands"]
        X = test_matrix(cands)
        single = [panel(X[:, i])["sharpe"] for i in range(len(cands))]
        o = order_by_score(cands)
        plus_sr = panel(X[:, o[:e7.TOP_M]].mean(axis=1))["sharpe"]
        gains.append(plus_sr - single[o[0]])
        enbs.append(roll_raw[key]["enb"])
        plus.append(plus_sr)
        fam_plus.append(panel(X[:, family_diverse_top(cands, e7.TOP_M)].mean(axis=1))["sharpe"])
    rho = stats.spearmanr(enbs, gains)
    out["enb_vs_ensemble_gain"] = {"spearman": float(rho.statistic), "p": float(rho.pvalue), "n": len(gains)}
    out["family_diverse_ablation"] = paired(fam_plus, plus, "family-diverse top-3 vs CA2I+ top-3 [sharpe, all seeds]")

    d, r = out["deployment"], out["rolling"]
    print("=== Candidate diversity (training-period returns) ===")
    print(f"Deployment (GPT-5.4, K=5): mean corr={d['mean_corr']:+.3f}  ENB={d['enb']:.2f}/5  "
          f"overlap={d['overlap_jaccard']:.3f}  families={d['n_families']:.1f}  "
          f"CA2I+ members from {d['top3_distinct_families']:.2f} distinct families")
    for t, v in dep.items():
        print(f"   {t:6s} corr={v['mean_corr']:+.3f} ENB={v['enb']:.2f}")
    print(f"Rolling (rule-based, K=10): mean corr={r['mean_corr']:+.3f}  within-family={r['within_family_corr']:+.3f}  "
          f"across-family={r['across_family_corr']:+.3f}  ENB={r['enb']:.2f}/10  overlap={r['overlap_jaccard']:.3f}  "
          f"CA2I+ members from {r['top3_distinct_families']:.2f} families (top-3 corr {r['top3_mean_corr']:+.3f})")
    b = out["rolling_within_vs_across"]
    print(f"  within - across = {b['mean_diff']:+.3f}  W_p={b['wilcoxon_p']:.3g}  CI=[{b['boot_ci'][0]:+.3f},{b['boot_ci'][1]:+.3f}]")
    e = out["enb_vs_ensemble_gain"]
    print(f"  ENB vs (CA2I+ - top-1) Sharpe gain: Spearman={e['spearman']:+.3f} p={e['p']:.3g} n={e['n']}")
    f = out["family_diverse_ablation"]
    print(f"  family-diverse top-3 vs CA2I+: diff={f['mean_diff']:+.3f} W_p={f['wilcoxon_p']:.3g} "
          f"CI=[{f['boot_ci'][0]:+.3f},{f['boot_ci'][1]:+.3f}] (means {f['mean_a']:+.3f} vs {f['mean_b']:+.3f})")

    with open(OUT / "candidate_diversity.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    # figure: deployment correlation heat map (mean over tickers, by family) + within/across histogram
    fams = config.STRATEGY_TYPES
    short = {"ma_crossover": "MA", "rsi_mean_reversion": "RSI", "bollinger_breakout": "BB",
             "macd_signal": "MACD", "momentum": "MOM"}
    M = np.zeros((5, 5)); cnt = np.zeros((5, 5))
    for t, cell in data["deployment"].items():
        cands = cell["cands"]
        C = train_matrix(cands).corr().values
        for i, j in itertools.product(range(len(cands)), repeat=2):
            a_, b_ = fams.index(cands[i]["type"]), fams.index(cands[j]["type"])
            M[a_, b_] += np.nan_to_num(C[i, j]); cnt[a_, b_] += 1
    M = M / np.maximum(cnt, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    im = ax1.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(5)); ax1.set_yticks(range(5))
    ax1.set_xticklabels([short[f] for f in fams]); ax1.set_yticklabels([short[f] for f in fams])
    for i in range(5):
        for j in range(5):
            ax1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax1.set_title(f"(a) GPT-5.4 candidates: mean return correlation\n(ENB = {d['enb']:.2f} of 5)")
    fig.colorbar(im, ax=ax1, fraction=0.046)
    ax2.hist(w[np.isfinite(w)], bins=25, alpha=0.7, color="#E91E63", label=f"within family (mean {np.nanmean(w):.2f})")
    ax2.hist(a[np.isfinite(a)], bins=25, alpha=0.7, color=config.COLOR_CONSEQUENCE_AWARE,
             label=f"across families (mean {np.nanmean(a):.2f})")
    ax2.set_xlabel("Mean pairwise return correlation per evaluation cell")
    ax2.set_ylabel("Number of cells")
    ax2.set_title("(b) $K$=10 pools: within- vs. across-family correlation")
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "candidate_diversity.png", dpi=config.FIGURE_DPI)
    print("Saved candidate_diversity.json and candidate_diversity.png")


if __name__ == "__main__":
    main()
