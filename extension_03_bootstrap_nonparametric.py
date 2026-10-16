"""
extension_03_bootstrap_nonparametric.py
===================================================================
Reviewer comments addressed: R2.4 (bootstrap confidence intervals and
non-parametric tests, in addition to the paired t-test).

All inputs are computed from the reproducible backtests (no hard-coded numbers).
For each stock:
  * CA2I test Sharpe        = test Sharpe of the max-training-Sharpe candidate.
  * Random (expected) Sharpe = mean test Sharpe over the K candidates, which is
    the exact expectation of uniform random selection. We also draw the 100-run
    empirical random distribution (seed-controlled) to bootstrap over draws.

We then report, for the CA2I - Random Sharpe difference across the 10 stocks:
  * paired t-test (parametric, two-tailed),
  * Wilcoxon signed-rank test (non-parametric paired),
  * sign test,
  * 10,000-resample bootstrap 95% CI of the mean difference.

We additionally test CA2I against the two strongest classical baselines
(Mean-Variance and Risk-Parity) for completeness.

Outputs: results/data/statistical_robustness.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

import config
import extension_common as ec
from extension_01_advanced_baselines import (
    mean_variance_weights, risk_parity_weights, portfolio_metrics)

warnings.filterwarnings("ignore")
OUT = Path(config.DATA_DIR)
RNG = np.random.default_rng(config.RANDOM_SEED)


def bootstrap_ci(diff, n_boot=10000, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_block(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    n_pos = int(np.sum(diff > 0)); n_neg = int(np.sum(diff < 0))
    sign_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) else float("nan")
    m, lo, hi = bootstrap_ci(diff)
    block = {
        "comparison": name,
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "wilcoxon_W": float(w_stat), "wilcoxon_p": float(w_p),
        "sign_pos": n_pos, "sign_neg": n_neg, "sign_p": float(sign_p),
        "boot_mean": m, "boot_ci_low": lo, "boot_ci_high": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }
    return block


def main():
    ca2i, rnd_exp, mv, rp = [], [], [], []
    rnd_draw_means = []  # per-stock mean over 100 empirical draws

    for t in config.TICKERS:
        ev = ec.evaluate_candidates(t)
        train, test = ev["train"], ev["test"]
        idx = int(np.argmax([r.sharpe_ratio for r in train]))
        ca2i.append(test[idx].sharpe_ratio)

        cand_sr = np.array([r.sharpe_ratio for r in test])
        rnd_exp.append(float(cand_sr.mean()))

        # empirical 100-draw random selection (uniform among candidates)
        draws = RNG.integers(0, len(cand_sr), size=config.NUM_RANDOM_BASELINE_RUNS)
        rnd_draw_means.append(float(cand_sr[draws].mean()))

        # classical portfolio baselines
        test_mat = ec.aligned_test_return_matrix(ev)[0].values
        train_mat = ec.aligned_train_return_matrix(ev).values
        K = min(test_mat.shape[1], train_mat.shape[1])
        test_mat, train_mat = test_mat[:, :K], train_mat[:, :K]
        mv.append(portfolio_metrics(mean_variance_weights(train_mat), test_mat)[1])
        rp.append(portfolio_metrics(risk_parity_weights(train_mat), test_mat)[1])

    results = {
        "n_stocks": len(config.TICKERS),
        "per_stock": {
            "ca2i_sharpe": ca2i, "random_expected_sharpe": rnd_exp,
            "random_100draw_mean_sharpe": rnd_draw_means,
            "mean_variance_sharpe": mv, "risk_parity_sharpe": rp,
        },
        "tests": [
            paired_block("CA2I vs Random (expected)", ca2i, rnd_exp),
            paired_block("CA2I vs Random (100-draw mean)", ca2i, rnd_draw_means),
            paired_block("CA2I vs Mean-Variance", ca2i, mv),
            paired_block("CA2I vs Risk-Parity", ca2i, rp),
        ],
    }

    with open(OUT / "statistical_robustness.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== STATISTICAL ROBUSTNESS (R2.4) — all values from live backtests ===")
    for b in results["tests"]:
        print(f"\n{b['comparison']}: mean {b['mean_a']:+.3f} vs {b['mean_b']:+.3f} "
              f"(diff {b['mean_diff']:+.3f})")
        print(f"  paired t : t={b['t_stat']:+.3f}, p={b['t_p']:.4f}")
        print(f"  Wilcoxon : W={b['wilcoxon_W']:.1f}, p={b['wilcoxon_p']:.4f}")
        print(f"  sign     : {b['sign_pos']}+/{b['sign_neg']}-, p={b['sign_p']:.4f}")
        print(f"  bootstrap 95% CI of mean diff: [{b['boot_ci_low']:+.3f}, "
              f"{b['boot_ci_high']:+.3f}]  excludes 0? {b['ci_excludes_zero']}")
    print(f"\nSaved: {OUT/'statistical_robustness.json'}")


if __name__ == "__main__":
    main()
