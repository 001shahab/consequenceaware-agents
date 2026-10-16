"""
extension_01_advanced_baselines.py
===================================================================
Reviewer comments addressed: R1.1, R2.3 ("stronger / state-of-the-art
baselines": established portfolio optimisation, return-based heuristic,
diversified ensembles, and an oracle upper bound).

Every baseline operates inside the SAME per-asset candidate-selection task as
CA2I: the LLM proposes K=5 candidate strategies per asset, each baseline makes
its decision using TRAINING-period information only, and we report the realised
TEST-period (Aug--Oct 2025) outcome. This isolates the value of the *selection
rule* and keeps all methods comparable.

Baselines
---------
  CA2I (ours)        : pick candidate with max training Sharpe.
  Return-greedy      : pick candidate with max training cumulative return
                       (R2.3's explicit "returns rather than Sharpe" heuristic).
  Calmar-select      : pick candidate with max training Calmar ratio.
  MinVol-select      : pick candidate with min training volatility.
  Ensemble (1/K)     : equal-weight portfolio over ALL K candidates
                       (a diversified, no-selection benchmark).
  Mean-Variance (MV) : Markowitz max-Sharpe long-only weights over the K
                       candidates estimated on TRAINING daily returns, applied
                       out-of-sample (established portfolio optimisation).
  Risk-Parity        : inverse-volatility weights over the K candidates
                       (training vols), applied out-of-sample.
  Buy & Hold         : passive benchmark.
  Oracle             : max TEST Sharpe (clairvoyant upper bound, not attainable).

Outputs
-------
  results/data/advanced_baselines_per_stock.csv
  results/data/advanced_baselines_summary.csv
  console table for the paper.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

import config
import extension_common as ec

warnings.filterwarnings("ignore")
OUT = Path(config.DATA_DIR)


# ── portfolio weight solvers over the K candidate-strategy return series ──
def mean_variance_weights(train_mat: np.ndarray) -> np.ndarray:
    """Long-only max-Sharpe (tangency) weights with a ridge-regularised
    covariance, solved by projected grid over the simplex via the analytical
    unconstrained solution clipped to the simplex. Robust for small K."""
    mu = train_mat.mean(axis=0) * ec.ANN - ec.RF_ANNUAL
    cov = np.cov(train_mat.T) * ec.ANN
    K = train_mat.shape[1]
    cov += np.eye(K) * 1e-6  # ridge for invertibility
    try:
        w = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        w = np.ones(K)
    w = np.clip(w, 0, None)            # long-only
    if w.sum() <= 1e-12:
        w = np.ones(K)
    return w / w.sum()


def risk_parity_weights(train_mat: np.ndarray) -> np.ndarray:
    vol = train_mat.std(axis=0)
    inv = 1.0 / np.where(vol > 1e-12, vol, np.nan)
    inv = np.nan_to_num(inv, nan=0.0)
    if inv.sum() <= 1e-12:
        inv = np.ones(train_mat.shape[1])
    return inv / inv.sum()


def portfolio_metrics(weights: np.ndarray, test_mat: np.ndarray):
    """Daily portfolio returns = test_mat @ weights; return (cum_ret, sharpe)."""
    port = test_mat @ weights
    cum = float(np.prod(1.0 + port) - 1.0)
    import pandas as pd
    sr = ec.sharpe_from_returns(pd.Series(port))
    return cum, sr


def main():
    per_stock = []
    agg = {k: {"sr": [], "ret": []} for k in
           ["CA2I", "ReturnGreedy", "Calmar", "MinVol",
            "Ensemble", "MeanVariance", "RiskParity", "BuyHold", "Oracle"]}

    for t in config.TICKERS:
        ev = ec.evaluate_candidates(t)
        train, test, bh = ev["train"], ev["test"], ev["bh"]
        strat_names = [s.name for s in ev["strategies"]]

        # selection-rule baselines (decide on TRAIN, realise on TEST)
        def pick(metric_fn):
            idx = int(np.argmax([metric_fn(r) for r in train]))
            return idx, test[idx]

        idx_ca2i, r_ca2i = pick(lambda r: r.sharpe_ratio)
        idx_ret,  r_ret  = pick(lambda r: r.cumulative_return)
        idx_cal,  r_cal  = pick(lambda r: r.calmar_ratio)
        idx_mv,   r_mv   = pick(lambda r: -r.volatility)   # min vol = max(-vol)

        # portfolio baselines over candidate return series
        test_mat = ec.aligned_test_return_matrix(ev)[0].values
        train_mat = ec.aligned_train_return_matrix(ev).values
        # align K
        K = min(test_mat.shape[1], train_mat.shape[1])
        test_mat, train_mat = test_mat[:, :K], train_mat[:, :K]

        ens_ret, ens_sr = portfolio_metrics(np.ones(K) / K, test_mat)
        mv_w = mean_variance_weights(train_mat)
        mvp_ret, mvp_sr = portfolio_metrics(mv_w, test_mat)
        rp_w = risk_parity_weights(train_mat)
        rp_ret, rp_sr = portfolio_metrics(rp_w, test_mat)

        # oracle (clairvoyant)
        idx_or = int(np.argmax([r.sharpe_ratio for r in test]))
        r_or = test[idx_or]

        row = {
            "ticker": t,
            "CA2I_sel": strat_names[idx_ca2i],
            "CA2I_sr": r_ca2i.sharpe_ratio, "CA2I_ret": r_ca2i.cumulative_return,
            "ReturnGreedy_sr": r_ret.sharpe_ratio, "ReturnGreedy_ret": r_ret.cumulative_return,
            "Calmar_sr": r_cal.sharpe_ratio, "Calmar_ret": r_cal.cumulative_return,
            "MinVol_sr": r_mv.sharpe_ratio, "MinVol_ret": r_mv.cumulative_return,
            "Ensemble_sr": ens_sr, "Ensemble_ret": ens_ret,
            "MeanVariance_sr": mvp_sr, "MeanVariance_ret": mvp_ret,
            "RiskParity_sr": rp_sr, "RiskParity_ret": rp_ret,
            "BuyHold_sr": bh.sharpe_ratio, "BuyHold_ret": bh.cumulative_return,
            "Oracle_sr": r_or.sharpe_ratio, "Oracle_ret": r_or.cumulative_return,
        }
        per_stock.append(row)

        agg["CA2I"]["sr"].append(r_ca2i.sharpe_ratio); agg["CA2I"]["ret"].append(r_ca2i.cumulative_return)
        agg["ReturnGreedy"]["sr"].append(r_ret.sharpe_ratio); agg["ReturnGreedy"]["ret"].append(r_ret.cumulative_return)
        agg["Calmar"]["sr"].append(r_cal.sharpe_ratio); agg["Calmar"]["ret"].append(r_cal.cumulative_return)
        agg["MinVol"]["sr"].append(r_mv.sharpe_ratio); agg["MinVol"]["ret"].append(r_mv.cumulative_return)
        agg["Ensemble"]["sr"].append(ens_sr); agg["Ensemble"]["ret"].append(ens_ret)
        agg["MeanVariance"]["sr"].append(mvp_sr); agg["MeanVariance"]["ret"].append(mvp_ret)
        agg["RiskParity"]["sr"].append(rp_sr); agg["RiskParity"]["ret"].append(rp_ret)
        agg["BuyHold"]["sr"].append(bh.sharpe_ratio); agg["BuyHold"]["ret"].append(bh.cumulative_return)
        agg["Oracle"]["sr"].append(r_or.sharpe_ratio); agg["Oracle"]["ret"].append(r_or.cumulative_return)

    # write per-stock CSV
    with open(OUT / "advanced_baselines_per_stock.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_stock[0].keys()))
        w.writeheader(); w.writerows(per_stock)

    # summary
    print("\n=== ADVANCED BASELINES: mean TEST Sharpe / return over 10 stocks ===")
    print(f"{'Method':<14}{'mean SR':>10}{'std SR':>10}{'mean ret':>11}")
    summary_rows = []
    for k in ["CA2I", "ReturnGreedy", "Calmar", "MinVol", "Ensemble",
              "MeanVariance", "RiskParity", "BuyHold", "Oracle"]:
        msr, ssr = np.mean(agg[k]["sr"]), np.std(agg[k]["sr"])
        mret = np.mean(agg[k]["ret"])
        print(f"{k:<14}{msr:>+10.3f}{ssr:>10.3f}{mret:>+11.2%}")
        summary_rows.append({"method": k, "mean_sharpe": msr,
                             "std_sharpe": ssr, "mean_return": mret})
    with open(OUT / "advanced_baselines_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "mean_sharpe", "std_sharpe", "mean_return"])
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nSaved: {OUT/'advanced_baselines_per_stock.csv'}")
    print(f"Saved: {OUT/'advanced_baselines_summary.csv'}")


if __name__ == "__main__":
    main()
