"""
extension_07_improved_selector.py
===================================================================
Methodological upgrade of the CA2I consequence model (reviewer-response /
"make it actually work" extension).

Motivation
----------
The original CA2I selects the single candidate with the highest *single-window*
in-sample Sharpe ratio.  Section VI shows this is statistically indistinguishable
from random selection: a single contiguous in-sample window is a noisy, biased
estimator of out-of-sample skill (winner's curse / backtest overfitting), so the
candidate that looks best in-sample is often the one that overfit noise.

Yet the *oracle* (best candidate ex-post) reaches a mean test Sharpe of ~2.15
versus ~0.37 for random selection -- i.e. the candidate pool already contains
excellent strategies; the original selector recovers only ~8% of the available
skill.  This module closes that gap with a more principled consequence model:

    CA2I+  =  walk-forward cross-validated, robustness-penalised selection
              of a top-m ensemble.

Three principled, PRE-COMMITTED design choices (NOT tuned on test data):
  1. Walk-forward cross-validation.  Each candidate is scored on M=5 contiguous
     out-of-sample blocks *inside* the training window (expanding warm-up), not
     on one in-sample fit.  This directly attacks overfitting.
  2. Robustness penalty (lower-confidence-bound score).
        score = mean_fold_Sharpe - LAMBDA * std_fold_Sharpe
     Candidates whose edge is unstable across sub-periods are penalised.
  3. Top-m ensemble.  Instead of betting on a single pick, we equally weight the
     top m=3 feasible candidates, diversifying away idiosyncratic selection error.
A hard drawdown feasibility filter (median fold MaxDD >= -RHO_MAX) is retained,
consistent with the paper's hard-constraint design.

Everything is deterministic and offline (rule-based generator + the same
backtesting engine), so reviewers can reproduce every number.

Outputs:
  results/data/improved_selector_primary.csv     (per-ticker, primary split)
  results/data/improved_selector_ksweep.csv      (K-sweep means)
  results/data/improved_selector_stats.json       (statistical tests)
  results/data/improved_selector_rolling.json     (pooled rolling evaluation)
  results/figures/improved_selector.png
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List

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

# ── Pre-committed hyper-parameters (fixed a priori, never tuned on test) ──────
M_FOLDS = 5            # walk-forward CV blocks inside the training window
LAMBDA = 0.5           # robustness penalty weight on fold-Sharpe std
TOP_M = 3              # ensemble size
RHO_MAX = 0.30         # hard drawdown feasibility cap (median fold MaxDD)
WARMUP_BARS = 250      # initial warm-up reserved before the first CV fold
ANN = config.TRADING_DAYS_PER_YEAR
RF = config.RISK_FREE_RATE
SEED = config.RANDOM_SEED


# ───────────────────────────────────────────────────────────────────────────
# Candidate pool (deterministic, arbitrary size K)
# ───────────────────────────────────────────────────────────────────────────
def _stable_hash(s: str) -> int:
    """Process-independent hash (Python's built-in hash() is randomised per run
    via PYTHONHASHSEED, which would make candidate pools non-reproducible)."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100000


def make_pool(ticker: str, summary: Dict[str, float], k: int,
              seed_offset: int = 0) -> List:
    """Deterministically generate a diverse pool of k candidate strategies by
    drawing parameter variants across all five strategy families. Fully
    reproducible across processes (uses a stable hash, not Python's hash())."""
    pool = []
    v = 0
    while len(pool) < k:
        for stype in config.STRATEGY_TYPES:
            if len(pool) >= k:
                break
            seed = (SEED + 100000 * seed_offset + 1000 * v
                    + _stable_hash(ticker + stype)) % (2 ** 31 - 1)
            rng = np.random.RandomState(seed)
            pool.append(sg._generate_single_rule_based(ticker, summary, stype, rng))
        v += 1
    return pool[:k]


# ───────────────────────────────────────────────────────────────────────────
# Walk-forward CV scoring on the training window
# ───────────────────────────────────────────────────────────────────────────
def cv_scores(strategy, train: pd.DataFrame) -> Dict[str, float]:
    """Score one candidate by walk-forward CV inside the training window.
    Returns mean/std fold Sharpe, the LCB score, and median fold MaxDD."""
    n = len(train)
    if n < WARMUP_BARS + 2 * M_FOLDS:
        # too short: fall back to a single in-sample fit
        r = ca.backtest_strategy(strategy, train)
        return {"mean": r.sharpe_ratio, "std": 0.0,
                "lcb": r.sharpe_ratio, "dd": r.max_drawdown}

    body = train.iloc[WARMUP_BARS:]
    bounds = np.linspace(0, len(body), M_FOLDS + 1).astype(int)
    fold_sr, fold_dd = [], []
    for i in range(M_FOLDS):
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo < 15:
            continue
        fold = body.iloc[lo:hi]
        # expanding warm-up = everything in train before this fold
        warm = train.loc[: fold.index[0]].iloc[:-1]
        res = ca.backtest_strategy(strategy, fold, warmup_data=warm.tail(WARMUP_BARS))
        fold_sr.append(res.sharpe_ratio)
        fold_dd.append(res.max_drawdown)

    if not fold_sr:
        r = ca.backtest_strategy(strategy, train)
        return {"mean": r.sharpe_ratio, "std": 0.0,
                "lcb": r.sharpe_ratio, "dd": r.max_drawdown}

    mean = float(np.mean(fold_sr))
    std = float(np.std(fold_sr))
    return {"mean": mean, "std": std,
            "lcb": mean - LAMBDA * std,
            "dd": float(np.median(fold_dd))}


def ensemble_returns(selected_results: List) -> pd.Series:
    """Equal-weight, daily-rebalanced average of the selected strategies'
    test daily-return series."""
    cols = {}
    for j, r in enumerate(selected_results):
        cols[j] = ec.daily_returns(r)
    mat = pd.DataFrame(cols).dropna(how="all").fillna(0.0)
    return mat.mean(axis=1)


# ───────────────────────────────────────────────────────────────────────────
# Backtest a candidate pool once (size k), return arrays we can subset by K
# ───────────────────────────────────────────────────────────────────────────
def build_arrays(ticker: str, train: pd.DataFrame, test: pd.DataFrame,
                 k: int, seed: int) -> Dict:
    summary = dp.compute_market_summary(train)
    pool = make_pool(ticker, summary, k, seed_offset=seed)
    wu = train.tail(WARMUP_BARS)
    insample, test_res, lcb, dd = [], [], [], []
    for s in pool:
        insample.append(ca.backtest_strategy(s, train).sharpe_ratio)
        test_res.append(ca.backtest_strategy(s, test, warmup_data=wu))
        c = cv_scores(s, train)
        lcb.append(c["lcb"]); dd.append(c["dd"])
    return {
        "insample": np.array(insample),
        "test_sr": np.array([r.sharpe_ratio for r in test_res]),
        "test_dd": np.array([r.max_drawdown for r in test_res]),
        "test_res": test_res,
        "lcb": np.array(lcb),
        "cvdd": np.array(dd),
    }


def metrics_from_arrays(a: Dict, k: int) -> Dict[str, float]:
    """Derive every selector's out-of-sample metrics from the first k candidates."""
    insample = a["insample"][:k]
    test_sr = a["test_sr"][:k]
    test_dd = a["test_dd"][:k]
    test_res = a["test_res"][:k]
    lcb = a["lcb"][:k]
    cvdd = a["cvdd"][:k]

    random_exp = float(test_sr.mean())
    random_dd = float(test_dd.mean())
    oracle = float(test_sr.max())
    insample_idx = int(np.argmax(insample))
    ca2i_orig = float(test_sr[insample_idx])
    ca2i_orig_dd = float(test_dd[insample_idx])

    feasible = [i for i in range(k) if cvdd[i] >= -RHO_MAX] or list(range(k))
    order = sorted(feasible, key=lambda i: -lcb[i])
    cvlcb = float(test_sr[order[0]])

    top = order[:TOP_M]
    ens_ret = ensemble_returns([test_res[i] for i in top])
    ca2i_plus = ec.sharpe_from_returns(ens_ret)
    ca2i_plus_dd = ec.max_drawdown_from_equity((1.0 + ens_ret).cumprod())

    return {
        "random": random_exp, "random_dd": random_dd,
        "ca2i_orig": ca2i_orig, "ca2i_orig_dd": ca2i_orig_dd,
        "cvlcb": cvlcb,
        "ca2i_plus": ca2i_plus, "ca2i_plus_dd": ca2i_plus_dd,
        "oracle": oracle,
    }


# ───────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ───────────────────────────────────────────────────────────────────────────
def bootstrap_ci(diff, n_boot=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_tests(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), 1.0
    n_pos, n_neg = int(np.sum(diff > 0)), int(np.sum(diff < 0))
    sign_p = stats.binomtest(n_pos, max(n_pos + n_neg, 1), 0.5).pvalue
    lo, hi = bootstrap_ci(diff)
    return {
        "label": label, "n": int(len(diff)),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "sign_pos": n_pos, "sign_neg": n_neg, "sign_p": float(sign_p),
        "boot_ci": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


# ───────────────────────────────────────────────────────────────────────────
# Rolling evaluation (pooled, higher power)
# ───────────────────────────────────────────────────────────────────────────
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
ROLLING_K = 10
N_SEEDS = 10            # candidate-generation seeds we average over


def run_rolling(seeds):
    """Pooled rolling evaluation, averaged over candidate-generation seeds.
    Returns seed-averaged per-pair values, per-seed pooled diffs (robustness),
    and per-window means."""
    full = {t: ec.load_prices(t, "full") for t in config.TICKERS}
    # collect per (window,ticker) across seeds
    keys = ("random", "ca2i_orig", "ca2i_plus", "ca2i_plus_dd", "random_dd", "ca2i_orig_dd")
    pair_acc = {}          # (window,ticker) -> {metric: [over seeds]}
    per_seed_diff = {"plus_vs_rand": [], "orig_vs_rand": []}
    for seed in seeds:
        seed_rand, seed_plus, seed_orig = [], [], []
        for w in WINDOWS:
            for t in config.TICKERS:
                f = full[t]
                train = f[f.index <= pd.Timestamp(w["train_end"])]
                test = f[(f.index >= pd.Timestamp(w["test_start"])) &
                         (f.index <= pd.Timestamp(w["test_end"]))]
                if len(train) < WARMUP_BARS + 40 or len(test) < 20:
                    continue
                m = metrics_from_arrays(build_arrays(t, train, test, ROLLING_K, seed), ROLLING_K)
                key = (w["label"], t)
                pair_acc.setdefault(key, {kk: [] for kk in keys})
                for kk in keys:
                    pair_acc[key][kk].append(m[kk])
                seed_rand.append(m["random"]); seed_plus.append(m["ca2i_plus"]); seed_orig.append(m["ca2i_orig"])
        per_seed_diff["plus_vs_rand"].append(float(np.mean(np.array(seed_plus) - np.array(seed_rand))))
        per_seed_diff["orig_vs_rand"].append(float(np.mean(np.array(seed_orig) - np.array(seed_rand))))

    # seed-averaged per pair
    avg = {kk: [] for kk in keys}
    labels_order = [w["label"] for w in WINDOWS]
    per_window = {l: {"random": [], "ca2i_orig": [], "ca2i_plus": []} for l in labels_order}
    for (label, t), d in pair_acc.items():
        for kk in keys:
            avg[kk].append(float(np.mean(d[kk])))
        per_window[label]["random"].append(float(np.mean(d["random"])))
        per_window[label]["ca2i_orig"].append(float(np.mean(d["ca2i_orig"])))
        per_window[label]["ca2i_plus"].append(float(np.mean(d["ca2i_plus"])))

    pd_arr = np.array(per_seed_diff["plus_vs_rand"])
    res = {
        "k": ROLLING_K, "n_seeds": len(seeds), "n_pairs": len(avg["random"]),
        "seed_avg__ca2i_plus_vs_random": paired_tests(avg["ca2i_plus"], avg["random"], "CA2I+ vs Random (seed-avg)"),
        "seed_avg__ca2i_plus_vs_orig": paired_tests(avg["ca2i_plus"], avg["ca2i_orig"], "CA2I+ vs CA2I(orig) (seed-avg)"),
        "seed_avg__orig_vs_random": paired_tests(avg["ca2i_orig"], avg["random"], "CA2I(orig) vs Random (seed-avg)"),
        "per_seed_plus_vs_rand_diff": per_seed_diff["plus_vs_rand"],
        "per_seed_orig_vs_rand_diff": per_seed_diff["orig_vs_rand"],
        "per_seed_plus_positive": int((pd_arr > 0).sum()),
        "per_seed_plus_mean": float(pd_arr.mean()),
        "per_seed_plus_min": float(pd_arr.min()),
        "per_seed_plus_max": float(pd_arr.max()),
        "mean_dd": {"random": float(np.mean(avg["random_dd"])),
                    "ca2i_orig": float(np.mean(avg["ca2i_orig_dd"])),
                    "ca2i_plus": float(np.mean(avg["ca2i_plus_dd"]))},
        "per_window_mean": {l: {m: float(np.mean(per_window[l][m])) if per_window[l][m] else 0.0
                                for m in ("random", "ca2i_orig", "ca2i_plus")}
                            for l in labels_order},
    }
    return res


# ───────────────────────────────────────────────────────────────────────────
def main():
    seeds = list(range(N_SEEDS))
    print(f"=== CA2I+ : cross-validated robust ensemble selector "
          f"(seed-averaged over {N_SEEDS} generation seeds) ===\n")

    # ---- Primary split, K-sweep (seed-averaged) -------------------------
    KS = (5, 10, 15, 20)
    # per (ticker) cache arrays at max K for each seed, subset for each K
    maxK = max(KS)
    # accumulate per-ticker metric across seeds, for each K
    ksweep = {k: {m: [] for m in ("random", "ca2i_orig", "cvlcb", "ca2i_plus", "oracle")} for k in KS}
    primary_pairs = {m: {} for m in ("random", "ca2i_orig", "ca2i_plus", "ca2i_plus_dd", "random_dd", "ca2i_orig_dd")}
    for seed in seeds:
        for t in config.TICKERS:
            train = ec.load_prices(t, "train")
            test = ec.load_prices(t, "test")
            arr = build_arrays(t, train, test, maxK, seed)
            for k in KS:
                mk = metrics_from_arrays(arr, k)
                for m in ksweep[k]:
                    ksweep[k][m].append(mk[m])
                if k == 15:
                    for m in primary_pairs:
                        primary_pairs[m].setdefault(t, []).append(mk[m])
    ksweep_mean = {k: {m: float(np.mean(v)) for m, v in ksweep[k].items()} for k in KS}

    # seed-averaged primary per-ticker vectors (for stats at K=15)
    tickers = config.TICKERS
    pavg = {m: np.array([np.mean(primary_pairs[m][t]) for t in tickers]) for m in primary_pairs}
    stats_out = {"primary_k15": {
        "ca2i_plus_vs_random": paired_tests(pavg["ca2i_plus"], pavg["random"], "CA2I+ vs Random"),
        "ca2i_plus_vs_orig": paired_tests(pavg["ca2i_plus"], pavg["ca2i_orig"], "CA2I+ vs CA2I(orig)"),
        "orig_vs_random": paired_tests(pavg["ca2i_orig"], pavg["random"], "CA2I(orig) vs Random"),
        "mean_dd": {"random": float(pavg["random_dd"].mean()),
                    "ca2i_orig": float(pavg["ca2i_orig_dd"].mean()),
                    "ca2i_plus": float(pavg["ca2i_plus_dd"].mean())},
    }}

    for k in KS:
        r = ksweep_mean[k]
        print(f"K={k:>2}: random={r['random']:+.3f}  CA2I_orig={r['ca2i_orig']:+.3f}  "
              f"CV-LCB={r['cvlcb']:+.3f}  CA2I+={r['ca2i_plus']:+.3f}  oracle={r['oracle']:+.3f}")

    pd.DataFrame({m: pavg[m] for m in pavg}, index=tickers).to_csv(OUT / "improved_selector_primary.csv")
    with open(OUT / "improved_selector_ksweep.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["K", "random", "ca2i_orig", "cvlcb", "ca2i_plus", "oracle"])
        for k in KS:
            r = ksweep_mean[k]
            wtr.writerow([k, r["random"], r["ca2i_orig"], r["cvlcb"], r["ca2i_plus"], r["oracle"]])

    print("\n--- Statistical tests (primary split, K=15, n=10, seed-averaged) ---")
    for key, blk in stats_out["primary_k15"].items():
        if key == "mean_dd":
            continue
        print(f"  {blk['label']}: diff={blk['mean_diff']:+.3f}  t_p={blk['t_p']:.3f}  "
              f"Wilcoxon_p={blk['wilcoxon_p']:.3f}  sign {blk['sign_pos']}+/{blk['sign_neg']}-  "
              f"bootCI=[{blk['boot_ci'][0]:+.3f},{blk['boot_ci'][1]:+.3f}]  sig={blk['ci_excludes_zero']}")

    # ---- Rolling pooled evaluation (seed-averaged) ----------------------
    print(f"\n--- Rolling pooled evaluation (K={ROLLING_K}, seed-averaged) ---")
    rolling = run_rolling(seeds)
    for key in ("seed_avg__ca2i_plus_vs_random", "seed_avg__ca2i_plus_vs_orig", "seed_avg__orig_vs_random"):
        b = rolling[key]
        print(f"  {b['label']}: n={b['n']}  mean {b['mean_a']:+.3f} vs {b['mean_b']:+.3f}  "
              f"diff={b['mean_diff']:+.3f}  t_p={b['t_p']:.4f}  Wilcoxon_p={b['wilcoxon_p']:.4f}  "
              f"sign {b['sign_pos']}+/{b['sign_neg']}-  "
              f"bootCI=[{b['boot_ci'][0]:+.3f},{b['boot_ci'][1]:+.3f}]  sig={b['ci_excludes_zero']}")
    print(f"  per-seed robustness: CA2I+ beats random in "
          f"{rolling['per_seed_plus_positive']}/{rolling['n_seeds']} seeds "
          f"(diff mean={rolling['per_seed_plus_mean']:+.3f}, "
          f"range [{rolling['per_seed_plus_min']:+.3f},{rolling['per_seed_plus_max']:+.3f}])")
    print(f"  mean MaxDD: random={rolling['mean_dd']['random']:.3f}  "
          f"orig={rolling['mean_dd']['ca2i_orig']:.3f}  CA2I+={rolling['mean_dd']['ca2i_plus']:.3f}")

    stats_out["rolling"] = rolling
    with open(OUT / "improved_selector_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    with open(OUT / "improved_selector_rolling.json", "w") as f:
        json.dump(rolling, f, indent=2)

    # ---- Figure ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ks = list(KS)
    ax1.plot(ks, [ksweep_mean[k]["random"] for k in ks], "o--", color=config.COLOR_RANDOM_BASELINE, label="Random (expected)")
    ax1.plot(ks, [ksweep_mean[k]["ca2i_orig"] for k in ks], "s--", color="#9C27B0", label="CA$^2$I (original)")
    ax1.plot(ks, [ksweep_mean[k]["ca2i_plus"] for k in ks], "D-", color=config.COLOR_CONSEQUENCE_AWARE, lw=2.2, label="CA$^2$I+ (CV ensemble)")
    ax1.plot(ks, [ksweep_mean[k]["oracle"] for k in ks], "^:", color="#4CAF50", label="Oracle (upper bound)")
    ax1.set_xlabel("Candidate-set size $K$"); ax1.set_ylabel("Mean test Sharpe (10 equities)")
    ax1.set_title("(a) CA$^2$I+ vs. baselines across $K$"); ax1.set_xticks(ks)
    ax1.axhline(0, color="k", lw=0.5); ax1.legend(fontsize=8, loc="best")

    labels = [w["label"] for w in WINDOWS]
    x = np.arange(len(labels)); width = 0.27
    pw = rolling["per_window_mean"]
    ax2.bar(x - width, [pw[l]["random"] for l in labels], width, label="Random", color=config.COLOR_RANDOM_BASELINE)
    ax2.bar(x, [pw[l]["ca2i_orig"] for l in labels], width, label="CA$^2$I (orig)", color="#9C27B0")
    ax2.bar(x + width, [pw[l]["ca2i_plus"] for l in labels], width, label="CA$^2$I+", color=config.COLOR_CONSEQUENCE_AWARE)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, fontsize=8)
    ax2.set_ylabel("Mean test Sharpe (10 equities)")
    ax2.set_title("(b) Regime robustness (rolling windows)")
    ax2.axhline(0, color="k", lw=0.5); ax2.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(FIG / "improved_selector.png", dpi=config.FIGURE_DPI)
    print("\nSaved improved_selector.png and CSV/JSON outputs.")


if __name__ == "__main__":
    main()
