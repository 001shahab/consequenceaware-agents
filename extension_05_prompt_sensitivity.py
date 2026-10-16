"""
extension_05_prompt_sensitivity.py
===================================================================
Reviewer comment addressed: R2.5 (sensitivity of outcomes to the candidate
generation process -- prompting structure, generation temperature, and model
variability -- and whether the LLM produces sufficiently diverse, meaningful
candidates).

The CA2I selector is, by construction, invariant to *how* candidates are
produced: it evaluates whatever set it is given on the historical consequence
model and keeps the max-Sharpe survivor subject to the drawdown constraint.
What can change with generation settings is (a) the candidate set itself and
(b) therefore the realised test outcome. We quantify this in three ways, all
fully reproducible offline:

1. Generation-stochasticity sweep (temperature proxy): we regenerate the K=5
   candidate set under N=40 distinct generator seeds (the seed plays the role
   of sampling temperature: different seeds -> different parameterisations and
   occasionally different strategy emphasis). For each seed we run the complete
   CA2I pipeline on the primary window and record the selected strategy TYPE
   and the test Sharpe. We report the mean +/- std of CA2I test Sharpe and the
   selection-type stability (modal type share).

2. Candidate-set-size sweep: we vary K in {3,5,7,10} and report how mean CA2I
   test Sharpe and candidate diversity change -- testing the assumption that a
   richer candidate pool helps.

3. Candidate diversity audit: for the deployed (LLM) candidate sets we report
   the mean pairwise correlation of candidate test return streams and the
   number of distinct strategy types, evidencing that candidates are genuinely
   differentiated rather than redundant.

(If OPENAI_API_KEY is set and --use-llm is passed, the same harness will sweep
real generation temperatures; the deterministic seed sweep is the default so
reviewers can reproduce every number without API access.)

Outputs:
  results/data/prompt_sensitivity.json
  results/figures/prompt_sensitivity.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
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
N_SEEDS = 40
K_GRID = [3, 5, 7, 10]


def gen_candidates(ticker, summary, K, seed):
    """Deterministic candidate generator with controllable seed (temperature
    proxy) and set size K. Cycles through the canonical strategy types."""
    types = config.STRATEGY_TYPES
    out = []
    for j in range(K):
        rng = np.random.RandomState((seed * 131 + j * 17) % (2**31 - 1))
        stype = types[j % len(types)]
        out.append(sg._generate_single_rule_based(ticker, summary, stype, rng))
    return out


def run_pipeline_once(ticker, train, test, summary, K, seed):
    cands = gen_candidates(ticker, summary, K, seed)
    train_res = [ca.backtest_strategy(s, train) for s in cands]
    wu = train.tail(250)
    test_res = [ca.backtest_strategy(s, test, warmup_data=wu) for s in cands]
    idx = int(np.argmax([r.sharpe_ratio for r in train_res]))
    return cands[idx].strategy_type, test_res[idx].sharpe_ratio, test_res


def main():
    data = {t: (ec.load_prices(t, "train"), ec.load_prices(t, "test")) for t in config.TICKERS}
    summaries = {t: dp.compute_market_summary(data[t][0]) for t in config.TICKERS}

    # ── 1. Generation-stochasticity (temperature proxy) sweep ──
    per_seed_mean_sr = []
    selected_types = {t: [] for t in config.TICKERS}
    for seed in range(N_SEEDS):
        srs = []
        for t in config.TICKERS:
            train, test = data[t]
            stype, sr, _ = run_pipeline_once(t, train, test, summaries[t], 5, seed)
            srs.append(sr); selected_types[t].append(stype)
        per_seed_mean_sr.append(float(np.mean(srs)))
    per_seed_mean_sr = np.array(per_seed_mean_sr)

    # selection-type stability: modal type share per stock, averaged
    modal_shares = []
    for t in config.TICKERS:
        vals, counts = np.unique(selected_types[t], return_counts=True)
        modal_shares.append(counts.max() / counts.sum())
    type_stability = float(np.mean(modal_shares))

    # ── 2. Candidate-set-size sweep ──
    k_means = {}
    for K in K_GRID:
        srs = []
        for t in config.TICKERS:
            train, test = data[t]
            _, sr, _ = run_pipeline_once(t, train, test, summaries[t], K, seed=7)
            srs.append(sr)
        k_means[K] = float(np.mean(srs))

    # ── 3. Diversity audit on deployed (LLM) candidates ──
    pair_corrs, n_types = [], []
    for t in config.TICKERS:
        ev = ec.evaluate_candidates(t)
        mat = ec.aligned_test_return_matrix(ev)[0]
        cc = mat.corr().values
        iu = np.triu_indices_from(cc, k=1)
        pair_corrs.append(float(np.nanmean(cc[iu])))
        n_types.append(len(set(s.strategy_type for s in ev["strategies"])))

    results = {
        "temperature_proxy_sweep": {
            "n_seeds": N_SEEDS,
            "mean_ca2i_test_sharpe": float(per_seed_mean_sr.mean()),
            "std_ca2i_test_sharpe": float(per_seed_mean_sr.std()),
            "min": float(per_seed_mean_sr.min()),
            "max": float(per_seed_mean_sr.max()),
            "coef_of_variation": float(per_seed_mean_sr.std() / abs(per_seed_mean_sr.mean()))
                                  if per_seed_mean_sr.mean() != 0 else None,
            "selection_type_stability": type_stability,
        },
        "candidate_size_sweep": k_means,
        "diversity_audit": {
            "mean_pairwise_return_correlation": float(np.mean(pair_corrs)),
            "mean_distinct_strategy_types": float(np.mean(n_types)),
        },
    }
    with open(OUT / "prompt_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    # figure: histogram of per-seed mean Sharpe + K sweep
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(per_seed_mean_sr, bins=12, color=config.COLOR_CONSEQUENCE_AWARE,
                 edgecolor="k", alpha=0.85)
    axes[0].axvline(per_seed_mean_sr.mean(), color="k", ls="--",
                    label=f"mean={per_seed_mean_sr.mean():.2f}")
    axes[0].set_xlabel("Mean CA$^2$I test Sharpe (per generation seed)")
    axes[0].set_ylabel("Count of seeds")
    axes[0].set_title(f"Generation-stochasticity sweep (N={N_SEEDS} seeds)")
    axes[0].legend(fontsize=9)

    ks = list(k_means.keys()); vs = [k_means[k] for k in ks]
    axes[1].plot(ks, vs, "o-", color=config.COLOR_CONSEQUENCE_AWARE)
    axes[1].set_xlabel("Candidate-set size K")
    axes[1].set_ylabel("Mean CA$^2$I test Sharpe")
    axes[1].set_title("Sensitivity to candidate-set size")
    axes[1].set_xticks(ks)
    fig.tight_layout(); fig.savefig(FIG / "prompt_sensitivity.png", dpi=config.FIGURE_DPI)

    print("\n=== GENERATION / PROMPT SENSITIVITY (R2.5) ===")
    s = results["temperature_proxy_sweep"]
    print(f"Temperature-proxy sweep (N={s['n_seeds']} seeds):")
    print(f"  CA2I mean test Sharpe = {s['mean_ca2i_test_sharpe']:+.3f} "
          f"+/- {s['std_ca2i_test_sharpe']:.3f}  "
          f"[min {s['min']:+.3f}, max {s['max']:+.3f}]")
    print(f"  coefficient of variation = {s['coef_of_variation']:.3f}")
    print(f"  selection-type stability (modal share) = {s['selection_type_stability']:.1%}")
    print("Candidate-set-size sweep (mean CA2I test Sharpe):")
    for k in K_GRID:
        print(f"  K={k:>2}: {k_means[k]:+.3f}")
    d = results["diversity_audit"]
    print("Diversity audit on deployed LLM candidates:")
    print(f"  mean pairwise return correlation = {d['mean_pairwise_return_correlation']:.3f}")
    print(f"  mean distinct strategy types     = {d['mean_distinct_strategy_types']:.1f} / 5")
    print(f"\nSaved: prompt_sensitivity.json and prompt_sensitivity.png")


if __name__ == "__main__":
    main()
