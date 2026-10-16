"""
extension_08_ensemble_comparison.py
===================================================================
Second-round reviewer comment R1.1: "the equal-weight ensemble appears to be
a more practical choice ... demonstrate whether the proposed method possesses
advantages in other aspects, such as maximum drawdown or average annualised
cumulative return."

The equal-weight (1/K) ensemble executes *every* generated candidate with no
consequence screening: it is the m = K, score-free special case of the CA2I+
selector. This script answers the comment with four analyses, all on the
cached evaluations of extension_cache.py and with every CA2I+ hyper-parameter
fixed in advance:

  1. Full return/risk panel (Sharpe, annualised and cumulative return,
     maximum drawdown, Calmar, Sortino, CVaR-5%) for random, base CA2I,
     CA2I+ (top-3) and the equal-weight ensemble, on the deployment window
     (GPT-5.4 candidates, n = 10) and the multi-regime rolling evaluation
     (n = 80, averaged over 10 generation seeds).
  2. Equal-size control: CA2I+ top-m versus the expected value over *all*
     random m-subsets, for m in {1, 2, 3, 5}. This separates the value of the
     consequence ranking from the value of diversification alone.
  3. Ex-ante risk-forecast skill: within-cell Spearman correlation between
     CAM's cross-validated drawdown / score and the realised test drawdown /
     Sharpe of each candidate.
  4. Risk-budget sweep: realised risk of the top-3 consequence-screened
     ensemble as the hard drawdown budget rho_max is tightened (the agent
     abstains, i.e. stays in cash, when no candidate is feasible).

Outputs:
  results/data/ensemble_comparison.json
  results/figures/ensemble_comparison.png
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

OUT = Path(config.DATA_DIR)
FIG = Path(config.FIGURES_DIR)
ANN = config.TRADING_DAYS_PER_YEAR
RF = config.RISK_FREE_RATE
TOP_M = e7.TOP_M
RHO_MAX = e7.RHO_MAX
M_GRID = (1, 2, 3, 5)
RHO_GRID = (0.05, 0.075, 0.10, 0.15, 0.20, 0.30, None)
METRICS = ("sharpe", "ann_return", "cum_return", "max_dd", "calmar", "sortino", "cvar5")


# ───────────────────────────────────────────────────────────────────────────
# Fast numpy metrics on a daily-return vector
# ───────────────────────────────────────────────────────────────────────────
def panel(r: np.ndarray) -> Dict[str, float]:
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return {m: 0.0 for m in METRICS}
    eq = np.cumprod(1.0 + r)
    cum = float(eq[-1] - 1.0)
    ann = float((1.0 + cum) ** (ANN / n) - 1.0) if cum > -1 else -1.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))
    dd = float(np.min(np.concatenate([[1.0], eq]) / peak - 1.0))
    sd = r.std(ddof=1) if n > 1 else 0.0
    sharpe = float((r.mean() * ANN - RF) / (sd * np.sqrt(ANN))) if sd > 1e-12 else 0.0
    down = r[r < 0]
    dsd = down.std(ddof=1) if len(down) > 1 else 0.0
    sortino = float((r.mean() * ANN - RF) / (dsd * np.sqrt(ANN))) if dsd > 1e-12 else 0.0
    q = max(int(np.ceil(0.05 * n)), 1)
    return {"sharpe": sharpe, "ann_return": ann, "cum_return": cum, "max_dd": dd,
            "calmar": ann / abs(dd) if abs(dd) > 1e-9 else 0.0,
            "sortino": sortino, "cvar5": float(np.sort(r)[:q].mean())}


def test_matrix(cands: List[Dict]) -> np.ndarray:
    df = pd.DataFrame({i: c["test_ret"] for i, c in enumerate(cands)}).dropna(how="all").fillna(0.0)
    return df.values


def mean_panel(panels: List[Dict]) -> Dict[str, float]:
    return {m: float(np.mean([p[m] for p in panels])) for m in METRICS}


def order_by_score(cands, rho_max=RHO_MAX, abstain=False):
    k = len(cands)
    feas = [i for i in range(k) if rho_max is None or cands[i]["cvdd"] >= -rho_max]
    if not feas and not abstain:
        feas = list(range(k))
    return sorted(feas, key=lambda i: -cands[i]["lcb"])


# ───────────────────────────────────────────────────────────────────────────
# Per-cell analysis
# ───────────────────────────────────────────────────────────────────────────
def analyse_cell(cell: Dict) -> Dict:
    cands = cell["cands"]
    k = len(cands)
    X = test_matrix(cands)
    single = [panel(X[:, i]) for i in range(k)]
    order = order_by_score(cands)
    res = {
        "random": mean_panel(single),
        "ca2i_base": single[int(np.argmax([c["insample_sr"] for c in cands]))],
        "ew_ensemble": panel(X.mean(axis=1)),
    }
    for m in M_GRID:
        if m > k:
            continue
        res[f"plus_m{m}"] = panel(X[:, order[:m]].mean(axis=1))
        res[f"randsub_m{m}"] = mean_panel([panel(X[:, list(s)].mean(axis=1))
                                           for s in itertools.combinations(range(k), m)])
    res["ca2i_plus"] = res[f"plus_m{TOP_M}"]

    # ex-ante risk-forecast skill (within-cell rank correlations)
    cvdd = [c["cvdd"] for c in cands]
    lcb = [c["lcb"] for c in cands]
    tdd = [p["max_dd"] for p in single]
    tsr = [p["sharpe"] for p in single]
    res["skill"] = {"rho_dd": float(stats.spearmanr(cvdd, tdd).statistic),
                    "rho_sr": float(stats.spearmanr(lcb, tsr).statistic)}

    # risk-budget sweep (abstain = cash when no candidate is feasible)
    sweep, gated, sweep_rets = {}, {}, {}
    for rho in RHO_GRID:
        o = order_by_score(cands, rho_max=rho, abstain=True)[:TOP_M]
        r = X[:, o].mean(axis=1) if o else np.zeros(X.shape[0])
        p = panel(r)
        p["abstain"] = float(len(o) == 0)
        p["n_members"] = float(len(o))
        sweep[str(rho)] = p
        sweep_rets[str(rho)] = r
        # consequence-gated equal weight: diversify over every feasible candidate
        feas = [i for i in range(k) if rho is None or cands[i]["cvdd"] >= -rho]
        g = panel(X[:, feas].mean(axis=1) if feas else np.zeros(X.shape[0]))
        g["feasible_frac"] = len(feas) / k
        g["abstain"] = float(len(feas) == 0)
        gated[str(rho)] = g
    res["sweep"] = sweep
    res["gated"] = gated
    res["_ew_returns"] = X.mean(axis=1)
    res["_sweep_returns"] = sweep_rets
    res["_gated_returns"] = {str(rho): (X[:, [i for i in range(k) if rho is None or cands[i]["cvdd"] >= -rho]]
                                        .mean(axis=1) if any(rho is None or c["cvdd"] >= -rho for c in cands)
                                        else np.zeros(X.shape[0])) for rho in RHO_GRID}
    return res


def nan_safe_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def seed_average(cells: Dict) -> Dict:
    """Average rolling results over seeds for each (window, ticker) pair."""
    grouped: Dict = {}
    for (seed, w, t), r in cells.items():
        grouped.setdefault((w, t), []).append(r)
    out = {}
    for key, rs in grouped.items():
        avg = {}
        for sel in rs[0]:
            if sel.startswith("_"):
                continue
            if sel == "skill":
                avg[sel] = {k: nan_safe_mean([r[sel][k] for r in rs]) for k in rs[0][sel]}
            elif sel in ("sweep", "gated"):
                avg[sel] = {rho: {m: float(np.mean([r[sel][rho][m] for r in rs])) for m in rs[0][sel][rho]}
                            for rho in rs[0][sel]}
            else:
                avg[sel] = {m: float(np.mean([r[sel][m] for r in rs])) for m in METRICS}
        out[key] = avg
    return out


def paired(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = 1.0
    lo, hi = e7.bootstrap_ci(d)
    return {"label": label, "n": int(len(d)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(d.mean()), "t_p": t_p, "wilcoxon_p": w_p,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "boot_ci": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def one_sample(x, label):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    w_p = float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    lo, hi = e7.bootstrap_ci(x)
    return {"label": label, "n": int(len(x)), "mean": float(x.mean()), "wilcoxon_p": w_p,
            "positive": int((x > 0).sum()), "negative": int((x < 0).sum()),
            "boot_ci": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def col(rows, sel, m):
    return [rows[k][sel][m] for k in rows]


def summarise(rows, sels):
    return {s: {m: float(np.mean(col(rows, s, m))) for m in METRICS} for s in sels}


def print_panel(title, summ):
    print(f"\n{title}")
    print(f"{'selector':14s}" + "".join(f"{m:>11s}" for m in METRICS))
    for s, v in summ.items():
        print(f"{s:14s}" + "".join(f"{v[m]:>+11.3f}" for m in METRICS))


def main():
    data = xc.load()
    dep = {t: analyse_cell(c) for t, c in data["deployment"].items()}
    roll_cells = {k: analyse_cell(c) for k, c in data["rolling"].items()}
    roll = seed_average(roll_cells)

    main_sels = ["random", "ca2i_base", "ca2i_plus", "ew_ensemble"]
    dep_sum = summarise(dep, main_sels)
    roll_sum = summarise(roll, main_sels)
    print_panel("=== (1a) Deployment window, GPT-5.4 candidates (K=5, n=10) ===", dep_sum)
    print_panel("=== (1b) Rolling multi-regime (K=10, n=80, 10 seeds) ===", roll_sum)

    tests = {}
    for setting, rows in (("rolling", roll), ("deployment", dep)):
        for m in METRICS:
            tests[f"{setting}_plus_vs_ew_{m}"] = paired(col(rows, "ca2i_plus", m), col(rows, "ew_ensemble", m),
                                                        f"CA2I+ vs EW [{m}, {setting}]")
        for m in ("cum_return", "ann_return"):
            tests[f"{setting}_base_vs_ew_{m}"] = paired(col(rows, "ca2i_base", m), col(rows, "ew_ensemble", m),
                                                        f"CA2I base vs EW [{m}, {setting}]")
    print("\n--- (1c) Paired differences ---")
    for b in tests.values():
        print(f"  {b['label']:42s} diff={b['mean_diff']:+.4f} t_p={b['t_p']:.3f} W_p={b['wilcoxon_p']:.3f} "
              f"{b['wins']}+/{b['losses']}- CI=[{b['boot_ci'][0]:+.4f},{b['boot_ci'][1]:+.4f}] "
              f"sig={b['ci_excludes_zero']}")

    # (2) equal-size control
    ctrl = {}
    print("\n=== (2) Consequence-ranked top-m vs. random m-subsets (rolling, n=80) ===")
    for m in M_GRID:
        for metric in ("sharpe", "max_dd", "sortino"):
            ctrl[f"m{m}_{metric}"] = paired(col(roll, f"plus_m{m}", metric), col(roll, f"randsub_m{m}", metric),
                                            f"top-{m} vs random-{m} [{metric}]")
        b = ctrl[f"m{m}_sharpe"]
        print(f"  m={m}: Sharpe top={b['mean_a']:+.3f} rand={b['mean_b']:+.3f} diff={b['mean_diff']:+.3f} "
              f"t_p={b['t_p']:.4f} W_p={b['wilcoxon_p']:.4f} {b['wins']}+/{b['losses']}- "
              f"CI=[{b['boot_ci'][0]:+.3f},{b['boot_ci'][1]:+.3f}] sig={b['ci_excludes_zero']}  | "
              f"DD diff={ctrl[f'm{m}_max_dd']['mean_diff']:+.4f} (sig={ctrl[f'm{m}_max_dd']['ci_excludes_zero']})")
    dep_ctrl = paired(col(dep, f"plus_m{TOP_M}", "sharpe"), col(dep, f"randsub_m{TOP_M}", "sharpe"),
                      "top-3 vs random-3 [sharpe, deployment]")
    print(f"  deployment m=3: diff={dep_ctrl['mean_diff']:+.3f} W_p={dep_ctrl['wilcoxon_p']:.3f}")

    # (3) risk-forecast skill
    skill = {"rolling_rho_dd": one_sample([roll[k]["skill"]["rho_dd"] for k in roll], "Spearman(CV DD, test DD), rolling"),
             "rolling_rho_sr": one_sample([roll[k]["skill"]["rho_sr"] for k in roll], "Spearman(CV score, test SR), rolling"),
             "deploy_rho_dd": one_sample([dep[k]["skill"]["rho_dd"] for k in dep], "Spearman(CV DD, test DD), deployment"),
             "deploy_rho_sr": one_sample([dep[k]["skill"]["rho_sr"] for k in dep], "Spearman(CV score, test SR), deployment")}
    print("\n=== (3) Ex-ante risk-forecast skill of CAM ===")
    for b in skill.values():
        print(f"  {b['label']:42s} mean rho={b['mean']:+.3f} W_p={b['wilcoxon_p']:.4f} "
              f"{b['positive']}+/{b['negative']}- CI=[{b['boot_ci'][0]:+.3f},{b['boot_ci'][1]:+.3f}] "
              f"sig={b['ci_excludes_zero']}")

    # (4) risk-budget sweep
    sweep = {}
    print("\n=== (4) Risk-budget sweep, top-3 screened ensemble (rolling, n=80) ===")
    print(f"  {'rho_max':>8s} {'meanDD':>8s} {'worstDD':>8s} {'DD<-10%':>8s} {'Sharpe':>8s} {'cum':>8s} {'abstain':>8s}")
    for rho in map(str, RHO_GRID):
        dds = np.array([roll[k]["sweep"][rho]["max_dd"] for k in roll])
        sweep[rho] = {"mean_dd": float(dds.mean()), "worst_dd": float(dds.min()),
                      "n_dd_below_10pct": int((dds < -0.10).sum()),
                      "sharpe": float(np.mean([roll[k]["sweep"][rho]["sharpe"] for k in roll])),
                      "cum_return": float(np.mean([roll[k]["sweep"][rho]["cum_return"] for k in roll])),
                      "abstain_rate": float(np.mean([roll[k]["sweep"][rho]["abstain"] for k in roll]))}
        s = sweep[rho]
        print(f"  {rho:>8s} {s['mean_dd']:>+8.3f} {s['worst_dd']:>+8.3f} {s['n_dd_below_10pct']:>8d} "
              f"{s['sharpe']:>+8.3f} {s['cum_return']:>+8.3f} {s['abstain_rate']:>8.3f}")
    ew_dd = np.array(col(roll, "ew_ensemble", "max_dd"))
    sweep["ew_reference"] = {"mean_dd": float(ew_dd.mean()), "worst_dd": float(ew_dd.min()),
                             "n_dd_below_10pct": int((ew_dd < -0.10).sum()),
                             "sharpe": roll_sum["ew_ensemble"]["sharpe"],
                             "cum_return": roll_sum["ew_ensemble"]["cum_return"]}
    print(f"  {'EW':>8s} {ew_dd.mean():>+8.3f} {ew_dd.min():>+8.3f} {int((ew_dd < -0.10).sum()):>8d} "
          f"{roll_sum['ew_ensemble']['sharpe']:>+8.3f} {roll_sum['ew_ensemble']['cum_return']:>+8.3f}")
    rho_tight, rho_loose = "0.05", "None"
    sweep_test = paired([roll[k]["sweep"][rho_tight]["max_dd"] for k in roll],
                        [roll[k]["sweep"][rho_loose]["max_dd"] for k in roll],
                        "DD at rho=5% vs unconstrained")
    print(f"  tight vs unconstrained DD diff={sweep_test['mean_diff']:+.4f} W_p={sweep_test['wilcoxon_p']:.4g} "
          f"sig={sweep_test['ci_excludes_zero']}")
    ew_vs_tight = paired([roll[k]["sweep"][rho_tight]["max_dd"] for k in roll], list(ew_dd),
                         "DD at rho=5% vs EW")
    print(f"  rho=5% vs EW DD diff={ew_vs_tight['mean_diff']:+.4f} W_p={ew_vs_tight['wilcoxon_p']:.4g} "
          f"sig={ew_vs_tight['ci_excludes_zero']}")

    # (5) consequence-gated equal weight vs plain equal weight
    print("\n=== (5) Consequence-gated equal-weight (diversify over feasible set) vs EW ===")
    gated_out = {}
    for setting, rows in (("rolling", roll), ("deployment", dep)):
        for rho in ("0.075", "0.1", "0.15", "0.2"):
            g = {m: [rows[k]["gated"][rho][m] for k in rows] for m in ("sharpe", "max_dd", "cum_return", "cvar5")}
            blk = {m: paired(g[m], col(rows, "ew_ensemble", m), f"gated EW rho={rho} vs EW [{m}, {setting}]")
                   for m in g}
            blk["worst_dd"] = [float(np.min(g["max_dd"])), float(np.min(col(rows, "ew_ensemble", "max_dd")))]
            blk["feasible_frac"] = float(np.mean([rows[k]["gated"][rho]["feasible_frac"] for k in rows]))
            gated_out[f"{setting}_{rho}"] = blk
            print(f"  {setting:10s} rho={rho:5s} feasible={blk['feasible_frac']:.2f} | "
                  + " | ".join(f"{m} {blk[m]['mean_a']:+.4f} vs {blk[m]['mean_b']:+.4f} "
                               f"(W_p={blk[m]['wilcoxon_p']:.3f}, sig={blk[m]['ci_excludes_zero']})"
                               for m in ("sharpe", "max_dd", "cum_return", "cvar5"))
                  + f" | worst DD {blk['worst_dd'][0]:+.3f} vs {blk['worst_dd'][1]:+.3f}")

    # (6) exposure-matched control: EW de-levered (ex post) to the same mean drawdown
    print("\n=== (6) Exposure-matched control: EW scaled by alpha to equal mean drawdown (rolling) ===")
    grouped = {}
    for (seed, w, t), r in roll_cells.items():
        grouped.setdefault((w, t), []).append(r)
    delever = {}
    for label, getter in (("ca2i_plus_rho0.075", lambda r: r["_sweep_returns"]["0.075"]),
                          ("ca2i_plus_rho0.1", lambda r: r["_sweep_returns"]["0.1"]),
                          ("gated_ew_rho0.075", lambda r: r["_gated_returns"]["0.075"]),
                          ("gated_ew_rho0.1", lambda r: r["_gated_returns"]["0.1"])):
        def cell_means(alpha, src):
            rows_ = []
            for rs in grouped.values():
                ps = [panel(src(r)) for r in rs]
                rows_.append({m: float(np.mean([p[m] for p in ps])) for m in ("sharpe", "max_dd", "cum_return", "cvar5")})
            return rows_
        target = cell_means(1.0, getter)
        tgt_dd = np.mean([x["max_dd"] for x in target])
        lo, hi = 0.01, 1.0
        for _ in range(30):
            a = (lo + hi) / 2
            dd = np.mean([x["max_dd"] for x in cell_means(a, lambda r, a=a: a * r["_ew_returns"])])
            hi, lo = (a, lo) if dd < tgt_dd else (hi, a)
        ew_s = cell_means(a, lambda r, a=a: a * r["_ew_returns"])
        blk = {m: paired([x[m] for x in target], [x[m] for x in ew_s], f"{label} vs EW x{a:.2f} [{m}]")
               for m in ("sharpe", "cum_return", "cvar5")}
        blk["alpha"] = float(a)
        blk["worst_dd"] = [float(min(x["max_dd"] for x in target)), float(min(x["max_dd"] for x in ew_s))]
        delever[label] = blk
        print(f"  {label:20s} alpha={a:.3f} | " + " | ".join(
            f"{m} {blk[m]['mean_a']:+.4f} vs {blk[m]['mean_b']:+.4f} (W_p={blk[m]['wilcoxon_p']:.3f})"
            for m in ("sharpe", "cum_return", "cvar5")) + f" | worst DD {blk['worst_dd'][0]:+.3f} vs {blk['worst_dd'][1]:+.3f}")

    print("\n=== (7) Risk-budgeted CA2I+ vs EW (rolling) ===")
    budget_vs_ew = {}
    for rho in ("0.075", "0.1"):
        for m in ("sharpe", "max_dd", "cum_return", "cvar5"):
            b = paired([roll[k]["sweep"][rho][m] for k in roll], col(roll, "ew_ensemble", m),
                       f"CA2I+ rho={rho} vs EW [{m}, rolling]")
            budget_vs_ew[f"{rho}_{m}"] = b
            print(f"  {b['label']:42s} {b['mean_a']:+.4f} vs {b['mean_b']:+.4f} W_p={b['wilcoxon_p']:.3g} "
                  f"CI=[{b['boot_ci'][0]:+.4f},{b['boot_ci'][1]:+.4f}] sig={b['ci_excludes_zero']}")

    out = {"deployment": dep_sum, "rolling": roll_sum, "tests": tests,
           "gated_equal_weight": gated_out, "exposure_matched_control": delever,
           "risk_budget_vs_ew": budget_vs_ew,
           "equal_size_control": ctrl, "equal_size_control_deployment": dep_ctrl,
           "risk_forecast_skill": skill, "risk_budget_sweep": sweep,
           "sweep_tight_vs_unconstrained": sweep_test, "sweep_tight_vs_ew": ew_vs_tight,
           "ablation_rolling": {f"m{m}": {"top": float(np.mean(col(roll, f'plus_m{m}', 'sharpe'))),
                                          "random": float(np.mean(col(roll, f'randsub_m{m}', 'sharpe')))}
                                for m in M_GRID}}
    with open(OUT / "ensemble_comparison.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(M_GRID) + 1)
    top = [out["ablation_rolling"][f"m{m}"]["top"] for m in M_GRID] + [roll_sum["ew_ensemble"]["sharpe"]]
    rnd = [out["ablation_rolling"][f"m{m}"]["random"] for m in M_GRID] + [roll_sum["ew_ensemble"]["sharpe"]]
    ax1.bar(x - 0.2, rnd, 0.4, color=config.COLOR_RANDOM_BASELINE, label="Random $m$-subset (expected)")
    ax1.bar(x + 0.2, top, 0.4, color=config.COLOR_CONSEQUENCE_AWARE, label="Consequence-ranked top-$m$")
    ax1.set_xticks(x); ax1.set_xticklabels([f"m={m}" for m in M_GRID] + ["m=K=10\n(equal-weight)"], fontsize=8)
    ax1.set_ylabel("Mean test Sharpe (rolling, $n$=80)")
    ax1.set_title("(a) Value of the consequence ranking at fixed ensemble size")
    ax1.axhline(0, color="k", lw=0.5); ax1.legend(fontsize=8, loc="upper left")

    rhos = [r for r in map(str, RHO_GRID)]
    lab = [f"{float(r)*100:g}%" if r != "None" else "none" for r in rhos]
    ax2.plot(range(len(rhos)), [-100 * sweep[r]["mean_dd"] for r in rhos], "o-",
             color=config.COLOR_CONSEQUENCE_AWARE, label="CA$^2$I$^+$ mean max DD")
    ax2.plot(range(len(rhos)), [-100 * sweep[r]["worst_dd"] for r in rhos], "s--",
             color="#1565C0", label="CA$^2$I$^+$ worst max DD")
    ax2.axhline(-100 * sweep["ew_reference"]["mean_dd"], color="#9E9E9E", ls="-", lw=1.2, label="Equal-weight mean max DD")
    ax2.axhline(-100 * sweep["ew_reference"]["worst_dd"], color="#9E9E9E", ls=":", lw=1.2, label="Equal-weight worst max DD")
    ax2.set_xticks(range(len(rhos))); ax2.set_xticklabels(lab)
    ax2.set_xlabel(r"Hard drawdown budget $\rho_{\max}$ (on CV folds)")
    ax2.set_ylabel("Realised test drawdown (%)")
    ax2.set_title("(b) Ex-ante risk budget controls realised risk")
    ax2.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "ensemble_comparison.png", dpi=config.FIGURE_DPI)
    print("\nSaved ensemble_comparison.json and ensemble_comparison.png")


if __name__ == "__main__":
    main()
