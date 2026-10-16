"""
extension_09_generate_and_rank.py
===================================================================
Second-round reviewer comment R2.1: "While the authors emphasise that CA2I
differs from generate-and-rank by operating on simulated outcomes rather
than textual proxies, the manuscript could benefit from a more systematic
comparison."

We hold the candidate pool fixed and vary only *what signal the ranker
reads*, producing a controlled ablation ladder:

  Vanilla      : execute a proposal without ranking (expected over proposals).
  Text-desc    : generate-and-rank on the candidate's own natural-language
                 description/rationale (a transparent lexicon-based judge of
                 stated safety and confidence; stands in for any ranker whose
                 input is the proposal text).
  Text-spec    : generate-and-rank on the *declared* specification read on
                 paper (tightest stated stop-loss = "safest"), no simulation.
  CA2I (base)  : rank on simulated outcomes (in-sample Sharpe).
  CA2I+        : rank on cross-validated simulated outcomes, top-3 ensemble.

Manipulation test: in every cell we rewrite the description of one candidate
so that it reads as a conservative, capital-preserving strategy, leaving its
executable parameters untouched. Two decoys are used: the candidate with
CAM's lowest score, and the candidate CAM flags as riskiest (deepest
cross-validated drawdown). A text-based ranker
can be steered by this edit; CAM's input (the simulated consequence vector) is
unchanged, so its decision is invariant by construction. We measure the
selection rate of the relabelled candidate and the realised outcome.

We also time CAM per candidate to quantify its test-time cost, which is CPU
simulation rather than additional LLM tokens.

Outputs:
  results/data/generate_and_rank.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

import config
import consequence_assessment as ca
import extension_cache as xc
import extension_common as ec
import extension_07_improved_selector as e7
from extension_08_ensemble_comparison import panel, test_matrix, order_by_score, paired

OUT = Path(config.DATA_DIR)

POSITIVE = ["low-risk", "low risk", "conservative", "robust", "stable", "capital preservation",
            "preserve capital", "disciplined", "protect", "protective", "tight", "safe", "reliable",
            "controlled", "proven", "consistent", "high-probability", "defensive", "steady", "prudent"]
NEGATIVE = ["aggressive", "volatile", "volatility", "risky", "speculative", "high-risk", "whipsaw",
            "drawdown", "leverage", "chase", "chasing", "explosive", "spike"]
DECOY_TEXT = ("A conservative, low-risk capital preservation strategy with tight, disciplined risk "
              "controls. It is robust and stable across regimes, protective in drawdowns, and "
              "designed as a prudent, defensive, high-probability approach with reliable, "
              "consistent and controlled exposure.")


def text_score(text: str) -> float:
    t = text.lower()
    pos = sum(len(re.findall(re.escape(w), t)) for w in POSITIVE)
    neg = sum(len(re.findall(re.escape(w), t)) for w in NEGATIVE)
    return float(pos - neg) / max(len(t.split()), 1) * 100.0


def declared_stop(params: Dict) -> float:
    return float(params.get("stop_loss_pct", params.get("trailing_stop_pct", 1.0)))


def pick(cands: List[Dict], texts: List[str]) -> Dict[str, int]:
    return {
        "text_desc": int(np.argmax([text_score(x) for x in texts])),
        "text_spec": int(np.argmin([declared_stop(c["params"]) for c in cands])),
        "ca2i_base": int(np.argmax([c["insample_sr"] for c in cands])),
    }


def analyse_cell(cell: Dict) -> Dict:
    cands = cell["cands"]
    X = test_matrix(cands)
    single = [panel(X[:, i]) for i in range(len(cands))]
    texts = [c["text"] for c in cands]
    order = order_by_score(cands)
    res = {"vanilla": {m: float(np.mean([p[m] for p in single])) for m in ("sharpe", "max_dd", "cum_return")}}
    picks = pick(cands, texts)
    for name, i in picks.items():
        res[name] = {m: single[i][m] for m in ("sharpe", "max_dd", "cum_return")}
    p = panel(X[:, order[:e7.TOP_M]].mean(axis=1))
    res["ca2i_plus"] = {m: p[m] for m in ("sharpe", "max_dd", "cum_return")}

    # manipulation: relabel (a) the candidate with CAM's lowest score and
    # (b) the candidate CAM flags as riskiest (deepest cross-validated drawdown)
    order_m = order_by_score(cands)               # CAM never reads text
    for tag, worst in (("score", int(np.argmin([c["lcb"] for c in cands]))),
                       ("risk", int(np.argmin([c["cvdd"] for c in cands])))):
        manip_texts = list(texts)
        manip_texts[worst] = DECOY_TEXT
        picks_m = pick(cands, manip_texts)
        res[f"manip_{tag}"] = {
            "text_desc_selects_decoy_before": float(picks["text_desc"] == worst),
            "text_desc_selects_decoy_after": float(picks_m["text_desc"] == worst),
            "cam_base_selects_decoy_before": float(picks["ca2i_base"] == worst),
            "cam_base_selects_decoy_after": float(picks_m["ca2i_base"] == worst),
            "cam_plus_includes_decoy_before": float(worst in order[:e7.TOP_M]),
            "cam_plus_includes_decoy_after": float(worst in order_m[:e7.TOP_M]),
            "text_desc_sharpe_before": single[picks["text_desc"]]["sharpe"],
            "text_desc_dd_before": single[picks["text_desc"]]["max_dd"],
            "text_desc_sharpe_after": single[picks_m["text_desc"]]["sharpe"],
            "text_desc_dd_after": single[picks_m["text_desc"]]["max_dd"],
            "decoy_sharpe": single[worst]["sharpe"], "decoy_dd": single[worst]["max_dd"],
            "pool_mean_dd": float(np.mean([p["max_dd"] for p in single])),
        }
    return res


def seed_average(cells):
    grouped = {}
    for (seed, w, t), r in cells.items():
        grouped.setdefault((w, t), []).append(r)
    return {k: {sel: {m: float(np.mean([r[sel][m] for r in rs])) for m in rs[0][sel]} for sel in rs[0]}
            for k, rs in grouped.items()}


def time_cam() -> Dict[str, float]:
    t = config.TICKERS[0]
    train, test = ec.load_prices(t, "train"), ec.load_prices(t, "test")
    strategies = ec.load_strategies(t)
    t0 = time.perf_counter()
    for s in strategies:
        ca.backtest_strategy(s, train)
    t1 = time.perf_counter()
    for s in strategies:
        e7.cv_scores(s, train)
    t2 = time.perf_counter()
    return {"insample_ms_per_candidate": 1000 * (t1 - t0) / len(strategies),
            "cv_ms_per_candidate": 1000 * (t2 - t1) / len(strategies),
            "llm_tokens_added": 0}


def main():
    import logging
    logging.disable(logging.INFO)
    data = xc.load()
    dep = {t: analyse_cell(c) for t, c in data["deployment"].items()}
    roll = seed_average({k: analyse_cell(c) for k, c in data["rolling"].items()})
    ladder = ["vanilla", "text_desc", "text_spec", "ca2i_base", "ca2i_plus"]

    out = {"ladder": {}, "tests": {}, "manipulation": {}}
    for setting, rows in (("deployment", dep), ("rolling", roll)):
        out["ladder"][setting] = {s: {m: float(np.mean([rows[k][s][m] for k in rows]))
                                      for m in ("sharpe", "max_dd", "cum_return")} for s in ladder}
        for s in ("text_desc", "text_spec", "ca2i_base"):
            for m in ("sharpe", "max_dd"):
                out["tests"][f"{setting}_plus_vs_{s}_{m}"] = paired([rows[k]["ca2i_plus"][m] for k in rows],
                                                                     [rows[k][s][m] for k in rows],
                                                                     f"CA2I+ vs {s} [{m}, {setting}]")
        for s in ("text_desc", "text_spec"):
            out["tests"][f"{setting}_{s}_vs_vanilla"] = paired([rows[k][s]["sharpe"] for k in rows],
                                                                [rows[k]["vanilla"]["sharpe"] for k in rows],
                                                                f"{s} vs vanilla [sharpe, {setting}]")
        for tag in ("score", "risk"):
            key = f"manip_{tag}"
            out["manipulation"][f"{setting}_{tag}"] = {m: float(np.mean([rows[k][key][m] for k in rows]))
                                                      for m in next(iter(rows.values()))[key]}
            out["manipulation"][f"{setting}_{tag}_dd_test"] = paired(
                [rows[k][key]["text_desc_dd_after"] for k in rows], [rows[k][key]["text_desc_dd_before"] for k in rows],
                f"text ranker DD after vs before relabelling ({tag} decoy, {setting})")
    out["timing"] = time_cam()

    for setting in ("deployment", "rolling"):
        print(f"\n=== Ablation ladder ({setting}) ===")
        for s, v in out["ladder"][setting].items():
            print(f"  {s:10s} Sharpe={v['sharpe']:+.3f}  MaxDD={v['max_dd']:+.3f}  cum={v['cum_return']:+.4f}")
        for tag in ("score", "risk"):
            mp = out["manipulation"][f"{setting}_{tag}"]
            t = out["manipulation"][f"{setting}_{tag}_dd_test"]
            print(f"  [{tag} decoy] text ranker selects it {100*mp['text_desc_selects_decoy_before']:.0f}% -> "
                  f"{100*mp['text_desc_selects_decoy_after']:.0f}%; CAM base {100*mp['cam_base_selects_decoy_before']:.0f}% -> "
                  f"{100*mp['cam_base_selects_decoy_after']:.0f}%; CAM+ includes it {100*mp['cam_plus_includes_decoy_before']:.0f}% -> "
                  f"{100*mp['cam_plus_includes_decoy_after']:.0f}%")
            print(f"      decoy realised SR={mp['decoy_sharpe']:+.3f} DD={mp['decoy_dd']:+.3f} (pool mean DD {mp['pool_mean_dd']:+.3f}); "
                  f"text ranker SR {mp['text_desc_sharpe_before']:+.3f}->{mp['text_desc_sharpe_after']:+.3f}, "
                  f"DD {mp['text_desc_dd_before']:+.3f}->{mp['text_desc_dd_after']:+.3f} "
                  f"(W_p={t['wilcoxon_p']:.3g}, sig={t['ci_excludes_zero']})")
    print("\n--- Paired tests ---")
    for b in out["tests"].values():
        print(f"  {b['label']:44s} diff={b['mean_diff']:+.3f} t_p={b['t_p']:.4f} W_p={b['wilcoxon_p']:.4f} "
              f"{b['wins']}+/{b['losses']}- CI=[{b['boot_ci'][0]:+.3f},{b['boot_ci'][1]:+.3f}] sig={b['ci_excludes_zero']}")
    print(f"\nCAM cost: {out['timing']['insample_ms_per_candidate']:.1f} ms (in-sample) + "
          f"{out['timing']['cv_ms_per_candidate']:.1f} ms (5-fold CV) per candidate, 0 extra LLM tokens")
    with open(OUT / "generate_and_rank.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved generate_and_rank.json")


if __name__ == "__main__":
    main()
