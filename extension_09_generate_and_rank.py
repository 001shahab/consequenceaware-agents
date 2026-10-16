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


def active_mean(X: np.ndarray) -> np.ndarray:
    active = np.abs(X) > 1e-12
    n_act = active.sum(axis=1)
    return np.where(n_act > 0, X.sum(axis=1) / np.maximum(n_act, 1), 0.0)


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
    pa = panel(active_mean(X[:, order[:e7.TOP_M]]))
    res["ca2i_plus_active"] = {m: pa[m] for m in ("sharpe", "max_dd", "cum_return")}
    feas75 = [i for i in order if abs(cands[i]["cvdd"]) <= 0.075]
    p75 = panel(X[:, feas75[:e7.TOP_M]].mean(axis=1) if feas75 else np.zeros(X.shape[0]))
    res["ca2i_plus_75"] = {m: p75[m] for m in ("sharpe", "max_dd", "cum_return")}

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
    llm_cache_path = OUT / "llm_judge_cache.json"
    llm_cache = json.loads(llm_cache_path.read_text()) if llm_cache_path.exists() else None
    dep = {t: analyse_cell(c) for t, c in data["deployment"].items()}
    roll_cells = {k: analyse_cell(c) for k, c in data["rolling"].items()}
    roll = seed_average(roll_cells)

    if llm_cache is not None:
        for t, cell in data["deployment"].items():
            cands = cell["cands"]
            X = test_matrix(cands)
            single = [panel(X[:, i]) for i in range(len(cands))]
            info = llm_cache["deployment"][t]
            ib, iar, ias = info["pick_before"], info["pick_after_risk"], info["pick_after_score"]
            worst_r = int(np.argmin([c["cvdd"] for c in cands]))
            worst_s = int(np.argmin([c["lcb"] for c in cands]))
            dep[t]["llm_judge"] = {m: single[ib][m] for m in ("sharpe", "max_dd", "cum_return")}
            dep[t]["manip_risk"]["llm_judge_selects_decoy_before"] = float(ib == worst_r)
            dep[t]["manip_risk"]["llm_judge_selects_decoy_after"] = float(iar == worst_r)
            dep[t]["manip_risk"]["llm_judge_sharpe_before"] = single[ib]["sharpe"]
            dep[t]["manip_risk"]["llm_judge_dd_before"] = single[ib]["max_dd"]
            dep[t]["manip_risk"]["llm_judge_sharpe_after"] = single[iar]["sharpe"]
            dep[t]["manip_risk"]["llm_judge_dd_after"] = single[iar]["max_dd"]
            dep[t]["manip_score"]["llm_judge_selects_decoy_before"] = float(ib == worst_s)
            dep[t]["manip_score"]["llm_judge_selects_decoy_after"] = float(ias == worst_s)
            dep[t]["manip_score"]["llm_judge_sharpe_before"] = single[ib]["sharpe"]
            dep[t]["manip_score"]["llm_judge_dd_before"] = single[ib]["max_dd"]
            dep[t]["manip_score"]["llm_judge_sharpe_after"] = single[ias]["sharpe"]
            dep[t]["manip_score"]["llm_judge_dd_after"] = single[ias]["max_dd"]
        for (w, t) in roll:
            cell = data["rolling"][(0, w, t)]
            cands = cell["cands"]
            X = test_matrix(cands)
            single = [panel(X[:, i]) for i in range(len(cands))]
            info = llm_cache["rolling"][f"{w}__{t}"]
            ib, iar, ias = info["pick_before"], info["pick_after_risk"], info["pick_after_score"]
            worst_r = int(np.argmin([c["cvdd"] for c in cands]))
            worst_s = int(np.argmin([c["lcb"] for c in cands]))
            roll[(w, t)]["llm_judge"] = {m: single[ib][m] for m in ("sharpe", "max_dd", "cum_return")}
            roll[(w, t)]["manip_risk"]["llm_judge_selects_decoy_before"] = float(ib == worst_r)
            roll[(w, t)]["manip_risk"]["llm_judge_selects_decoy_after"] = float(iar == worst_r)
            roll[(w, t)]["manip_risk"]["llm_judge_sharpe_before"] = single[ib]["sharpe"]
            roll[(w, t)]["manip_risk"]["llm_judge_dd_before"] = single[ib]["max_dd"]
            roll[(w, t)]["manip_risk"]["llm_judge_sharpe_after"] = single[iar]["sharpe"]
            roll[(w, t)]["manip_risk"]["llm_judge_dd_after"] = single[iar]["max_dd"]
            roll[(w, t)]["manip_score"]["llm_judge_selects_decoy_before"] = float(ib == worst_s)
            roll[(w, t)]["manip_score"]["llm_judge_selects_decoy_after"] = float(ias == worst_s)
            roll[(w, t)]["manip_score"]["llm_judge_sharpe_before"] = single[ib]["sharpe"]
            roll[(w, t)]["manip_score"]["llm_judge_dd_before"] = single[ib]["max_dd"]
            roll[(w, t)]["manip_score"]["llm_judge_sharpe_after"] = single[ias]["sharpe"]
            roll[(w, t)]["manip_score"]["llm_judge_dd_after"] = single[ias]["max_dd"]

    ladder = ["vanilla", "text_desc", "text_spec", "llm_judge", "ca2i_base", "ca2i_plus", "ca2i_plus_active", "ca2i_plus_75"] if llm_cache else ["vanilla", "text_desc", "text_spec", "ca2i_base", "ca2i_plus", "ca2i_plus_active", "ca2i_plus_75"]

    out = {"ladder": {}, "tests": {}, "manipulation": {}}
    for setting, rows in (("deployment", dep), ("rolling", roll)):
        out["ladder"][setting] = {s: {m: float(np.mean([rows[k][s][m] for k in rows]))
                                      for m in ("sharpe", "max_dd", "cum_return")} for s in ladder}
        for s in [x for x in ("text_desc", "text_spec", "llm_judge", "ca2i_base") if x in ladder]:
            for m in ("sharpe", "max_dd"):
                out["tests"][f"{setting}_plus_vs_{s}_{m}"] = paired([rows[k]["ca2i_plus"][m] for k in rows],
                                                                     [rows[k][s][m] for k in rows],
                                                                     f"CA2I+ vs {s} [{m}, {setting}]")
                out["tests"][f"{setting}_plus_active_vs_{s}_{m}"] = paired([rows[k]["ca2i_plus_active"][m] for k in rows],
                                                                           [rows[k][s][m] for k in rows],
                                                                           f"CA2I+active vs {s} [{m}, {setting}]")
                out["tests"][f"{setting}_plus_75_vs_{s}_{m}"] = paired([rows[k]["ca2i_plus_75"][m] for k in rows],
                                                                       [rows[k][s][m] for k in rows],
                                                                       f"CA2I+75 vs {s} [{m}, {setting}]")
        for s in [x for x in ("text_desc", "text_spec", "llm_judge") if x in ladder]:
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
    if llm_cache is not None:
        out["timing"]["llm_judge_model"] = llm_cache["model"]
        out["timing"]["llm_judge_deploy_ms_per_cell"] = float(np.mean([v["ms"] for v in llm_cache["deployment"].values()]))
        out["timing"]["llm_judge_deploy_tokens_per_cell"] = float(np.mean([v["tokens"] for v in llm_cache["deployment"].values()]))
        out["timing"]["llm_judge_rolling_ms_per_cell"] = float(np.mean([v["ms"] for v in llm_cache["rolling"].values()]))
        out["timing"]["llm_judge_rolling_tokens_per_cell"] = float(np.mean([v["tokens"] for v in llm_cache["rolling"].values()]))

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
