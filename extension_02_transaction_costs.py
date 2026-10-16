"""
extension_02_transaction_costs.py
===================================================================
Reviewer comments addressed: R2.6 (commissions + slippage) and the specific
turnover-bias hypothesis ("consequence-aware selection could systematically
prefer higher-turnover strategies").

Method
------
We re-evaluate the realised TEST-period performance of every method under a
realistic round-trip friction model. Because positions are binary (fully
invested or fully in cash), every trade incurs a cost c on entry and a cost c
on exit, where c = (commission + slippage) in basis points. We rebuild each
strategy's NET equity curve by applying a multiplicative (1 - c) factor to the
capital base at every entry and every exit date, then recompute net cumulative
return and net annualised Sharpe from the net daily returns. This is exact for
the binary-position backtester (no approximation of per-trade compounding).

We sweep c in {0, 5, 10, 20, 50} bps per side and report, for each cost level,
the mean NET Sharpe and NET return of CA2I, the return-greedy heuristic, and
buy-and-hold (B&H trades exactly once, so it is essentially cost-free).

Turnover-bias test
------------------
We compare the mean number of round-trip trades of the CA2I-selected strategy
against (i) the strategy a return-greedy rule would select and (ii) the average
over all K candidates. If CA2I-selected turnover <= candidate-average turnover,
the reviewer's turnover-bias concern is not supported by the data.

Outputs: results/data/transaction_costs.csv
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
import extension_common as ec

warnings.filterwarnings("ignore")
OUT = Path(config.DATA_DIR)
COST_GRID_BPS = [0, 5, 10, 20, 50]


def net_equity(result, cost_bps: float) -> pd.Series:
    """Apply round-trip friction to a backtest equity curve. Each entry and
    each exit multiplies the capital base by (1 - c)."""
    eq = result.equity_curve.copy()
    c = cost_bps / 10000.0
    if c <= 0 or not result.trades:
        return eq
    factor = pd.Series(1.0, index=eq.index)
    for tr in result.trades:
        for d in (tr.entry_date, tr.exit_date):
            ts = pd.Timestamp(d)
            # apply cost to all dates on/after the trade event
            factor.loc[factor.index >= ts] *= (1.0 - c)
    return eq * factor


def net_metrics(result, cost_bps: float):
    eq = net_equity(result, cost_bps)
    cum = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    sr = ec.sharpe_from_returns(eq.pct_change().dropna())
    return cum, sr


def main():
    rows = []
    # turnover accumulators
    ca2i_turn, ret_turn, cand_turn = [], [], []

    cache = {t: ec.evaluate_candidates(t) for t in config.TICKERS}

    for cost in COST_GRID_BPS:
        ca2i_ret, ca2i_sr = [], []
        rg_ret, rg_sr = [], []
        bh_ret, bh_sr = [], []
        for t in config.TICKERS:
            ev = cache[t]
            train, test, bh = ev["train"], ev["test"], ev["bh"]
            idx_ca2i = int(np.argmax([r.sharpe_ratio for r in train]))
            idx_rg = int(np.argmax([r.cumulative_return for r in train]))

            c1, s1 = net_metrics(test[idx_ca2i], cost)
            c2, s2 = net_metrics(test[idx_rg], cost)
            c3, s3 = net_metrics(bh, cost)
            ca2i_ret.append(c1); ca2i_sr.append(s1)
            rg_ret.append(c2); rg_sr.append(s2)
            bh_ret.append(c3); bh_sr.append(s3)

            if cost == 0:  # turnover only needs to be computed once
                ca2i_turn.append(test[idx_ca2i].total_trades)
                ret_turn.append(test[idx_rg].total_trades)
                cand_turn.append(np.mean([r.total_trades for r in test]))

        rows.append({
            "cost_bps_per_side": cost,
            "CA2I_net_sharpe": np.mean(ca2i_sr),
            "CA2I_net_return": np.mean(ca2i_ret),
            "ReturnGreedy_net_sharpe": np.mean(rg_sr),
            "ReturnGreedy_net_return": np.mean(rg_ret),
            "BuyHold_net_sharpe": np.mean(bh_sr),
            "BuyHold_net_return": np.mean(bh_ret),
        })

    with open(OUT / "transaction_costs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== NET TEST PERFORMANCE vs ROUND-TRIP COST (R2.6) ===")
    print(f"{'cost(bps)':>9} | {'CA2I SR':>8} {'CA2I ret':>9} | "
          f"{'RG SR':>7} {'RG ret':>8} | {'B&H SR':>7} {'B&H ret':>8}")
    for r in rows:
        print(f"{r['cost_bps_per_side']:>9} | {r['CA2I_net_sharpe']:>+8.3f} "
              f"{r['CA2I_net_return']:>+9.2%} | {r['ReturnGreedy_net_sharpe']:>+7.3f} "
              f"{r['ReturnGreedy_net_return']:>+8.2%} | {r['BuyHold_net_sharpe']:>+7.3f} "
              f"{r['BuyHold_net_return']:>+8.2%}")

    print("\n=== TURNOVER-BIAS TEST (R2.6) ===")
    print(f"Mean #round-trip trades, CA2I-selected      : {np.mean(ca2i_turn):.2f}")
    print(f"Mean #round-trip trades, return-greedy       : {np.mean(ret_turn):.2f}")
    print(f"Mean #round-trip trades, all candidates (avg): {np.mean(cand_turn):.2f}")
    verdict = ("NOT supported" if np.mean(ca2i_turn) <= np.mean(cand_turn)
               else "partially supported")
    print(f"Turnover-bias hypothesis is {verdict}: CA2I-selected turnover "
          f"({np.mean(ca2i_turn):.2f}) vs candidate average ({np.mean(cand_turn):.2f}).")


if __name__ == "__main__":
    main()
