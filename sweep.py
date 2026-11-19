#!/usr/bin/env python3
"""Score the real replay CSVs under several confidence filters.

Runs the model once per tournament, then reuses the same draws for every
(min-edge, min-sure) pair so the comparison is not confounded by seed noise.

    python3 sweep.py
    python3 sweep.py --draws 2000 --dir replays
"""
import argparse
from pathlib import Path

import numpy as np

from backtest import mean_se, profit
from bracket_edge import FEE, GAME_NOISE, PRIOR_SD, analyse, check, load_csv, rows


def score(t, res):
    n = len(t.teams)
    y = np.zeros(n)
    y[t.winner] = 1.0
    p = np.clip(res["mean"], 1e-6, 1 - 1e-6)
    m = np.clip(res["title"], 1e-6, 1 - 1e-6)
    m = m / m.sum()
    brier = np.sum((p - y) ** 2), np.sum((m - y) ** 2)
    logs = -np.log(p[t.winner]), -np.log(m[t.winner])
    return brier, logs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="replays")
    p.add_argument("--fee", type=float, default=FEE)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--prior-sd", type=float, default=PRIOR_SD)
    p.add_argument("--model-game-noise", type=float, default=GAME_NOISE)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    tours = []
    for path in sorted(Path(a.dir).glob("*.csv")):
        tours += load_csv(path)
    if not tours:
        raise SystemExit(f"no CSVs in {a.dir}")
    for t in tours:
        check(t)
        if t.winner is None:
            raise SystemExit(f"{t.name}: no winner row")

    rng = np.random.default_rng(a.seed)
    got = []
    brier, logs = [], []
    for t in tours:
        res = analyse(t, a.draws, a.prior_sd, a.model_game_noise, rng)
        rs = rows(t, res, a.fee, -1, -1)   # every team, no filter
        b, l = score(t, res)
        brier.append(b)
        logs.append(l)
        got.append((t, res, rs))
        print(f"{t.name:16} {len(t.teams):2} teams  "
              f"model top {max(r['model'] for r in rs):.3f}  "
              f"market top {max(r['market'] for r in rs) / sum(r['market'] for r in rs):.3f}")

    n = sum(len(r) for _, _, r in got)
    b = np.sum(brier, axis=0) / n * 100
    l = np.mean(logs, axis=0)
    print(f"\nforecasts: Brier x100 (per team) model {b[0]:.2f}, market {b[1]:.2f}; "
          f"log loss (per tournament) model {l[0]:.4f}, market {l[1]:.4f}")

    print(f"\n{'min edge':>8}{'min sure':>9}{'trades':>7}{'hit':>6}"
          f"{'profit/trade':>14}{'predicted':>10}")
    for me in (0.01, 0.02, 0.03, 0.05):
        for ms in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
            xs = []
            for t, _, rs in got:
                for r in rs:
                    if r["edge"] >= me and r["sure"] >= ms:
                        xs.append((r, profit(t, t.index(r["team"]), r["side"], a.fee)))
            if not xs:
                print(f"{me:8.2f}{ms:9.2f}{0:7}{'-':>6}{'-':>14}{'-':>10}")
                continue
            vals = [v for _, v in xs]
            m, se = mean_se(vals)
            hit = np.mean([v > 0 for v in vals])
            pe = np.mean([r["edge"] for r, _ in xs])
            print(f"{me:8.2f}{ms:9.2f}{len(xs):7}{hit:6.0%}"
                  f"{m * 100:+11.1f}c ±{se * 100:4.1f}"
                  f"{pe * 100:+9.1f}c")

    print("\nper tournament at min edge 3c, min sure 0.6:")
    for t, _, rs in got:
        xs = [(r, profit(t, t.index(r["team"]), r["side"], a.fee))
              for r in rs if r["edge"] >= 0.03 and r["sure"] >= 0.6]
        if xs:
            m, _ = mean_se([v for _, v in xs])
            print(f"  {t.name:16} {len(xs):2} trades  {m * 100:+.1f}c  "
                  f"({', '.join(r['team'] + ' ' + r['side'] for r, _ in xs)})")
        else:
            print(f"  {t.name:16}  0 trades")


if __name__ == "__main__":
    main()
