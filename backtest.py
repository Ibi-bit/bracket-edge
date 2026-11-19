#!/usr/bin/env python3
"""Backtest the bracket edge model on synthetic tournaments or a price CSV.

Synthetic tournaments have known true probabilities, so the report shows
whether flagged trades actually made money. A real CSV can be passed with
--csv once prices and results are logged.
"""
import argparse

import numpy as np

from bracket_edge import (
    FEE, MIN_EDGE, MIN_SURE, DRAWS, PRIOR_SD, GAME_NOISE,
    Tournament, analyse, check, logit, rows, sigmoid, simulate,
)


def synthetic(rng, n=8, title_noise=0.5, game_noise=0.2,
              overround=0.03, title_overround=0.05, prior_sd=0.6):
    """One tournament with known truth; returns it with truth attached."""
    t = Tournament("synthetic")
    for i in range(n):
        t.index(str(i))

    true_r = rng.normal(0.0, 1.0, n)
    game_idx = [(2 * i, 2 * i + 1) for i in range(n // 2)]
    true_p = sigmoid(np.array([true_r[a] - true_r[b] for a, b in game_idx]))

    # Game prices: logit noise, then the platform margin on both sides.
    px = sigmoid(logit(true_p) + rng.normal(0.0, game_noise, len(true_p)))
    pa, pb = px * (1 + overround), (1 - px) * (1 + overround)
    t.games = [(a, b, pa[i], pb[i]) for i, (a, b) in enumerate(game_idx)]

    true_title = simulate(np.array(game_idx), true_p[None, :], true_r[None, :],
                          n, n // 2)[0]
    z = logit(true_title) + (rng.normal(0.0, title_noise, n) if title_noise > 0 else 0.0)
    price = np.clip(sigmoid(z) * (1 + title_overround), 0.01, 0.99)
    t.title = {i: price[i] for i in range(n)}
    t.prior = {i: true_r[i] + rng.normal(0.0, prior_sd) for i in range(n)}
    t.winner = int(rng.choice(n, p=true_title))

    t.truth = true_title
    return t


def profit(t, team, side, fee):
    """Realised profit per $1 contract."""
    won = 1.0 if t.winner == team else 0.0
    if side == "YES":
        return won - t.title[team] - fee
    return t.title[team] - won - fee          # NO costs 1 - YES price


def scores(t, res):
    """Multiclass Brier (summed) and log loss for model, market and truth."""
    n = len(t.teams)
    y = np.zeros(n)
    y[t.winner] = 1.0
    p = np.clip(res["mean"], 1e-6, 1 - 1e-6)
    m = np.clip(res["title"], 1e-6, 1 - 1e-6)
    m = m / m.sum()                       # devig: raw title prices sum to 1.05
    q = np.clip(t.truth, 1e-6, 1 - 1e-6)
    brier = [np.sum((x - y) ** 2) for x in (p, m, q)]
    logs = [-np.log(x[t.winner]) for x in (p, m, q)]
    return brier, logs


def run(tours, a):
    """Analyse tournaments and return trades plus forecast scores."""
    rng = np.random.default_rng(a.seed + 1)
    trades, brier, logs = [], [], []
    for t in tours:
        res = analyse(t, a.draws, a.prior_sd, a.model_game_noise, rng)
        for r in rows(t, res, a.fee, a.min_edge, a.min_sure):
            if r["edge"] < a.min_edge:
                continue
            trades.append({
                "sure": r["sure"],
                "edge": r["edge"],
                "profit": profit(t, t.index(r["team"]), r["side"], a.fee),
                "flagged": r["action"] != "-",
            })
        if getattr(t, "truth", None) is not None and t.winner is not None:
            b, l = scores(t, res)
            brier.append(b)
            logs.append(l)
    return trades, brier, logs


def mean_se(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) <= 1:
        return xs.mean() if len(xs) == 1 else 0.0, 0.0
    return xs.mean(), xs.std(ddof=1) / np.sqrt(len(xs))


def report(name, trades, brier, logs, a):
    flagged = [x for x in trades if x["flagged"]]
    m, se = mean_se([x["profit"] for x in flagged])
    pe = np.mean([x["edge"] for x in flagged]) if flagged else 0.0
    print(f"\n{name}: {len(flagged)} flagged trades, "
          f"{m * 100:+.1f}c per trade (± {se * 100:.1f}c), "
          f"predicted {pe * 100:.1f}c")

    buckets = [(0.5, 0.8), (0.8, 0.9), (0.9, 1.01)]
    print(f"  {'sure':<10}{'trades':>8}{'mean profit':>14}")
    for lo, hi in buckets:
        xs = [x["profit"] for x in trades if lo <= x["sure"] < hi]
        if not xs:
            continue
        bm, bse = mean_se(xs)
        print(f"  {lo:.1f}-{min(hi, 1.0):.1f}   {len(xs):>8}"
              f"{bm * 100:>+11.1f}c ± {bse * 100:.1f}c")

    if brier:
        b = np.mean(brier, axis=0) * 100
        l = np.mean(logs, axis=0)
        print(f"  Brier (x100) model {b[0]:.2f}, market {b[1]:.2f}, truth {b[2]:.2f}")
        print(f"  log loss     model {l[0]:.4f}, market {l[1]:.4f}, truth {l[2]:.4f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", help="logged price/result CSV instead of simulation")
    p.add_argument("--n", type=int, default=None, help="tournaments per scenario")
    p.add_argument("--title-noise", type=float, default=None,
                   help="logit noise in the title market; omit to run both scenarios")
    p.add_argument("--fee", type=float, default=FEE)
    p.add_argument("--min-edge", type=float, default=MIN_EDGE)
    p.add_argument("--min-sure", type=float, default=MIN_SURE)
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--prior-sd", type=float, default=PRIOR_SD)
    p.add_argument("--model-game-noise", type=float, default=GAME_NOISE)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.csv:
        from bracket_edge import load_csv
        tours = load_csv(a.csv)
        for t in tours:
            check(t)
            if t.winner is None:
                raise SystemExit(f"{t.name}: no winner row, so realised profit cannot be scored")
        trades, brier, logs = run(tours, a)
        report(a.csv, trades, brier, logs, a)
        return

    rng = np.random.default_rng(a.seed)
    if a.title_noise is not None:
        scenarios = [(a.title_noise, a.n or 600)]
    else:
        scenarios = [(0.0, a.n or 600), (0.5, a.n or 1200)]
    for noise, n in scenarios:
        tours = [synthetic(rng, 8, noise) for _ in range(n)]
        trades, brier, logs = run(tours, a)
        label = "efficient title market" if noise == 0 else f"title noise {noise:g}"
        report(f"{label} ({n} tournaments)", trades, brier, logs, a)


if __name__ == "__main__":
    main()
