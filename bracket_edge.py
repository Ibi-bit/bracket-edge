#!/usr/bin/env python3
"""Bracket edge model for knockout tournament prediction markets.

Turns linked market prices (next-game and title) into a title probability for
every team, measures how uncertain that probability is, and flags trades whose
edge survives fees and uncertainty.

Only numpy is required. See the companion note for the method.
"""
import argparse
import csv

import numpy as np

FEE = 0.02
MIN_EDGE = 0.03
MIN_SURE = 0.90
DRAWS = 100
PRIOR_SD = 0.6
GAME_NOISE = 0.25
PRIOR_WEIGHT = 1.0
FIT_ITERS = 20


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def devig(pa, pb):
    """Remove the platform margin from a two-way price."""
    s = pa + pb
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"game prices must sum above zero, got {pa} and {pb}")
    return pa / s, pb / s


class Tournament:
    def __init__(self, name):
        self.name = name
        self.teams = []
        self.games = []      # (a, b, pa, pb); first n/2 are round 1, in bracket order
        self.title = {}      # team index -> YES ask
        self.prior = {}      # team index -> rating in logit units
        self.winner = None   # team index, if a result is known
        self._pos = {}       # team name -> index, kept in sync with teams

    def index(self, team):
        i = self._pos.get(team)
        if i is None:
            i = len(self.teams)
            self.teams.append(team)
            self._pos[team] = i
        return i


def load_csv(path):
    """Read one row per game, title, prior or result; one tournament per group."""
    tours, order = {}, []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("tournament") or "").strip()
            if not name:
                continue
            if name not in tours:
                tours[name] = Tournament(name)
                order.append(name)
            t = tours[name]
            typ = (row.get("type") or "").strip().lower()
            a = (row.get("a") or "").strip()
            b = (row.get("b") or "").strip()
            pa = (row.get("pa") or "").strip()
            pb = (row.get("pb") or "").strip()
            if typ == "game":
                t.games.append((t.index(a), t.index(b), float(pa), float(pb)))
            elif typ == "title":
                t.title[t.index(a)] = float(pa)
            elif typ == "prior":
                t.prior[t.index(a)] = float(pa)
            elif typ == "winner":
                t.winner = t.index(a)
            else:
                raise ValueError(f"{path}: unknown type {typ!r}")
    return [tours[n] for n in order]


def check(t):
    """Validate one tournament and return (n, games, title prices, priors)."""
    n = len(t.teams)
    if n < 2 or n & (n - 1):
        raise ValueError(f"{t.name}: team count must be a power of 2, got {n}")
    if len(t.games) < n // 2:
        raise ValueError(f"{t.name}: need {n // 2} first-round games, got {len(t.games)}")
    seen = set()
    for a, b, pa, pb in t.games:
        if not (np.isfinite(pa) and np.isfinite(pb)) or not (0 < pa < 1 and 0 < pb < 1):
            raise ValueError(f"{t.name}: game prices must be between 0 and 1, got {pa} and {pb}")
        if pa + pb <= 0:
            raise ValueError(f"{t.name}: game prices must sum above zero, got {pa} and {pb}")
    for a, b, _, _ in t.games[: n // 2]:
        if a in seen or b in seen:
            raise ValueError(f"{t.name}: first-round games must use each team once")
        seen.update((a, b))
    if len(seen) != n:
        raise ValueError(f"{t.name}: first-round games must cover every team")
    missing = [t.teams[i] for i in range(n) if i not in t.title]
    if missing:
        raise ValueError(f"{t.name}: no title price for {', '.join(missing)}")
    title = np.array([t.title[i] for i in range(n)], dtype=float)
    if not np.all(np.isfinite(title)) or np.any(title <= 0) or np.any(title >= 1):
        raise ValueError(f"{t.name}: title prices must be between 0 and 1")
    if title.sum() <= 0:
        raise ValueError(f"{t.name}: title prices must sum above zero")
    prior = np.array([t.prior.get(i, 0.0) for i in range(n)], dtype=float)
    if not np.all(np.isfinite(prior)):
        raise ValueError(f"{t.name}: priors must be finite numbers")
    return n, t.games, title, prior


def fit_ratings(game_idx, game_p, prior, prior_weight=PRIOR_WEIGHT, iters=FIT_ITERS):
    """Fit a rating per team so rating gaps match devigged game probabilities.

    game_p is (runs, games); prior is (runs, teams). Returns (runs, teams).
    Logistic loss plus a pull toward the prior.
    """
    runs, n = prior.shape
    games = len(game_idx)
    if games == 0:
        return prior.copy()
    sign = np.zeros((n, games))
    for g, (a, b) in enumerate(game_idx):
        sign[a, g] = 1.0
        sign[b, g] = -1.0
    ridge = prior_weight * np.eye(n)[None, :, :]
    r = prior.copy()
    for _ in range(iters):
        x = r @ sign                       # r_a - r_b
        s = sigmoid(x)
        w = s * (1.0 - s)
        grad = (s - game_p) @ sign.T + prior_weight * (r - prior)
        hess = np.einsum("ig,tg,jg->tij", sign, w, sign) + ridge
        r = r - np.linalg.solve(hess, grad[..., None])[..., 0]
    return r


def simulate(game_idx, game_p, ratings, n, round1):
    """Exact bracket probability per team. Shapes: ratings (runs, n)."""
    runs = ratings.shape[0]
    matchup = sigmoid(ratings[:, :, None] - ratings[:, None, :])
    for g, (a, b) in enumerate(game_idx):
        matchup[:, a, b] = game_p[:, g]
        matchup[:, b, a] = 1.0 - game_p[:, g]

    adv = np.zeros((runs, n))
    slots = []
    for g in range(round1):
        a, b = game_idx[g]
        adv[:, a] = game_p[:, g]
        adv[:, b] = 1.0 - game_p[:, g]
        slots.append([a, b])
    while len(slots) > 1:
        nxt = np.zeros((runs, n))
        joined = []
        for k in range(0, len(slots), 2):
            s1, s2 = slots[k], slots[k + 1]
            for i in s1:
                nxt[:, i] = adv[:, i] * np.sum(adv[:, s2] * matchup[:, i][:, s2], axis=1)
            for j in s2:
                nxt[:, j] = adv[:, j] * np.sum(adv[:, s1] * matchup[:, j][:, s1], axis=1)
            joined.append(s1 + s2)
        adv, slots = nxt, joined
    return adv


def analyse(t, draws=DRAWS, prior_sd=PRIOR_SD, game_noise=GAME_NOISE, rng=None):
    """Run the model with jittered inputs; return per-team title statistics."""
    n, games, title, prior = check(t)
    rng = rng if rng is not None else np.random.default_rng()
    game_idx = [(a, b) for a, b, _, _ in games]
    base = np.array([devig(pa, pb)[0] for _, _, pa, pb in games])

    jittered = sigmoid(logit(base)[None, :] + rng.normal(0.0, game_noise, (draws, len(games))))
    priors = prior[None, :] + rng.normal(0.0, prior_sd, (draws, n))
    ratings = fit_ratings(game_idx, jittered, priors)
    titles = simulate(game_idx, jittered, ratings, n, n // 2)

    return {
        "title": title,
        "runs": titles,
        "mean": titles.mean(axis=0),
        "sd": titles.std(axis=0),
        "lo": np.percentile(titles, 5, axis=0),
        "hi": np.percentile(titles, 95, axis=0),
    }


def rows(t, res, fee=FEE, min_edge=MIN_EDGE, min_sure=MIN_SURE):
    """One dict per team with the better side, edge, confidence and action."""
    out = []
    for i, team in enumerate(t.teams):
        price = res["title"][i]
        model = res["mean"][i]
        yes = model - price - fee
        no = price - model - fee
        if yes >= no:
            side, edge = "YES", yes
            sure = float(np.mean(res["runs"][:, i] > price + fee))
        else:
            side, edge = "NO", no
            sure = float(np.mean(res["runs"][:, i] < price - fee))
        action = side if edge >= min_edge and sure >= min_sure else "-"
        out.append({
            "team": team, "model": model, "sd": res["sd"][i],
            "lo": res["lo"][i], "hi": res["hi"][i], "market": price,
            "edge": edge, "sure": sure, "side": side, "action": action,
        })
    return out


def print_table(t, res, fee=FEE, min_edge=MIN_EDGE, min_sure=MIN_SURE):
    print(f"\n{t.name}  ({len(t.teams)} teams)")
    print(f"{'team':<14}{'model':>7}{'sd':>7}{'5%':>7}{'95%':>7}"
          f"{'market':>8}{'edge':>7}{'sure':>7}  action")
    for r in rows(t, res, fee, min_edge, min_sure):
        print(f"{r['team']:<14}{r['model']:>7.3f}{r['sd']:>7.3f}"
              f"{r['lo']:>7.3f}{r['hi']:>7.3f}{r['market']:>8.3f}"
              f"{r['edge']:>7.3f}{r['sure']:>7.2f}  {r['action']}")


def sample():
    """The four-team example from the note."""
    t = Tournament("sample")
    for team in "ABCD":
        t.index(team)
    t.games = [(0, 1, 0.70, 0.32), (2, 3, 0.55, 0.47)]
    t.title = {0: 0.34, 1: 0.22, 2: 0.26, 3: 0.18}
    t.prior = {0: 1.0, 1: 0.3, 2: 0.2, 3: -0.2}
    return t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", help="one row per game, title, prior or result")
    p.add_argument("--fee", type=float, default=FEE)
    p.add_argument("--min-edge", type=float, default=MIN_EDGE)
    p.add_argument("--min-sure", type=float, default=MIN_SURE)
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--prior-sd", type=float, default=PRIOR_SD)
    p.add_argument("--model-game-noise", type=float, default=GAME_NOISE)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    tours = load_csv(a.csv) if a.csv else [sample()]
    for t in tours:
        res = analyse(t, a.draws, a.prior_sd, a.model_game_noise, rng)
        print_table(t, res, a.fee, a.min_edge, a.min_sure)


if __name__ == "__main__":
    main()
